import os
import random
import argparse
import warnings

# import pdb
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from accelerate.utils import is_xpu_available
from peft import get_peft_model, prepare_model_for_kbit_training

from jovis_model.config import Config
from jovis_model.utils.helper import init_logger
from jovis_model.utils.train_utils import (
    get_policies,
    MemoryTrace,
    generate_peft_config,
)
from jovis_model.module import DataModule, ModelModule


class ModelRunner:
    def __init__(self, config: Config, mode: str = "train") -> None:
        self.logger = init_logger("runner")
        self.config = config
        self.mode = mode

        if is_xpu_available():
            torch.xpu.manual_seed(self.config.params.seed)
        torch.manual_seed(self.config.params.seed)
        random.seed(self.config.params.seed)

        if self.config.task != "bedrock":
            if self.config.params.enable_fsdp:
                dist.init_process_group("nccl")
                self.local_rank = int(os.environ["LOCAL_RANK"])
                self.rank = int(os.environ["RANK"])
            if torch.distributed.is_initialized():
                if is_xpu_available():
                    torch.xpu.set_device(self.local_rank)
                    torch.xpu_empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.set_device(self.local_rank)
                    torch.cuda.empty_cache()
            self.device_id = "cpu"
            if is_xpu_available():
                self.device_id = "xpu:0"
            elif torch.cuda.is_available():
                self.device_id = f"cuda:{torch.cuda.current_device()}"

        self.dm: DataModule = self.get_data_module()
        if self.config.params.enable_fsdp:
            if self.local_rank == 0:
                self.logger.info("data module loaded")
        else:
            self.logger.info("data module loaded")

        self.mm: ModelModule = self.get_model_module()
        if self.config.params.enable_fsdp:
            if self.local_rank == 0:
                self.logger.info("model module loaded")
        else:
            self.logger.info("model module loaded")

    def get_data_module(self) -> DataModule:
        dm = DataModule(self.config)
        if self.mode == "train":
            if getattr(dm.processor, "get_labels", None):
                self.config.params.num_labels = len(dm.processor.get_labels())
            for mode_ in ["train", "eval"]:
                if getattr(self.config, f"{mode_}_file_name", None):
                    if self.config.params.enable_fsdp:
                        kwargs = {}
                        dataset = dm.prepare_dataset(mode_)
                        kwargs["sampler"] = torch.utils.data.DistributedSampler(
                            dataset,
                            rank=dist.get_rank(),
                            num_replicas=dist.get_world_size(),
                            shuffle=mode_ == "train",
                        )
                        kwargs["batch_size"] = getattr(
                            self.config.params, f"{mode_}_batch_size", None
                        )
                        assert (
                            kwargs["batch_size"] is not None
                        ), f"There's no {mode_}_batch_size in params."
                        dataloader = torch.utils.data.DataLoader(
                            dataset,
                            num_workers=self.config.params.num_workers,
                            pin_memory=True,
                            **kwargs,
                        )
                    else:
                        batch_size = getattr(
                            self.config.params, f"{mode_}_batch_size", None
                        )
                        assert (
                            batch_size is not None
                        ), f"There's no {self.mode}_batch_size in params."
                        dataloader = dm.get_dataloader(
                            dataset_type=self.mode,
                            batch_size=batch_size,
                            shuffle=mode_ == "train",
                        )
                    setattr(dm, f"{mode_}_dataloader", dataloader)
                else:
                    setattr(dm, f"{mode_}_dataloader", None)
        return dm

    def get_model_module(self) -> ModelModule:
        mm = ModelModule(self.config)
        if self.config.task != "bedrock":
            if self.config.params.quantization:
                mm.processor.model = prepare_model_for_kbit_training(mm.processor.model)

            if self.config.params.use_fp16:
                mm.processor.model = mm.processor.model.to(torch.bfloat16)

            if self.config.params.use_peft:
                peft_config = generate_peft_config(self.config)
                mm.processor.model = get_peft_model(mm.processor.model, peft_config)

            if self.config.params.enable_fsdp:
                mixed_precision_policy, wrapping_policy = get_policies(self.config)
                mm.processor.model = FSDP(
                    mm.processor.model,
                    auto_wrap_policy=wrapping_policy,
                    cpu_offload=CPUOffload(offload_params=True),
                    mixed_precision=mixed_precision_policy,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    device_id=int(self.device_id.split(":")[-1]),
                    limit_all_gathers=True,
                )
            else:
                mm.processor.model.to(self.device_id)
        return mm

    def get_optimizers(self, model):
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.params.learning_rate,
            weight_decay=self.config.params.weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        return optimizer, scheduler

    def eval(
        self,
        epoch,
        model_processor,
        eval_dataloader,
        config,
        logger,
        local_rank=None,
        rank=None,
    ) -> None:
        if config.params.enable_fsdp:
            world_size = int(os.environ["WORLD_SIZE"])

        with MemoryTrace() as memtrace:
            model_processor.model.eval()

            total_loss = 0.0
            total_preds = []
            total_labels = []

            pbar = tqdm(
                colour="green",
                desc=f"Evaluating Epoch: {epoch+1}",
                total=len(eval_dataloader),
                dynamic_ncols=True,
            )
            for step, batch in enumerate(eval_dataloader):
                for idx_ in range(len(batch)):
                    if config.params.enable_fsdp:
                        batch[idx_] = batch[idx_].to(local_rank)
                    else:
                        batch[idx_] = batch[idx_].to("cuda:0")

                with torch.no_grad():
                    outputs = model_processor.validation_step(batch)
                    loss = outputs["loss"]
                    total_loss += loss.detach().float()
                    if getattr(model_processor, "metric", None):
                        preds, labels = model_processor.convert_outputs(outputs)
                        total_preds.append(preds)
                        total_labels.append(labels)

                pbar.update(1)
                pbar.set_description(
                    f"Evaluating Epoch: {epoch + 1}/{config.params.num_train_epochs}, step {step}/{len(eval_dataloader)} completed (loss: {loss.detach().float():.4f})"
                )
            pbar.close()

        if torch.cuda.device_count() > 1 and config.params.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        eval_epoch_loss = total_loss / len(eval_dataloader)
        if config.params.enable_fsdp:
            eval_epoch_loss = eval_epoch_loss / world_size

        if getattr(model_processor, "metric", None):
            total_preds = torch.cat(total_preds, dim=0)
            total_labels = torch.cat(total_labels, dim=0)

            if torch.cuda.device_count() > 1 and config.params.enable_fsdp:
                total_preds_global = [
                    torch.ones_like(total_preds) for _ in range(dist.get_world_size())
                ]
                total_labels_global = [
                    torch.ones_like(total_labels) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(total_preds_global, total_preds)
                dist.all_gather(total_labels_global, total_labels)
                total_preds_global = torch.cat(total_preds_global, dim=0)
                total_labels_global = torch.cat(total_labels_global, dim=0)
                total_preds = total_preds_global
                total_labels = total_labels_global

            total_preds = total_preds.cpu().numpy()
            total_labels = total_labels.cpu().numpy()
            metrics = model_processor.get_metric(total_preds, total_labels)
        else:
            metrics = {}
        if config.params.enable_fsdp:
            if rank == 0:
                logger.info(f"Epoch {epoch+1}: eval_epoch_loss : {eval_epoch_loss:.4f}")
                for metric, value in metrics.items():
                    logger.info(f"Epoch {epoch+1}: eval_epoch_{metric} : {value:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}: eval_epoch_loss : {eval_epoch_loss:.4f}")
            for metric, value in metrics.items():
                logger.info(f"Epoch {epoch+1}: eval_epoch_{metric}: {value:.4f}")

        if not config.params.enable_fsdp or rank == 0:
            memtrace.print_stats()

    def train(
        self,
        model_processor,
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
        config,
        logger,
        local_rank=None,
        rank=None,
    ) -> None:
        if config.params.use_fp16 and config.params.enable_fsdp:
            scaler = ShardedGradScaler()
        elif config.params.use_fp16 and not config.params.enable_fsdp:
            scaler = torch.cuda.amp.GradScaler()
        if config.params.enable_fsdp:
            world_size = int(os.environ["WORLD_SIZE"])

        autocast = torch.cuda.amp.autocast if config.params.use_fp16 else nullcontext

        if eval_dataloader:
            self.eval(
                epoch=0,
                model_processor=model_processor,
                eval_dataloader=eval_dataloader,
                config=config,
                logger=logger,
                local_rank=local_rank,
                rank=rank,
            )
        for epoch in range(config.params.num_train_epochs):
            with MemoryTrace() as memtrace:
                model_processor.model.train()
                total_loss = 0.0
                total_length = (
                    len(train_dataloader) // config.params.gradient_accumulation_steps
                )
                pbar = tqdm(
                    colour="blue",
                    desc=f"Training Epoch: {epoch+1}",
                    total=total_length,
                    dynamic_ncols=True,
                )
                for step, batch in enumerate(train_dataloader):
                    for idx_ in range(len(batch)):
                        if config.params.enable_fsdp:
                            if is_xpu_available():
                                batch[idx_] = batch[idx_].to(
                                    torch.device(f"xpu:{local_rank}")
                                )
                            else:
                                batch[idx_] = batch[idx_].to(local_rank)
                        else:
                            batch[idx_] = batch[idx_].to(self.device_id)
                    with autocast():
                        outputs = model_processor.training_step(batch)
                        loss = outputs["loss"]
                    total_loss += loss.detach().float()
                    if config.params.use_fp16:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    pbar.update(1)
                    pbar.set_description(
                        f"Training Epoch: {epoch+1}/{config.params.num_train_epochs} step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float():.4f})"
                    )
                pbar.close()

            if torch.cuda.device_count() > 1 and config.params.enable_fsdp:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            train_epoch_loss = total_loss / len(train_dataloader)
            if config.params.enable_fsdp:
                train_epoch_loss = train_epoch_loss / world_size

            if config.params.enable_fsdp:
                if rank == 0:
                    logger.info(
                        f"Epoch {epoch+1}: train_epoch_loss : {train_epoch_loss:.4f}"
                    )
            else:
                logger.info(
                    f"Epoch {epoch+1}: train_epoch_loss : {train_epoch_loss:.4f}"
                )
            if not config.params.enable_fsdp or rank == 0:
                memtrace.print_stats()

            if eval_dataloader:
                self.eval(
                    epoch=epoch,
                    model_processor=model_processor,
                    eval_dataloader=eval_dataloader,
                    config=config,
                    logger=logger,
                    local_rank=local_rank,
                    rank=rank,
                )

    def inference(self, model_processor, data_processor, sample_inputs):
        sample_inputs = data_processor._convert_features(sample_inputs)
        if self.config.task != "bedrock":
            sample_inputs = sample_inputs.to(self.device_id)
        outputs = model_processor.inference(sample_inputs)

        return outputs

    def run(self, sample_inputs=None):
        outputs = None
        if self.mode == "train":
            optimizer, lr_scheduler = self.get_optimizers(self.mm.processor.model)
            self.train(
                model_processor=self.mm.processor,
                train_dataloader=self.dm.train_dataloader,
                eval_dataloader=self.dm.eval_dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                config=self.config,
                logger=self.logger,
                local_rank=self.local_rank if self.config.params.enable_fsdp else None,
                rank=self.rank if self.config.params.enable_fsdp else None,
            )
        if self.mode == "inference":
            assert (
                sample_inputs is not None
            ), "please specify target inputs in inference mode."
            outputs = self.inference(
                model_processor=self.mm.processor,
                data_processor=self.dm.processor,
                sample_inputs=sample_inputs,
            )
        return outputs


if __name__ == "__main__":
    from transformers.utils import logging as hf_logging
    from huggingface_hub.utils import logging as hf_hub_logging

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()
    hf_hub_logging.set_verbosity_error()

    parser = argparse.ArgumentParser(
        description="runner paraser", argument_default=argparse.SUPPRESS
    )
    base_fields = Config.model_fields
    for name, field in base_fields.items():
        parser.add_argument(
            f"--{name}", type=field.annotation, required=field.is_required()
        )
    config, extra = parser.parse_known_args()
    config = vars(config)
    runner_mode = None
    conv_dict = {"true": True, "false": False}
    params = {}
    for k, v in zip(extra[0::2], extra[1::2]):
        if k[2:] == "mode":
            runner_mode = v
        else:
            if v in list(conv_dict.keys()):
                v = conv_dict[v]
            params[k[2:]] = v
    config["params"] = params
    config = Config(**config)

    runner = ModelRunner(config, mode=runner_mode)
    runner.run()
