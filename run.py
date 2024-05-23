import os

# import pdb
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from jovis_model.config import Config
from jovis_model.utils.helper import init_logger
from jovis_model.utils.train_utils import get_policies
from jovis_model.module import DataModule, ModelModule


class ModelRunner:
    def __init__(self, config: Config, mode: str = "train") -> None:
        self.logger = init_logger("runner")
        self.config = config
        self.mode = mode
        if self.config.task != "bedrock":
            if self.config.params.enable_fsdp:
                dist.init_process_group("nccl")
                self.local_rank = int(os.environ["LOCAL_RANK"])
                self.rank = int(os.environ["RANK"])
            if torch.distributed.is_initialized():
                torch.cuda.set_device(self.local_rank)
                torch.cuda.empty_cache()
            self.device_id = 0
            if torch.cuda.is_available():
                self.device_id = torch.cuda.current_device()
        self.dm: DataModule = self.get_data()
        self.mm: ModelModule = self.get_model()

    def get_data(self) -> DataModule:
        dm = DataModule(self.config)
        if self.mode in ["train", "eval"]:
            self.config.params.num_labels = len(dm.processor.get_labels())
            if self.config.params.enable_fsdp:
                kwargs = {}
                dataset = dm.prepare_dataset(self.mode)
                kwargs["sampler"] = torch.utils.data.DistributedSampler(
                    dataset,
                    rank=dist.get_rank(),
                    num_replicas=dist.get_world_size(),
                    shuffle=self.mode == "train",
                )
                kwargs["batch_size"] = getattr(
                    self.config.params, f"{self.mode}_batch_size", None
                )
                assert (
                    kwargs["batch_size"] is not None
                ), f"There's no {self.mode}_batch_size in params."
                # kwargs["collate_fn"] = default_data_collator
                train_dataloader = torch.utils.data.DataLoader(
                    dataset,
                    num_workers=self.config.params.num_workers,
                    pin_memory=True,
                    **kwargs,
                )
            else:
                batch_size = getattr(
                    self.config.params, f"{self.mode}_batch_size", None
                )
                assert (
                    batch_size is not None
                ), f"There's no {self.mode}_batch_size in params."
                train_dataloader = dm.get_dataloader(
                    dataset_type=self.mode,
                    batch_size=batch_size,
                    shuffle=self.mode == "train",
                )
            dm.train_dataloader = train_dataloader
        return dm

    def get_model(self) -> ModelModule:
        mm = ModelModule(self.config)
        if self.config.task != "bedrock":
            if self.config.params.enable_fsdp:
                mixed_precision_policy, wrapping_policy = get_policies(config)
                mm.processor.model = FSDP(
                    mm.processor.model,
                    auto_wrap_policy=wrapping_policy,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    device_id=self.device_id,
                    limit_all_gathers=True,
                )
            else:
                mm.processor.model.to(f"cuda:{self.device_id}")
        return mm

    def get_optimizers(self, model):
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.params.learning_rate,
            weight_decay=self.config.params.weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        return optimizer, scheduler

    def train(
        self,
        model_processor,
        train_dataloader,
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

        train_loss = []

        for epoch in range(config.params.num_train_epochs):
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
                        batch[idx_] = batch[idx_].to(local_rank)
                    else:
                        batch[idx_] = batch[idx_].to("cuda:0")
                with autocast():
                    outputs = model_processor.training_step(batch, step)
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
                    f"Training Epoch: {epoch+1}/{config.params.num_train_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
                )
            pbar.close()

            if torch.cuda.device_count() > 1 and config.params.enable_fsdp:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            train_epoch_loss = total_loss / len(train_dataloader)
            if config.params.enable_fsdp:
                train_epoch_loss = train_epoch_loss / world_size
            train_loss.append(train_epoch_loss)

            if config.params.enable_fsdp:
                if rank == 0:
                    logger.info(
                        f"Epoch {epoch+1}: train_epoch_loss : {train_epoch_loss}"
                    )
            else:
                logger.info(f"Epoch {epoch+1}: train_epoch_loss : {train_epoch_loss}")

    def inference(self, model_processor, data_processor, sample_inputs):
        sample_inputs = data_processor._convert_features(sample_inputs)
        if self.config.task != "bedrock":
            sample_inputs = sample_inputs.to(f"cuda:{self.device_id}")
        outputs = model_processor.inference(sample_inputs)

        return outputs

    def run(self, sample_inputs=None):
        outputs = None
        if self.mode == "train":
            optimizer, lr_scheduler = self.get_optimizers(self.mm.processor.model)
            self.train(
                model_processor=self.mm.processor,
                train_dataloader=self.dm.train_dataloader,
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
    import warnings
    from transformers.utils import logging as hf_logging
    from huggingface_hub.utils import logging as hf_hub_logging

    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()
    hf_hub_logging.set_verbosity_error()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    params = {
        "pkg": "klue",
        "task": "ynat",
        "use_hf_model": True,
        "data_dir": "/home/omnious/workspace/jovis/jovis-model/jovis_model/_db/klue/ynat-v1.1",
        "train_file_name": "ynat-v1.1_train.json",
        "dev_file_name": "ynat-v1.1_dev.json",
        "output_dir": "/home/omnious/workspace/jovis/jovis-model/outputs",
        "params": {
            "enable_fsdp": True,
            "use_fp16": True,
            "mixed_precision": True,
        },
    }
    config = Config(**params)
    runner = ModelRunner(config, mode="train")
    runner.run()
