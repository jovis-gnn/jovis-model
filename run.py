import os

# import pdb
import importlib
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from jovis_model.config import Config
from jovis_model.utils.helper import init_logger
from jovis_model.utils.module import ModelModules
from jovis_model.utils.train_utils import get_policies
from jovis_model.data import DataModule


class ModelRunner:
    def __init__(self, config: Config):
        self.logger = init_logger("runner")
        self.config = config
        self.get_dataloader()
        self.get_model()

    def get_dataloader(self):
        cdm = DataModule(self.config)
        train_dataloader = cdm.train_dataloader()
        self.config.params.num_labels = len(cdm.processor.get_labels())
        self.config.params.dataset_size = len(train_dataloader)

        return train_dataloader

    def get_model(self):
        module_name = ModelModules[f"{self.config.pkg}_{self.config.task}"]
        module, class_ = module_name.rsplit(".", 1)
        module = importlib.import_module(module)
        model = getattr(module, class_)(self.config, {})

        return model

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
        model,
        train_dataloader,
        optimizer,
        lr_scheduler,
        config,
        logger,
        local_rank=None,
        rank=None,
    ):
        if config.params.use_fp16 and config.params.enable_fsdp:
            scaler = ShardedGradScaler()
        elif config.params.use_fp16 and not config.params.enable_fsdp:
            scaler = torch.cuda.amp.GradScaler()
        if config.params.enable_fsdp:
            world_size = int(os.environ["WORLD_SIZE"])

        autocast = torch.cuda.amp.autocast if config.params.use_fp16 else nullcontext

        train_loss = []

        for epoch in range(config.params.num_train_epochs):
            model.model.train()
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
                    outputs = model.training_step(batch, step)
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

    def run(self, job="train"):
        if job == "train":
            if self.config.params.enable_fsdp:
                dist.init_process_group("nccl", init_method="env://")
                local_rank = int(os.environ["LOCAL_RANK"])
                rank = int(os.environ["RANK"])
            if torch.distributed.is_initialized():
                torch.cuda.set_device(local_rank)

            train_dataloader = self.get_dataloader()
            model = self.get_model()
            # pdb.set_trace()
            device_id = 0
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
            if self.config.params.enable_fsdp:
                mixed_precision_policy, wrapping_policy = get_policies(config)
                model.model = FSDP(
                    model.model,
                    auto_wrap_policy=wrapping_policy,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    device_id=device_id,
                    limit_all_gathers=True,
                )
            else:
                model.model.to("cuda")
            optimizer, lr_scheduler = self.get_optimizers(model.model)
            self.train(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                config=self.config,
                logger=self.logger,
                local_rank=local_rank if self.config.params.enable_fsdp else None,
                rank=rank if self.config.params.enable_fsdp else None,
            )


if __name__ == "__main__":
    from transformers.utils import logging as hf_logging
    from huggingface_hub import logging as hf_hub_logging

    hf_logging.set_verbosity_error()
    hf_hub_logging.set_verbosity_error()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    params = {
        "pkg": "klue",
        "task": "ynat",
        "data_dir": "/home/omnious/workspace/jovis/jovis-model/jovis_model/_db/klue/ynat-v1.1",
        "train_file_name": "ynat-v1.1_train.json",
        "dev_file_name": "ynat-v1.1_dev.json",
        "output_dir": "/home/omnious/workspace/jovis/jovis-model/outputs",
        "params": {"enable_fsdp": False},
    }
    config = Config(**params)
    runner = ModelRunner(config)
    runner.run(job="train")
