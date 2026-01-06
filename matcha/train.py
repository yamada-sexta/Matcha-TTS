"""
Plain PyTorch training script for Matcha-TTS.
Replaces the PyTorch Lightning trainer with a simple training loop.
"""
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from matcha import utils
from matcha.data.text_mel_datamodule import TextMelDataModule
from matcha.models.matcha_tts import MatchaTTS
from matcha.utils.utils import plot_tensor

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(cfg: DictConfig) -> MatchaTTS:
    """Create the Matcha-TTS model from config."""
    # Extract model parameters from config
    model_cfg = cfg.model

    # Get encoder, decoder, and cfm configs
    encoder_cfg = OmegaConf.to_container(model_cfg.encoder, resolve=True)
    decoder_cfg = OmegaConf.to_container(model_cfg.decoder, resolve=True)
    cfm_cfg = OmegaConf.to_container(model_cfg.cfm, resolve=True)

    # Convert encoder to namespace-like object (it uses attribute access)
    from types import SimpleNamespace

    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    encoder = dict_to_namespace(encoder_cfg)
    # decoder and cfm stay as dicts because they're unpacked with ** in the model
    decoder = decoder_cfg
    cfm = dict_to_namespace(cfm_cfg)

    model = MatchaTTS(
        n_vocab=model_cfg.n_vocab,
        n_spks=model_cfg.n_spks,
        spk_emb_dim=model_cfg.spk_emb_dim,
        n_feats=model_cfg.n_feats,
        encoder=encoder,
        decoder=decoder,
        cfm=cfm,
        data_statistics=OmegaConf.to_container(model_cfg.data_statistics, resolve=True) if model_cfg.data_statistics else None,
        out_size=model_cfg.out_size,
        prior_loss=model_cfg.get("prior_loss", True),
        use_precomputed_durations=model_cfg.get("use_precomputed_durations", False),
    )

    return model


def create_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    optimizer_cfg = cfg.model.optimizer
    optimizer_class = getattr(torch.optim, optimizer_cfg._target_.split(".")[-1])
    optimizer = optimizer_class(
        model.parameters(),
        lr=optimizer_cfg.lr,
        weight_decay=optimizer_cfg.get("weight_decay", 0.0),
    )
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    """Create learning rate scheduler if configured."""
    if not cfg.model.get("scheduler"):
        return None

    scheduler_cfg = cfg.model.scheduler
    if scheduler_cfg.get("scheduler"):
        scheduler_class_name = scheduler_cfg.scheduler._target_.split(".")[-1]
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_class_name, None)
        if scheduler_class:
            scheduler_container = OmegaConf.to_container(scheduler_cfg.scheduler, resolve=True)
            if isinstance(scheduler_container, dict):
                scheduler_params = {k: v for k, v in scheduler_container.items()
                                  if k != "_target_" and k != "_partial_"}
                return scheduler_class(optimizer, **scheduler_params)
    return None


def train_step(model: MatchaTTS, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Perform a single training step."""
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Get losses
    loss_dict = model.get_losses(batch)
    total_loss: torch.Tensor = sum(loss_dict.values())  # type: ignore[assignment]

    return {"loss": total_loss, **loss_dict}


@torch.no_grad()
def validate(model: MatchaTTS, val_loader, device: torch.device) -> Dict[str, float]:
    """Run validation and return average losses."""
    model.eval()
    total_losses: Dict[str, float] = {"dur_loss": 0.0, "prior_loss": 0.0, "diff_loss": 0.0, "loss": 0.0}
    num_batches = 0

    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        loss_dict = model.get_losses(batch)
        total_loss: torch.Tensor = sum(loss_dict.values())  # type: ignore[assignment]

        total_losses["loss"] += total_loss.item()
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                total_losses[k] += v.item()
            else:
                total_losses[k] += v
        num_batches += 1

    # Average losses
    avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
    model.train()
    return avg_losses


def log_validation_images(
    model: MatchaTTS,
    val_loader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
) -> None:
    """Log validation images to tensorboard."""
    model.eval()
    try:
        one_batch = next(iter(val_loader))

        # Log original samples on first epoch
        if epoch == 0:
            log.info("Plotting original samples")
            for i in range(min(2, one_batch["y"].shape[0])):
                y = one_batch["y"][i].unsqueeze(0).to(device)
                writer.add_image(
                    f"original/{i}",
                    plot_tensor(y.squeeze().cpu()),
                    epoch,
                    dataformats="HWC",
                )

        log.info("Synthesising validation samples...")
        for i in range(min(2, one_batch["x"].shape[0])):
            x = one_batch["x"][i].unsqueeze(0).to(device)
            x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(device)
            spks = one_batch["spks"][i].unsqueeze(0).to(device) if one_batch["spks"] is not None else None

            output = model.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=10, spks=spks)
            y_enc, y_dec = output["encoder_outputs"], output["decoder_outputs"]
            attn = output["attn"]

            writer.add_image(
                f"generated_enc/{i}",
                plot_tensor(y_enc.squeeze().cpu()),
                epoch,
                dataformats="HWC",
            )
            writer.add_image(
                f"generated_dec/{i}",
                plot_tensor(y_dec.squeeze().cpu()),
                epoch,
                dataformats="HWC",
            )
            writer.add_image(
                f"alignment/{i}",
                plot_tensor(attn.squeeze().cpu()),
                epoch,
                dataformats="HWC",
            )
    except Exception as e:
        log.warning(f"Failed to log validation images: {e}")
    finally:
        model.train()


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Main training function."""
    # Set seed
    if cfg.get("seed"):
        set_seed(cfg.seed)

    # Setup output directory
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create run directory with timestamp
    run_name = cfg.get("run_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Output directory: {run_dir}")

    # Save config
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        log.info("Using CPU")

    # Create data module
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = TextMelDataModule(
        name=cfg.data.name,
        train_filelist_path=cfg.data.train_filelist_path,
        valid_filelist_path=cfg.data.valid_filelist_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        cleaners=cfg.data.cleaners,
        add_blank=cfg.data.add_blank,
        n_spks=cfg.data.n_spks,
        n_fft=cfg.data.n_fft,
        n_feats=cfg.data.n_feats,
        sample_rate=cfg.data.sample_rate,
        hop_length=cfg.data.hop_length,
        win_length=cfg.data.win_length,
        f_min=cfg.data.f_min,
        f_max=cfg.data.f_max,
        data_statistics=OmegaConf.to_container(cfg.data.data_statistics, resolve=True) if cfg.data.data_statistics else None,
        seed=cfg.seed,
        load_durations=cfg.data.load_durations,
    )
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    assert datamodule.trainset is not None, "Train dataset not initialized"
    assert datamodule.validset is not None, "Validation dataset not initialized"
    log.info(f"Train dataset size: {len(datamodule.trainset)}")
    log.info(f"Val dataset size: {len(datamodule.validset)}")

    # Create model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = create_model(cfg)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)

    # Load checkpoint if provided
    start_epoch = 0
    global_step = 0
    if cfg.get("ckpt_path"):
        log.info(f"Loading checkpoint from {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        log.info(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Setup tensorboard
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    # Training settings
    trainer_cfg = cfg.trainer
    max_epochs = trainer_cfg.get("max_epochs", 1000)
    val_check_interval = trainer_cfg.get("val_check_interval", 1.0)
    log_every_n_steps = trainer_cfg.get("log_every_n_steps", 50)
    gradient_clip_val = trainer_cfg.get("gradient_clip_val", None)

    # Calculate validation frequency
    if isinstance(val_check_interval, float) and val_check_interval <= 1.0:
        val_every_n_batches = int(len(train_loader) * val_check_interval)
    else:
        val_every_n_batches = int(val_check_interval)
    val_every_n_batches = max(1, val_every_n_batches)

    log.info(f"Starting training for {max_epochs} epochs")
    log.info(f"Validation every {val_every_n_batches} batches")

    best_val_loss = float("inf")
    model.train()

    # Initialize to avoid uninitialized variable errors
    avg_epoch_loss: float = 0.0
    val_losses: Dict[str, float] = {"dur_loss": 0.0, "prior_loss": 0.0, "diff_loss": 0.0, "loss": 0.0}

    for epoch in range(start_epoch, max_epochs):
        epoch_losses: Dict[str, float] = {"dur_loss": 0.0, "prior_loss": 0.0, "diff_loss": 0.0, "loss": 0.0}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            # Training step
            loss_dict = train_step(model, batch, device)
            loss = loss_dict["loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            optimizer.step()

            # Update epoch losses
            epoch_losses["loss"] += loss.item()
            for k in ["dur_loss", "prior_loss", "diff_loss"]:
                loss_val = loss_dict[k]
                if isinstance(loss_val, torch.Tensor):
                    epoch_losses[k] += loss_val.item()
                elif isinstance(loss_val, (int, float)):
                    epoch_losses[k] += loss_val
            num_batches += 1
            global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to tensorboard
            if global_step % log_every_n_steps == 0:
                writer.add_scalar("loss/train", loss.item(), global_step)
                writer.add_scalar("sub_loss/train_dur_loss", loss_dict["dur_loss"].item() if isinstance(loss_dict["dur_loss"], torch.Tensor) else loss_dict["dur_loss"], global_step)
                writer.add_scalar("sub_loss/train_prior_loss", loss_dict["prior_loss"].item() if isinstance(loss_dict["prior_loss"], torch.Tensor) else loss_dict["prior_loss"], global_step)
                writer.add_scalar("sub_loss/train_diff_loss", loss_dict["diff_loss"].item() if isinstance(loss_dict["diff_loss"], torch.Tensor) else loss_dict["diff_loss"], global_step)
                writer.add_scalar("step", global_step, global_step)

                # Log learning rate
                for i, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"lr/group_{i}", param_group["lr"], global_step)

            # Validation
            if batch_idx > 0 and batch_idx % val_every_n_batches == 0:
                val_losses = validate(model, val_loader, device)
                log.info(f"Validation - Loss: {val_losses['loss']:.4f}")
                writer.add_scalar("loss/val", val_losses["loss"], global_step)
                for k, v in val_losses.items():
                    if k != "loss":
                        writer.add_scalar(f"sub_loss/val_{k}", v, global_step)

        # End of epoch
        avg_epoch_loss = epoch_losses["loss"] / max(num_batches, 1)
        log.info(f"Epoch {epoch + 1} - Average train loss: {avg_epoch_loss:.4f}")

        # Epoch-level validation
        val_losses = validate(model, val_loader, device)
        log.info(f"Epoch {epoch + 1} - Validation loss: {val_losses['loss']:.4f}")
        writer.add_scalar("loss/val_epoch", val_losses["loss"], epoch)

        # Log validation images
        log_validation_images(model, val_loader, device, writer, epoch)

        # Learning rate scheduler step
        if scheduler:
            scheduler.step()

        # Save checkpoint
        checkpoint_path = checkpoint_dir / "last.ckpt"
        model.save_checkpoint(
            str(checkpoint_path),
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
        )

        # Save best model
        if val_losses["loss"] < best_val_loss:
            best_val_loss = val_losses["loss"]
            best_checkpoint_path = checkpoint_dir / "best.ckpt"
            model.save_checkpoint(
                str(best_checkpoint_path),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                global_step=global_step,
            )
            log.info(f"New best model saved with val_loss: {best_val_loss:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            epoch_checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.ckpt"
            model.save_checkpoint(
                str(epoch_checkpoint_path),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                global_step=global_step,
            )

    writer.close()
    log.info("Training completed!")

    # Return metrics
    metric_dict = {
        "loss/train": avg_epoch_loss,
        "loss/val": val_losses["loss"],
    }
    object_dict = {
        "cfg": cfg,
        "model": model,
        "datamodule": datamodule,
    }

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # Print config
    if cfg.get("extras", {}).get("print_config", False):
        log.info("Configuration:")
        log.info(OmegaConf.to_yaml(cfg))

    # Train the model
    metric_dict, _ = train(cfg)

    # Return optimized metric
    metric_value = metric_dict.get(cfg.get("optimized_metric", "loss/val"))
    return metric_value


if __name__ == "__main__":
    main()
