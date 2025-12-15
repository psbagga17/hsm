# -*- coding: utf-8 -*-
"""Training module for HSM models."""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class TrainingMetrics:
    """Track training metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.epoch_losses = []
        self.batch_losses = []
        self.epoch_times = []
        self.batch_times = []
        self.val_losses = []

    def log_batch(self, loss: float, time_elapsed: float):
        self.batch_losses.append(loss)
        self.batch_times.append(time_elapsed)

    def log_epoch(self, avg_loss: float, time_elapsed: float):
        self.epoch_losses.append(avg_loss)
        self.epoch_times.append(time_elapsed)

    def log_validation(self, val_loss: float):
        self.val_losses.append(val_loss)

    def get_summary(self) -> Dict:
        return {
            "epoch_losses": self.epoch_losses,
            "epoch_times": self.epoch_times,
            "val_losses": self.val_losses,
            "avg_batch_time": sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0,
            "total_time": sum(self.epoch_times),
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    metrics: Optional[TrainingMetrics] = None,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, epoch_time)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    epoch_start = time.perf_counter()
    progress = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress:
        batch_start = time.perf_counter()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1

        batch_time = time.perf_counter() - batch_start
        if metrics:
            metrics.log_batch(batch_loss, batch_time)

        progress.set_postfix({"loss": f"{batch_loss:.4f}"})

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if metrics:
        metrics.log_epoch(avg_loss, epoch_time)

    return avg_loss, epoch_time


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Validate model. Returns avg loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="Validating", leave=False)

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]

        total_loss += loss.item()
        num_batches += 1

        progress.set_postfix({"val_loss": f"{loss.item():.4f}"})

    return total_loss / num_batches if num_batches > 0 else 0.0


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 5,
    save_dir: Optional[Path] = None,
    save_prefix: str = "model",
) -> TrainingMetrics:
    """Full training loop."""
    metrics = TrainingMetrics()

    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        avg_loss, epoch_time = train_epoch(
            model, train_dataloader, optimizer, device, metrics
        )
        print(f"  Train Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        if val_dataloader is not None:
            val_loss = validate(model, val_dataloader, device)
            metrics.log_validation(val_loss)
            print(f"  Val Loss: {val_loss:.4f}")

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = save_dir / f"{save_prefix}_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        print()

    print(f"{'='*60}")
    print("Training complete!")
    print(f"Total time: {sum(metrics.epoch_times):.2f}s")
    print(f"Final train loss: {metrics.epoch_losses[-1]:.4f}")
    if metrics.val_losses:
        print(f"Final val loss: {metrics.val_losses[-1]:.4f}")
    print(f"{'='*60}\n")

    return metrics
