"""Evaluation module for HSM experiments."""

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .inference import generate


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    run_name: str
    pretrained: bool
    freeze_transformer: bool
    use_prf: bool
    num_random_features: int = 256
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 5e-5
    train_samples: int = 10000
    val_samples: int = 500
    max_seq_len: int = 128


@dataclass
class EvaluationResults:
    """Results from an evaluation run."""
    config: Dict
    final_train_loss: float
    final_val_loss: float
    perplexity: float
    epoch_losses: List[float]
    val_losses: List[float]
    total_train_time: float
    avg_epoch_time: float
    avg_batch_time: float
    inference_time_per_token: float
    num_parameters: int
    num_hsm_parameters: int


def compute_perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model loss on dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs["loss"].item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    prompt: str = "The quick brown fox",
    num_tokens: int = 20,
    num_runs: int = 5,
    method: str = "greedy",
) -> float:
    """Measure avg inference time per token."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = generate(model, input_ids.clone(), max_new_tokens=num_tokens, method=method)
        elapsed = time.perf_counter() - start
        times.append(elapsed / num_tokens)

    if len(times) > 2:
        times = sorted(times)[1:-1]

    return sum(times) / len(times)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    hsm_params = 0
    if hasattr(model, "hsm_head"):
        hsm_params = sum(p.numel() for p in model.hsm_head.parameters())

    return {"total": total, "trainable": trainable, "hsm": hsm_params}


def create_results(
    config: ExperimentConfig,
    train_metrics,
    val_loss: float,
    inference_time: float,
    param_counts: Dict[str, int],
) -> EvaluationResults:
    return EvaluationResults(
        config=asdict(config),
        final_train_loss=train_metrics.epoch_losses[-1] if train_metrics.epoch_losses else 0.0,
        final_val_loss=val_loss,
        perplexity=compute_perplexity(val_loss),
        epoch_losses=train_metrics.epoch_losses,
        val_losses=train_metrics.val_losses,
        total_train_time=sum(train_metrics.epoch_times),
        avg_epoch_time=sum(train_metrics.epoch_times) / len(train_metrics.epoch_times) if train_metrics.epoch_times else 0.0,
        avg_batch_time=train_metrics.get_summary()["avg_batch_time"],
        inference_time_per_token=inference_time,
        num_parameters=param_counts["total"],
        num_hsm_parameters=param_counts["hsm"],
    )


def save_results(results: EvaluationResults, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"Results saved to {output_path}")


def load_results(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def print_results_summary(results: EvaluationResults):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Run name: {results.config['run_name']}")
    print(f"  Pretrained: {results.config['pretrained']}")
    print(f"  Freeze transformer: {results.config['freeze_transformer']}")
    print(f"  Use PRF: {results.config['use_prf']}")

    print(f"\nLoss Metrics:")
    print(f"  Final train loss: {results.final_train_loss:.4f}")
    print(f"  Final val loss: {results.final_val_loss:.4f}")
    print(f"  Perplexity: {results.perplexity:.2f}")

    print(f"\nRuntime Metrics:")
    print(f"  Total train time: {results.total_train_time:.2f}s")
    print(f"  Avg epoch time: {results.avg_epoch_time:.2f}s")
    print(f"  Avg batch time: {results.avg_batch_time * 1000:.2f}ms")
    print(f"  Inference time/token: {results.inference_time_per_token * 1000:.2f}ms")

    print(f"\nModel Parameters:")
    print(f"  Total: {results.num_parameters:,}")
    print(f"  HSM head: {results.num_hsm_parameters:,}")

    print("=" * 60 + "\n")


def compare_results(results_list: List[EvaluationResults]):
    print("\n" + "=" * 120)
    print("COMPARISON OF EXPERIMENT RUNS")
    print("=" * 120)

    header = (
        f"{'Run':<25} {'Pretrain':<10} {'Freeze':<8} {'PRF':<6} "
        f"{'Epoch(min)':<12} {'Inf(s)':<10} {'Loss':<10} {'PPL':<10}"
    )
    print(header)
    print("-" * 120)

    for r in results_list:
        avg_epoch_min = r.avg_epoch_time / 60.0
        inf_time_sec = r.inference_time_per_token * 20

        row = (
            f"{r.config['run_name']:<25} "
            f"{'Yes' if r.config['pretrained'] else 'No':<10} "
            f"{'Yes' if r.config['freeze_transformer'] else 'No':<8} "
            f"{'Yes' if r.config['use_prf'] else 'No':<6} "
            f"{avg_epoch_min:<12.2f} "
            f"{inf_time_sec:<10.3f} "
            f"{r.final_val_loss:<10.4f} "
            f"{r.perplexity:<10.2f}"
        )
        print(row)

    print("=" * 120 + "\n")

    print("\nDETAILED METRICS:")
    print("-" * 60)
    for r in results_list:
        print(f"\n{r.config['run_name']}:")
        print(f"  Train Loss: {r.final_train_loss:.4f}")
        print(f"  Val Loss: {r.final_val_loss:.4f}")
        print(f"  Perplexity: {r.perplexity:.2f}")
        print(f"  Total Train Time: {r.total_train_time:.1f}s ({r.total_train_time/60:.2f}min)")
        print(f"  Avg Epoch Time: {r.avg_epoch_time:.1f}s ({r.avg_epoch_time/60:.2f}min)")
        print(f"  Inference Time/Token: {r.inference_time_per_token*1000:.2f}ms")
        print(f"  Parameters: {r.num_parameters:,} (HSM: {r.num_hsm_parameters:,})")
