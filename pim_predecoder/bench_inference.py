"""ResNet-18 CIFAR-10 workload-scale accuracy translation under PIM bit-flip noise.

Two-stage composition (not an end-to-end run), exactly as described in the
Methods section "Workload-scale accuracy translation" of the manuscript:

  Stage 1 - weight-corruption sensitivity curve: a ResNet-18 trained on
  CIFAR-10, quantised to symmetric INT8 weights, is evaluated with
  independent per-bit weight flips injected at fixed BER levels, over
  ``--n_seeds`` random seeds per level.

  Stage 2 - pipeline mapping: the curve is read at the measured
  post-decoding BER each ECC pipeline achieves. The BERs come from
  ``result_data.aggregate_1024_e2()`` (the recalibrated 100,000-codeword
  E2 runs behind main-text Table ``tab:ablation_1024``), so the mapping is
  kept in lockstep with the ablation table. The raw pipeline is mapped at
  ``p_bit`` itself, since no decoding intervenes.

The network therefore never sees the correlated drift channel or the
decoder; the result is a workload-scale translation of the ECC measurement.

Usage:
    python3 -m pim_predecoder.bench_inference            # 10 seeds per point
    python3 -m pim_predecoder.bench_inference --n_seeds 3 --device cpu
    python3 -m pim_predecoder.bench_inference --force_train   # retrain from scratch

Outputs:
    pim_predecoder/results/inference_accuracy.json  (consumed by plot_figures.py)
    pim_predecoder/results/inference_accuracy.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from pim_predecoder.result_data import E2_SEEDS, RESULTS, aggregate_1024_e2

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
MODEL_PATH = CACHE_DIR / "resnet18_cifar10_int8.pt"
ACC_CSV = RESULTS / "inference_accuracy.csv"
ACC_JSON = RESULTS / "inference_accuracy.json"


# ---------------------------------------------------------------------------
# Model training / loading
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    data_root = Path(__file__).resolve().parent.parent / "data"
    trainset = torchvision.datasets.CIFAR10(
        root=str(data_root), train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=str(data_root), train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def _resnet18_cifar10() -> nn.Module:
    """ResNet-18 adapted for 32x32 CIFAR-10 (no initial stride-2)."""
    from torchvision.models.resnet import BasicBlock, ResNet

    class CIFARResNet(ResNet):
        def _forward_impl(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    model = CIFARResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)
    model.maxpool = nn.Identity()
    return model


def train_resnet18(train_loader, n_epochs: int = 15, device: str = "cpu"):
    model = _resnet18_cifar10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        scheduler.step()
        acc = 100.0 * correct / total
        print(f"  epoch {epoch + 1:2d}/{n_epochs}  "
              f"loss={running_loss / len(train_loader):.3f}  "
              f"train-acc={acc:.1f}%")
    return model


def load_or_train_model(device: str = "cpu", n_epochs: int = 15,
                        force_train: bool = False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not force_train:
        print(f"[bench] loading cached model from {MODEL_PATH}")
        model = _resnet18_cifar10()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device,
                                         weights_only=True))
        model = model.to(device)
        return model

    print("[bench] training ResNet-18 on CIFAR-10 ...")
    train_loader, _ = get_cifar10_loaders(batch_size=128)
    model = train_resnet18(train_loader, n_epochs=n_epochs, device=device)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[bench] model saved to {MODEL_PATH}")
    return model


def evaluate(model: nn.Module, test_loader, device: str = "cpu") -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Weight quantisation + noise injection
# ---------------------------------------------------------------------------

def quantize_weights_int8(model: nn.Module):
    """In-place symmetric INT8 quantisation of Conv2d and Linear weights."""
    for name, param in model.named_parameters():
        if "weight" not in name or param.dim() < 2:
            continue
        w = param.data
        s = w.abs().max() / 127.0
        if s < 1e-8:
            continue
        w_q = torch.round(w / s).clamp(-128, 127)
        param.data = (w_q * s).to(w.dtype)


def inject_weight_bitflips(model: nn.Module, ber: float,
                           rng: np.random.Generator, device: str = "cpu"):
    """Flip individual bits of INT8 weights with probability `ber` per bit.

    The model is modified in-place. Bit positions 0..7 are flipped
    independently; the weights stay on the same symmetric INT8 grid the
    quantisation established.
    """
    if ber <= 0:
        return
    for name, param in model.named_parameters():
        if "weight" not in name or param.dim() < 2:
            continue
        w = param.data.cpu().numpy()
        # Re-quantise on the stored grid (same scale as quantize_weights_int8).
        s = np.abs(w).max() / 127.0
        if s < 1e-8:
            continue
        # Use uint8 for bit manipulation -- same bit layout as int8.
        w_uint = np.round(w / s).astype(np.int8).view(np.uint8)
        shape = w_uint.shape
        total_weights = int(np.prod(shape))
        for bit in range(8):
            flips = rng.random(total_weights) < ber
            if flips.any():
                mask = flips.reshape(shape)
                w_uint[mask] ^= (1 << bit)
        # Dequantize back
        w_q = w_uint.view(np.int8).astype(np.float32) * s
        param.data = torch.from_numpy(w_q).to(param.data.device)


# ---------------------------------------------------------------------------
# Pipeline BER mapping (stage 2)
# ---------------------------------------------------------------------------

def load_pipeline_bers() -> dict:
    """Post-decoding BERs per pipeline, aligned with main-text Table
    ``tab:ablation_1024`` via ``result_data.aggregate_1024_e2()``
    (E2 strong drift, 100,000 codewords per seed, recalibrated thresholds,
    mean over the model seeds ``result_data.E2_SEEDS``). ``GEMV_BER`` is
    the *deployed* rate: a seed whose gate is closed contributes B'."""
    rows = []
    for r in aggregate_1024_e2():
        rows.append({
            "p_bit": r.p_bit,
            "Bp_BER": r.bp_mean,
            "GEMV_BER": r.gemv_mean,
            "Bp_BER_std": r.bp_std,
            "GEMV_BER_std": r.gemv_std,
            "use_gemv": r.use_gemv_fraction,
        })
    return {
        # Repository-relative, so the released JSON does not embed the path of
        # the machine it was generated on.
        "ablation_source": "pim_predecoder/results/revision_n100k/recal",
        "ecc_seeds": list(E2_SEEDS),
        "rows": rows,
    }


def unique_pipeline_configs(mapping: dict) -> dict[tuple[str, float], float]:
    """Unique (pipeline label, p_bit) -> BER entries for the mapping stage.

    The raw pipeline is mapped at ``p_bit`` itself (no decoding intervenes);
    the B' and GRC-PIM pipelines are mapped at their measured post-decoding
    BERs.
    """
    configs: dict[tuple[str, float], float] = {}
    for row in mapping["rows"]:
        configs[("B'+NB-LDPC", row["p_bit"])] = row["Bp_BER"]
        configs[("B'+GEMV+NB-LDPC", row["p_bit"])] = row["GEMV_BER"]
        configs[("raw (no ECC)", row["p_bit"])] = row["p_bit"]
    return configs


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="ResNet-18 workload-scale accuracy translation under "
                   "PIM bit-flip noise (two-stage composition)")
    ap.add_argument("--n_epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_seeds", type=int, default=10,
                    help="Noise seeds per BER level (manuscript protocol: 10)")
    ap.add_argument("--force_train", action="store_true")
    # Additional BER sweep points beyond the pipeline-mapped ones
    ap.add_argument("--extra_bers", nargs="*", type=float,
                    default=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
                    help="Extra BER points for the sensitivity sweep")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cpu":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[bench] device={device}")

    # ---- 1. Load / train model -------------------------------------------
    model_clean = load_or_train_model(device=device, n_epochs=args.n_epochs,
                                      force_train=args.force_train)
    quantize_weights_int8(model_clean)

    # ---- 2. Load test data -----------------------------------------------
    _, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

    # ---- 3. Clean accuracy baseline -------------------------------------
    acc_clean = evaluate(model_clean, test_loader, device=device)
    print(f"\n[bench] clean INT8 accuracy: {acc_clean:.2f}%")

    # ---- 4. Load pipeline BER values -------------------------------------
    mapping = load_pipeline_bers()

    results: list[dict] = []

    def run_point(ber: float) -> tuple[float, float]:
        accs = []
        for seed in range(args.n_seeds):
            model_copy = _resnet18_cifar10()
            model_copy.load_state_dict(model_clean.state_dict())
            model_copy = model_copy.to(device)
            quantize_weights_int8(model_copy)
            rng = np.random.default_rng(seed)
            inject_weight_bitflips(model_copy, ber, rng, device=device)
            accs.append(evaluate(model_copy, test_loader, device=device))
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1) if len(accs) > 1 else 0)
        return mean_acc, std_acc

    # ---- 5. Sensitivity sweep (stage 1) -----------------------------------
    broad_bers = args.extra_bers
    print(f"\n[bench] running broad BER sweep: {len(broad_bers)} levels x "
          f"{args.n_seeds} seeds")
    for ber in broad_bers:
        mean_acc, std_acc = run_point(ber)
        results.append({
            "pipeline": "broad-sweep",
            "p_bit": None,
            "ber": ber,
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "n_seeds": args.n_seeds,
        })
        print(f"  BER={ber:.1e}: {mean_acc:.2f}% +/- {std_acc:.2f}%")

    # ---- 6. Pipeline-mapped points (stage 2) ------------------------------
    configs = unique_pipeline_configs(mapping)
    print(f"\n[bench] running pipeline-mapped points: {len(configs)} configs x "
          f"{args.n_seeds} seeds")
    for (label, p_bit), ber in sorted(configs.items(), key=lambda x: -x[0][1]):
        accs = []
        for seed in range(args.n_seeds):
            model_copy = _resnet18_cifar10()
            model_copy.load_state_dict(model_clean.state_dict())
            model_copy = model_copy.to(device)
            quantize_weights_int8(model_copy)
            rng = np.random.default_rng(seed + 1000)
            inject_weight_bitflips(model_copy, ber, rng, device=device)
            accs.append(evaluate(model_copy, test_loader, device=device))
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1) if len(accs) > 1 else 0)
        results.append({
            "pipeline": label,
            "p_bit": p_bit,
            "ber": ber,
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "n_seeds": args.n_seeds,
        })
        print(f"  {label} @ p={p_bit:.0e} (BER={ber:.1e}): "
              f"{mean_acc:.2f}% +/- {std_acc:.2f}%")

    # ---- 7. Write output ------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"  Summary: ResNet-18 CIFAR-10 accuracy vs BER (workload-scale")
    print(f"  translation; clean INT8 reference: {acc_clean:.2f}%)")
    print(f"{'=' * 80}")
    for p_bit in sorted({r["p_bit"] for r in results if r["p_bit"] is not None}):
        by_pipe = {r["pipeline"]: r for r in results
                   if r["p_bit"] == p_bit}
        raw = by_pipe.get("raw (no ECC)")
        bp = by_pipe.get("B'+NB-LDPC")
        gv = by_pipe.get("B'+GEMV+NB-LDPC")
        print(f"  p={p_bit:.0e}: raw={raw['accuracy_mean']:.1f}%  "
              f"B'={bp['accuracy_mean']:.1f}%  "
              f"GRC-PIM={gv['accuracy_mean']:.1f}%  "
              f"gain={gv['accuracy_mean'] - bp['accuracy_mean']:+.1f} pp")

    output = {
        "model": "ResNet-18",
        "dataset": "CIFAR-10",
        "clean_accuracy": acc_clean,
        "weight_precision": "INT8",
        "n_seeds_per_point": args.n_seeds,
        "ablation_source": mapping["ablation_source"],
        "ecc_seeds": mapping["ecc_seeds"],
        "results": results,
    }
    with open(ACC_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[bench] wrote {ACC_JSON}")

    with open(ACC_CSV, "w") as f:
        f.write("pipeline,p_bit,ber,accuracy_mean,accuracy_std,n_seeds\n")
        for row in results:
            f.write(f"{row['pipeline']},{row['p_bit']},{row['ber']},"
                    f"{row['accuracy_mean']:.4f},{row['accuracy_std']:.4f},"
                    f"{row['n_seeds']}\n")
    print(f"[bench] wrote {ACC_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
