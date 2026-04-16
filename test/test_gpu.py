from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def format_gb(num_bytes: int) -> float:
    return round(num_bytes / (1024 ** 3), 2)


def format_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024 ** 2), 2)


def default_matmul_sizes(total_memory_gb: float) -> list[int]:
    if total_memory_gb < 8:
        return [2048, 4096]
    if total_memory_gb < 12:
        return [2048, 4096, 6144]
    return [2048, 4096, 6144, 8192]


def query_nvidia_smi(device_index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.total,memory.used,pstate",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    if result.returncode != 0 or not result.stdout.strip():
        return {}

    parts = [item.strip() for item in result.stdout.strip().split(",")]
    if len(parts) != 7:
        return {}

    return {
        "name": parts[0],
        "driver_version": parts[1],
        "temperature_c": try_int(parts[2]),
        "utilization_percent": try_int(parts[3]),
        "memory_total_mb": try_int(parts[4]),
        "memory_used_mb": try_int(parts[5]),
        "pstate": parts[6],
    }


def try_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_device_report(device_index: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device_index)
    report = {
        "index": device_index,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_gb": format_gb(props.total_memory),
        "multi_processor_count": props.multi_processor_count,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    report["nvidia_smi"] = query_nvidia_smi(device_index)
    return report


def supports_bf16() -> bool:
    return bool(
        hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )


def device_to_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def benchmark_matmul(
    device: torch.device,
    size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    device_index = device_to_index(device)
    torch.cuda.reset_peak_memory_stats(device_index)
    label = str(dtype).replace("torch.", "")

    try:
        a = torch.randn((size, size), device=device, dtype=dtype)
        b = torch.randn((size, size), device=device, dtype=dtype)

        for _ in range(warmup):
            _ = a @ b
        torch.cuda.synchronize(device_index)

        times_ms: list[float] = []
        out = None
        for _ in range(repeats):
            start = time.perf_counter()
            out = a @ b
            torch.cuda.synchronize(device_index)
            times_ms.append((time.perf_counter() - start) * 1000)

        if out is None:
            raise RuntimeError("matmul benchmark did not run")

        checksum = float(out.float().abs().mean().item())
        if not math.isfinite(checksum):
            raise RuntimeError(f"{label} matmul produced non-finite output")

        avg_ms = sum(times_ms) / len(times_ms)
        tflops = (2.0 * (size ** 3)) / (avg_ms / 1000.0) / 1e12
        return {
            "status": "ok",
            "dtype": label,
            "size": size,
            "warmup": warmup,
            "repeats": repeats,
            "avg_ms": round(avg_ms, 3),
            "min_ms": round(min(times_ms), 3),
            "max_ms": round(max(times_ms), 3),
            "tflops": round(tflops, 3),
            "peak_memory_mb": format_mb(torch.cuda.max_memory_allocated(device_index)),
            "checksum": round(checksum, 6),
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return {
            "status": "oom",
            "dtype": label,
            "size": size,
            "error": str(exc),
        }
    finally:
        cleanup_cuda()


def benchmark_memory(
    device: torch.device,
    fraction: float,
) -> dict[str, Any]:
    if fraction <= 0:
        return {"status": "skipped", "reason": "memory benchmark disabled"}

    device_index = device_to_index(device)
    torch.cuda.reset_peak_memory_stats(device_index)
    free_bytes = None
    total_bytes = None
    if hasattr(torch.cuda, "mem_get_info"):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)

    if free_bytes is None or total_bytes is None:
        total_bytes = torch.cuda.get_device_properties(device_index).total_memory
        free_bytes = total_bytes

    target_bytes = int(free_bytes * fraction)
    if target_bytes < 16 * 1024 * 1024:
        return {"status": "skipped", "reason": "free memory too small"}

    start = time.perf_counter()
    try:
        buffer = torch.empty(target_bytes // 4, device=device, dtype=torch.float32)
        buffer.uniform_(0.0, 1.0)
        checksum = float(buffer[::4096].sum().item())
        torch.cuda.synchronize(device_index)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "status": "ok",
            "fraction": fraction,
            "requested_gb": format_gb(target_bytes),
            "free_before_gb": format_gb(free_bytes),
            "peak_memory_mb": format_mb(torch.cuda.max_memory_allocated(device_index)),
            "elapsed_ms": round(elapsed_ms, 3),
            "checksum": round(checksum, 6),
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return {
            "status": "oom",
            "fraction": fraction,
            "requested_gb": format_gb(target_bytes),
            "error": str(exc),
        }
    finally:
        cleanup_cuda()


def pick_train_shape(total_memory_gb: float) -> dict[str, int]:
    if total_memory_gb < 8:
        return {"batch_size": 512, "input_dim": 1024, "hidden_dim": 2048, "num_classes": 512}
    if total_memory_gb < 12:
        return {"batch_size": 1024, "input_dim": 2048, "hidden_dim": 3072, "num_classes": 1024}
    return {"batch_size": 1536, "input_dim": 2048, "hidden_dim": 4096, "num_classes": 2048}


def benchmark_training(
    device: torch.device,
    steps: int,
    warmup: int,
    total_memory_gb: float,
) -> dict[str, Any]:
    if steps <= 0:
        return {"status": "skipped", "reason": "training benchmark disabled"}

    device_index = device_to_index(device)
    shape = pick_train_shape(total_memory_gb)
    amp_dtype = torch.bfloat16 if supports_bf16() else torch.float16
    amp_enabled = True

    model = nn.Sequential(
        nn.Linear(shape["input_dim"], shape["hidden_dim"]),
        nn.GELU(),
        nn.Linear(shape["hidden_dim"], shape["hidden_dim"]),
        nn.GELU(),
        nn.Linear(shape["hidden_dim"], shape["num_classes"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler_enabled = amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    inputs = torch.randn(shape["batch_size"], shape["input_dim"], device=device)
    targets = torch.randint(0, shape["num_classes"], (shape["batch_size"],), device=device)

    def run_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
            if amp_enabled
            else nullcontext()
        ):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("training loss is not finite")
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return float(loss.detach().float().item())

    torch.cuda.reset_peak_memory_stats(device_index)
    try:
        last_loss = None
        for _ in range(warmup):
            last_loss = run_step()
        torch.cuda.synchronize(device_index)

        times_ms: list[float] = []
        for _ in range(steps):
            start = time.perf_counter()
            last_loss = run_step()
            torch.cuda.synchronize(device_index)
            times_ms.append((time.perf_counter() - start) * 1000)

        if last_loss is None:
            raise RuntimeError("training benchmark did not run")

        avg_ms = sum(times_ms) / len(times_ms)
        return {
            "status": "ok",
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "batch_size": shape["batch_size"],
            "input_dim": shape["input_dim"],
            "hidden_dim": shape["hidden_dim"],
            "num_classes": shape["num_classes"],
            "steps": steps,
            "avg_step_ms": round(avg_ms, 3),
            "samples_per_sec": round(shape["batch_size"] / (avg_ms / 1000.0), 2),
            "peak_memory_mb": format_mb(torch.cuda.max_memory_allocated(device_index)),
            "final_loss": round(last_loss, 6),
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return {"status": "oom", "error": str(exc), **shape}
    finally:
        cleanup_cuda()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CUDA GPU benchmark for RTX 5060-class cards")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per benchmark")
    parser.add_argument("--repeats", type=int, default=20, help="Measured iterations per matmul size")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=None,
        help="Matrix sizes for matmul benchmark, e.g. --sizes 2048 4096 6144",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.15,
        help="Fraction of currently free GPU memory used by the memory test",
    )
    parser.add_argument("--train-steps", type=int, default=20, help="Measured training steps")
    parser.add_argument("--json", action="store_true", help="Print JSON report only")
    return parser


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "status": "error",
            "reason": "CUDA is not available. Check the NVIDIA driver and the CUDA-enabled PyTorch build.",
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        }

    device_count = torch.cuda.device_count()
    if args.device < 0 or args.device >= device_count:
        return {
            "status": "error",
            "reason": f"CUDA device index {args.device} is out of range. device_count={device_count}",
        }

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device_report = get_device_report(args.device)
    sizes = args.sizes or default_matmul_sizes(device_report["total_memory_gb"])
    dtype_plan: list[torch.dtype] = [torch.float32, torch.float16]
    if supports_bf16():
        dtype_plan.append(torch.bfloat16)

    matmul_results: list[dict[str, Any]] = []
    for dtype in dtype_plan:
        for size in sizes:
            matmul_results.append(
                benchmark_matmul(
                    device=device,
                    size=size,
                    dtype=dtype,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )

    memory_result = benchmark_memory(device=device, fraction=args.memory_fraction)
    training_result = benchmark_training(
        device=device,
        steps=args.train_steps,
        warmup=max(2, args.warmup // 2),
        total_memory_gb=device_report["total_memory_gb"],
    )

    name = device_report["name"].lower()
    looks_like_5060 = "5060" in name
    return {
        "status": "ok",
        "looks_like_5060": looks_like_5060,
        "device": device_report,
        "matmul": matmul_results,
        "memory": memory_result,
        "training": training_result,
    }


def print_human_readable(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("GPU benchmark report")
    print("=" * 72)

    if report["status"] != "ok":
        print(f"status           : {report['status']}")
        print(f"reason           : {report['reason']}")
        if "torch_version" in report:
            print(f"torch version    : {report['torch_version']}")
            print(f"torch cuda build : {report['torch_cuda_version']}")
        return

    device = report["device"]
    print(f"device           : {device['name']}")
    print(f"device index     : {device['index']}")
    print(f"compute cap      : {device['capability']}")
    print(f"memory           : {device['total_memory_gb']} GB")
    print(f"SM count         : {device['multi_processor_count']}")
    print(f"torch            : {device['torch_version']}")
    print(f"cuda build       : {device['torch_cuda_version']}")
    print(f"cudnn            : {device['cudnn_version']}")
    print(f"5060 detected    : {'yes' if report['looks_like_5060'] else 'no'}")

    smi = device.get("nvidia_smi") or {}
    if smi:
        print(f"driver           : {smi.get('driver_version')}")
        print(f"temp/util        : {smi.get('temperature_c')} C / {smi.get('utilization_percent')}%")
        print(f"mem used         : {smi.get('memory_used_mb')} / {smi.get('memory_total_mb')} MB")
        print(f"pstate           : {smi.get('pstate')}")

    print()
    print("Matmul benchmarks")
    for item in report["matmul"]:
        if item["status"] == "ok":
            print(
                f"  {item['dtype']:>8} size={item['size']:<5} "
                f"avg={item['avg_ms']:>8} ms  tflops={item['tflops']:>8}  "
                f"peak_mem={item['peak_memory_mb']:>8} MB"
            )
        else:
            print(
                f"  {item['dtype']:>8} size={item['size']:<5} status={item['status']} "
                f"error={item.get('error', '')}"
            )

    print()
    print("Memory benchmark")
    memory = report["memory"]
    if memory["status"] == "ok":
        print(
            f"  requested={memory['requested_gb']} GB  "
            f"elapsed={memory['elapsed_ms']} ms  "
            f"peak_mem={memory['peak_memory_mb']} MB"
        )
    else:
        print(f"  status={memory['status']} reason={memory.get('reason', memory.get('error', ''))}")

    print()
    print("Training benchmark")
    training = report["training"]
    if training["status"] == "ok":
        print(
            f"  amp={training['amp_dtype']}  batch={training['batch_size']}  "
            f"avg_step={training['avg_step_ms']} ms  "
            f"samples/s={training['samples_per_sec']}  "
            f"peak_mem={training['peak_memory_mb']} MB  "
            f"final_loss={training['final_loss']}"
        )
    else:
        print(f"  status={training['status']} reason={training.get('reason', training.get('error', ''))}")


def main() -> int:
    args = build_parser().parse_args()
    report = run_report(args)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human_readable(report)

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
