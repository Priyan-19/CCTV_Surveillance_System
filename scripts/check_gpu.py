"""GPU and PyTorch CUDA Environment Diagnostic Script.

Verifies whether PyTorch is properly installed with CUDA support,
identifies the active GPU hardware, and displays allocated VRAM capacity.
"""

from __future__ import annotations

import sys
import torch


def check_gpu_environment() -> None:
    """Inspect and display Python version, PyTorch version, CUDA availability,

    and NVIDIA GPU properties.
    """
    print("=" * 60)
    print("        SYSTEM & GPU ACCELERATION CHECK (PyTorch / CUDA)       ")
    print("=" * 60)

    # 1. Python Environment
    print(f"Python Version        : {sys.version.split()[0]} ({sys.executable})")

    # 2. PyTorch Information
    print(f"PyTorch Version       : {torch.__version__}")
    print(f"PyTorch CUDA Built-in : {torch.version.cuda}")

    # 3. CUDA Availability Check
    cuda_available: bool = torch.cuda.is_available()
    print(f"CUDA is Available     : {cuda_available}")

    if not cuda_available:
        print("\n[ERROR] torch.cuda.is_available() returned False!")
        print("Possible causes:")
        print("  - A CPU-only wheel of PyTorch is installed.")
        print("  - NVIDIA GPU driver is missing or outdated.")
        print("  - PyTorch CUDA version mismatch with installed driver.")
        print("Please check nvidia-smi and reinstall PyTorch with the correct CUDA index URL.")
        print("=" * 60)
        return

    # 4. Device Details
    device_count: int = torch.cuda.device_count()
    current_device_id: int = torch.cuda.current_device()
    gpu_name: str = torch.cuda.get_device_name(current_device_id)
    device_props = torch.cuda.get_device_properties(current_device_id)
    total_vram_gb: float = device_props.total_memory / (1024**3)

    print(f"Detected GPU Count    : {device_count}")
    print(f"Current Device ID     : {current_device_id}")
    print(f"GPU Hardware Model    : {gpu_name}")
    print(f"Compute Capability    : {device_props.major}.{device_props.minor}")
    print(f"Total VRAM            : {total_vram_gb:.2f} GB")

    # 5. Quick Tensor Allocation Test on GPU
    try:
        test_tensor = torch.zeros((1, 1), device="cuda")
        print("Tensor Allocation     : SUCCESS (allocated dummy tensor on CUDA)")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        print(f"Tensor Allocation     : FAILED with error: {exc}")

    print("=" * 60)
    print("Status: GPU acceleration is READY for deep learning pipelines.")
    print("=" * 60)


if __name__ == "__main__":
    check_gpu_environment()
