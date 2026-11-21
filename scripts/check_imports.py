"""Library Dependency Verification Script.

Tests importing each required core dependency and reports either
its installed version and status (OK) or the exact error traceback.
"""

from __future__ import annotations

import importlib
import sys
from typing import Dict, List, Tuple


def test_imports() -> bool:
    """Attempt to import all required libraries and report status."""
    libraries_to_check: List[Tuple[str, str]] = [
        ("torch", "PyTorch deep learning framework"),
        ("torchvision", "PyTorch computer vision models & utilities"),
        ("ultralytics", "Ultralytics YOLO object detection"),
        ("cv2", "OpenCV computer vision library"),
        ("numpy", "NumPy array computation library"),
        ("yaml", "PyYAML configuration parser"),
        ("insightface", "InsightFace deep face recognition library"),
        ("onnxruntime", "ONNX Runtime execution engine"),
        ("fastapi", "FastAPI modern web API framework"),
        ("uvicorn", "Uvicorn ASGI web server"),
        ("pytest", "Pytest testing framework"),
    ]

    print("=" * 70)
    print("               DEPENDENCY IMPORT VERIFICATION                   ")
    print("=" * 70)
    print(f"Python Executable: {sys.executable}\n")

    results: Dict[str, Tuple[bool, str]] = {}

    for mod_name, description in libraries_to_check:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "version detected (no __version__ attr)")
            results[mod_name] = (True, ver)
            print(f"[OK]   {mod_name:<15} ({ver:<15}) - {description}")
        except Exception as err:
            err_msg = f"{type(err).__name__}: {err}"
            results[mod_name] = (False, err_msg)
            print(f"[FAIL] {mod_name:<15} FAILED: {err_msg}")

    print("=" * 70)
    total = len(libraries_to_check)
    passed = sum(1 for status, _ in results.values() if status)
    failed = total - passed

    print(f"Summary: {passed}/{total} passed, {failed}/{total} failed.")
    print("=" * 70)

    if failed > 0:
        print("\nAction required: Fix failed packages before proceeding.")
        return False

    print("\nAll required core dependencies imported successfully!")
    return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
