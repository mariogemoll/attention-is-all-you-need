"""
PyTorch and CUDA Environment Check
Checks NVIDIA drivers, CUDA toolkit, and PyTorch CUDA integration.
Generic script that can be used in any project.
"""

import platform
import subprocess
import sys
from typing import Tuple, Union


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"✓ {msg}")


def print_warning(msg: str) -> None:
    """Print warning message."""
    print(f"⚠ WARNING: {msg}")


def print_error(msg: str) -> None:
    """Print error message."""
    print(f"✗ ERROR: {msg}")


def run_command(cmd: str, shell: bool = False) -> Tuple[int, str, str]:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd if shell else cmd.split(), capture_output=True, text=True, shell=shell, timeout=10
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_python() -> None:
    """Check Python version."""
    print_header("Python Environment")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor} is supported")
    else:
        print_warning(
            f"Python {version.major}.{version.minor} may not be optimal. " f"Recommended: 3.8+"
        )

    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")


def check_nvidia_driver() -> bool:
    """Check NVIDIA driver and GPU."""
    print_header("NVIDIA Driver & GPU")

    returncode, stdout, stderr = run_command("nvidia-smi")

    if returncode != 0:
        print_error("nvidia-smi not found or failed to run")
        print_error("No NVIDIA GPU detected or drivers not installed")
        return False

    print(stdout)

    # Parse driver version
    for line in stdout.split("\n"):
        if "Driver Version:" in line:
            print_success("NVIDIA driver detected")
            break

    return True


def check_cuda_toolkit() -> None:
    """Check CUDA toolkit installation."""
    print_header("CUDA Toolkit")

    returncode, stdout, stderr = run_command("nvcc --version")

    if returncode != 0:
        print_warning("CUDA toolkit (nvcc) not found in PATH")
        print("  This is OK if you're using PyTorch with bundled CUDA")
    else:
        print(stdout)
        print_success("CUDA toolkit found")


def check_pytorch() -> bool:
    """Check PyTorch installation and CUDA availability."""
    print_header("PyTorch & CUDA Integration")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print_success("PyTorch is installed")
    except ImportError:
        print_error("PyTorch is not installed")
        print("  Install with: pip install torch torchvision torchaudio")
        return False

    # Check CUDA availability
    print(f"\nCUDA compiled version: {torch.version.cuda}")
    cudnn_version: Union[int, str] = (
        torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        if torch.backends.cudnn.is_available()  # type: ignore[no-untyped-call]
        else "N/A"
    )
    print(f"cuDNN version: {cudnn_version}")

    try:
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            print_success("PyTorch can access CUDA")
            device_count = torch.cuda.device_count()
            print(f"Number of CUDA devices: {device_count}")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  Device {i}: {props.name}")
                print(f"    Compute capability: {props.major}.{props.minor}")
                mem_gb = props.total_memory / 1024**3
                print(f"    Total memory: {mem_gb:.2f} GB")
                print(f"    Multi-processors: {props.multi_processor_count}")

            # Test basic CUDA operations
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                _ = torch.matmul(x, y)
                print_success("Basic CUDA operations work")
            except Exception as e:
                print_error(f"CUDA operations failed: {e}")
                return False
        else:
            print_error("CUDA is not available to PyTorch")
            print("\nPossible issues:")
            print("  1. Driver/CUDA version mismatch")
            print("  2. PyTorch installed without CUDA support")
            print("  3. GPU not detected")
            return False

    except Exception as e:
        print_error(f"Error checking CUDA: {e}")
        return False

    return True


def main() -> None:
    """Run all checks."""
    print("=" * 80)
    print("  PYTORCH & CUDA ENVIRONMENT CHECK")
    print("  " + "=" * 76)
    print(f"  Date: {subprocess.getoutput('date')}")
    print("=" * 80)

    check_python()
    has_gpu = check_nvidia_driver()
    check_cuda_toolkit()
    pytorch_ok = check_pytorch()

    # Final summary
    print_header("Summary")

    if pytorch_ok and has_gpu:
        print_success("System is ready for GPU-accelerated deep learning!")
        print("✓ PyTorch with CUDA support is working correctly")
    elif pytorch_ok and not has_gpu:
        print_warning("PyTorch is installed but no GPU detected (CPU-only mode)")
    else:
        print_error("PyTorch CUDA is not working properly")

    print("\n" + "=" * 80)

    sys.exit(0 if (pytorch_ok and has_gpu) else 1)


if __name__ == "__main__":
    main()
