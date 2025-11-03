#!/bin/bash
# PyTorch and CUDA environment check

echo "=========================================="
echo "  PYTORCH & CUDA CHECK"
echo "=========================================="
echo ""

# Check NVIDIA
echo "=== NVIDIA GPU ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
    echo "✓ NVIDIA driver installed"
else
    echo "✗ nvidia-smi not found"
fi
echo ""

# Check CUDA
echo "=== CUDA Toolkit ==="
if command -v nvcc &> /dev/null; then
    nvcc --version
    echo "✓ CUDA toolkit installed"
else
    echo "⚠ CUDA toolkit not in PATH (OK if using PyTorch bundled CUDA)"
fi
echo ""

# Check Python
echo "=== Python ==="
python3 --version
echo ""

# Run PyTorch checks
echo "=== PyTorch & CUDA ==="
python3 -c "
import sys
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    
    # Check CUDA version
    try:
        print(f'CUDA compiled version: {torch.version.cuda}')
    except:
        print('CUDA compiled version: N/A')
    
    # Check cuDNN
    try:
        if torch.backends.cudnn.is_available():
            print(f'cuDNN version: {torch.backends.cudnn.version()}')
    except:
        pass
    
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f'  Device {i}: {props.name}')
            print(f'    Memory: {props.total_memory / 1024**3:.2f} GB')
            print(f'    Compute capability: {props.major}.{props.minor}')
        
        # Test CUDA operations
        print('Testing CUDA operations...')
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        del x, y, z
        print('✓ CUDA operations work!')
    else:
        print('✗ CUDA not available')
        sys.exit(1)
except ImportError:
    print('✗ PyTorch not installed')
    sys.exit(1)
except Exception as e:
    print(f'✗ Error: {e}')
    sys.exit(1)
"

EXIT_CODE=$?
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ PyTorch with CUDA is working"
else
    echo "  ✗ Issues detected"
fi
echo "=========================================="
exit $EXIT_CODE
