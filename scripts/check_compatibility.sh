#!/bin/bash

# Check system memory
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
MIN_MEM=4096  # 4GB minimum

# Check CPU
CPU_CORES=$(nproc)
MIN_CORES=2

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
MIN_PYTHON="3.8"

# Check GPU (optional)
if command -v nvidia-smi &> /dev/null; then
    GPU_AVAILABLE=true
else
    GPU_AVAILABLE=false
fi

# Print results
echo "System Compatibility Check"
echo "========================"
echo "Memory: $TOTAL_MEM MB (Minimum: $MIN_MEM MB)"
echo "CPU Cores: $CPU_CORES (Minimum: $MIN_CORES)"
echo "Python Version: $PYTHON_VERSION (Minimum: $MIN_PYTHON)"
echo "GPU Available: $GPU_AVAILABLE"

# Check if system meets requirements
if [ $TOTAL_MEM -lt $MIN_MEM ]; then
    echo "WARNING: Insufficient memory"
    echo "AI features may be limited"
fi

if [ $CPU_CORES -lt $MIN_CORES ]; then
    echo "WARNING: Insufficient CPU cores"
    echo "AI features may be limited"
fi