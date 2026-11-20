#!/usr/bin/env python3
"""
SNN Sparse Format Crossover Calculator
Estimates when block-sparse matrix multiplication beats dense
based on NVIDIA Tensor Core architecture data.
"""

def crossover_density(architecture):
    """Return the density threshold below which block-sparse is faster."""
    thresholds = {
        "Volta": 0.40,
        "Ampere": 0.50,
        "Hopper": 0.55,
    }
    return thresholds.get(architecture, 0.45)

def memory_savings(neurons, density, block_size=16):
    """Estimate memory savings of block-sparse over dense."""
    dense_bytes = neurons * neurons * 4  # float32
    blocks = (neurons + block_size - 1) // block_size
    non_zero_blocks = int(blocks * blocks * density)
    sparse_bytes = non_zero_blocks * (2 + block_size * block_size) * 4
    return dense_bytes, sparse_bytes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SNN Sparse Format Crossover Calculator")
    parser.add_argument("--neurons", type=int, default=10000, help="Number of neurons")
    parser.add_argument("--density", type=float, default=0.3, help="Connection density (0-1)")
    parser.add_argument("--arch", default="Ampere", help="GPU architecture")
    args = parser.parse_args()

    threshold = crossover_density(args.arch)
    dense, sparse = memory_savings(args.neurons, args.density)
    print(f"Architecture: {args.arch}")
    print(f"Density: {args.density:.0%}, Threshold: {threshold:.0%}")
    print(f"Dense memory: {dense / 1e6:.2f} MB")
    print(f"Sparse memory: {sparse / 1e6:.2f} MB")
    if args.density < threshold:
        print("Block-sparse is faster.")
    else:
        print("Dense is faster.")
