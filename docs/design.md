# 🏗️ Architecture & Design

This document details the internal engineering of `hex-turbo`. The design goal is to maximize throughput for **Memory-Safe Rust** by leveraging SIMD (Single Instruction, Multiple Data) and minimizing CPU pipeline stalls.

## Design Philosophy: Instruction Level Parallelism (ILP)

Traditional Hex implementations iterate byte-by-byte using lookup tables or branching logic. This causes two primary bottlenecks:
1.  **Data Dependency:** Each iteration depends on the previous state, preventing the CPU from executing instructions in parallel.
2.  **Branch Misprediction:** Frequent checks for case or invalid characters inside the hot loop flush the pipeline.

`hex-turbo` replaces byte-level logic with **Vectorized Data Movement**:
*   **Batch Processing:** We load 32 or 64 bytes into vector registers and process them simultaneously.
*   **Branchless Logic:** Character conversion and error detection are handled via bitmasks and parallel shuffles, removing conditional jumps from the hot path.

## 1. Scalar Implementation (SWAR)

For architectures without supported SIMD, we utilize a **SWAR (SIMD Within A Register)** approach.

*   **Wide Loads:** Instead of processing `u8`, we cast pointers to `u64`. This allows us to move 8 bytes of data with a single instruction.
*   **Bitwise Logic:** We construct the Hex characters using bitwise shifts and masks on 64-bit integers, rather than individual byte lookups.
*   **Safety:** While this relies on `unsafe` pointer arithmetic, the bounds are verified via Kani Model Checking to ensure memory safety.

## 2. AVX2 Implementation

The AVX2 path is the primary engine for modern x86_64 CPUs.

### Port Pressure Optimization
We distribute the workload by mixing logical operations with shuffles to saturate multiple execution ports (Ports 0, 1, and 5) simultaneously.

### The "Lane Stitching" Strategy
AVX2 `ymm` registers are 256-bit, but many intrinsics treat them as two separate 128-bit "lanes." We use a permutation strategy to ensure data flows continuously across lane boundaries without falling back to scalar code.

## 3. AVX512 Implementation

The AVX512 implementation targets Zen 4 and newer Intel CPUs (Sapphire Rapids). It offers two distinct advantages:

1.  **Bit-Level Masking (`k` registers):**
    Critical for handling the "tail" of the data (the last 1-31 bytes). We generate a mask for the remaining bytes and process the tail in a single vector instruction.

2.  **Larger Register File:**
    With 32 `zmm` registers, we keep all lookup tables and constants resident in registers to avoid L1 cache latency.

## 4. Runtime Dispatch

The library automatically detects the best execution path at runtime.

**Priority Order:**
1.  **AVX512** (Detected & Verified Safe)
2.  **AVX2**
3.  **Scalar** (Fallback)

## 5. Performance Characteristics

Hex encoding/decoding is primarily a data transformation task.

### Decoder (~84 GiB/s)
Decoding converts 2 characters into 1 byte.
*   `hex-turbo` achieves massive throughput by using parallel nibble extraction.
*   On Zen 4 hardware, we effectively saturate the memory bandwidth.

### Encoder (~81 GiB/s)
Encoding expands 1 byte into 2 characters.
*   Despite the data expansion, our branchless SIMD approach keeps throughput exceptionally high.
*   We use a parallel lookup strategy using `vpshufb` or `vpermb` to map nibbles to characters in a single pass.

## Summary

`hex-turbo` provides the **maximum possible speed within the constraints of Rust's memory safety guarantees**. By leveraging modern SIMD extensions and formal verification, we deliver C-level performance with Rust-level safety.
