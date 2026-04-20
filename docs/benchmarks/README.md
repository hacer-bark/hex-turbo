# ⚡ Benchmarks & Methodology

This directory contains detailed performance reports for `hex-turbo` across various hardware architectures.

## 📊 Benchmark Reports

### ☁️ Server / Data Center (HFT & Cloud)
High-performance environments where AVX512 and AVX2 are typically available.

*   **[Intel Xeon Platinum 8488C](./intel_xeon_8488c.md)** (WIP)
    *   **Environment:** AWS `c7i.large`
    *   **Features:** AVX512, AVX2
    *   **Context:** Modern HFT/Cloud standard.

*   **[AMD EPYC Genoa](./amd_epyc_genoa.md)**
    *   **Environment:** Vultr `c8a.large`
    *   **Features:** AVX512, AVX2
    *   **Context:** Modern AMD Cloud.

## 🧪 Methodology

All benchmarks were conducted using [criterion.rs](https://github.com/bheisler/criterion.rs) to ensure statistical significance, utilizing a rigorous configuration to filter out OS noise.

### 1. Test Configuration
To ensure high confidence intervals, we utilize longer-than-average test durations and dynamic sampling:

*   **Warm-up Time:** **5 seconds** (Primes branch predictor and L1/L2 caches).
*   **Measurement Time:** **15 seconds** per group.
*   **Noise Threshold:** **0.05** (5%).
*   **Sampling Size:** 50 samples (To maintain reasonable runtime).

### 2. Input Scaling
We benchmark against a logarithmic spread of data sizes to capture performance characteristics across different cache levels (L1 vs L2 vs RAM).

| Size | Use Case |
| :--- | :--- |
| **32 B** | Small strings, headers, keys. |
| **512 B** | API responses, JSON snippets. |
| **4 KB** | Typical OS page size. |
| **64 KB** | Medium binary blobs. |
| **512 KB** | Large documents/images. |
| **1 MB** | Data streams. |
| **10 MB** | Heavy bulk processing. |

> **Note:** Benchmark plots use a **Logarithmic Scale** on the X-axis to visualize the massive range between 32B and 10MB.

### 3. Comparison Targets (`BENCH_TARGET`)
Our benchmark suite is controlled via the `BENCH_TARGET` environment variable. This allows isolated comparisons between specific implementations. You can provide a comma-separated list of targets.

| Target | Description |
| :--- | :--- |
| `turbo` | **(Default)** The standard `hex-turbo` API (LOWER_CASE & UPPER_CASE). |
| `turbo-buff` | The `hex-turbo` zero-allocation API (LOWER_CASE & UPPER_CASE). |
| `simd` | The `hex-simd` crate. |
| `fast` | The `faster-hex` crate. |
| `std` | The classic `hex` crate. |
| `all` | Runs all of the above. |

### 4. Reproduction
You can reproduce these results locally using the following commands:

```bash
# Run comparison: hex-turbo vs hex-simd
BENCH_TARGET=turbo,simd cargo bench

# Run EVERYTHING (Warning: Takes a long time)
BENCH_TARGET=all cargo bench

# Bench ONLY the zero-allocation performance
BENCH_TARGET=turbo-buff cargo bench
```
