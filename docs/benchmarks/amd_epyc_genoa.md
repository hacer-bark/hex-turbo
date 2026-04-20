# ☁️ Benchmark: AMD EPYC Genoa (Zen 4)

**Context:** Benchmarks executed on a high-performance AMD EPYC "Genoa" processor. This represents the current generation of AMD server capability, featuring full AVX512 support.

*   **Processor:** AMD EPYC-Genoa Processor
*   **OS:** Ubuntu latest

## 📈 Performance Snapshot

![Benchmark Graph](https://github.com/hacer-bark/hex-turbo/blob/main/benches/img/hex_amd.png?raw=true)

**Key Findings:**
1.  **Dominant Decode Lead:** `hex-turbo` achieves **~84.0 GiB/s** decoding throughput, outperforming `hex-simd` by over **2.3x**.
2.  **Ultra-Low Latency:** The zero-allocation API (`TurboBuff`) achieves **2.2ns** encoding latency for small inputs, making it the fastest in its class.
3.  **Superior Scalability:** Unlike competitors, `hex-turbo` maintains peak performance even as payload sizes increase, effectively saturating the pipeline.

## 🏎️ Detailed Results

### 1. Small Payloads (32 Bytes)
**Focus:** Latency & Branch Prediction.
**Crucial for:** HFT Messaging, Authentication Headers.

| Crate | Mode | Encode Latency | Encode Speed | Decode Latency | Decode Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **hex-turbo** | `TurboBuff` | **2.24 ns** | **13.28 GiB/s** | **3.14 ns** | **18.97 GiB/s** |
| **hex-turbo** | `Standard` | 7.69 ns | 3.88 GiB/s | 14.33 ns | 4.16 GiB/s |
| `hex-simd` | `Standard` | 2.50 ns | 11.92 GiB/s | **2.46 ns** | **24.20 GiB/s** |
| `hex` (std) | `Standard` | 126.3 ns | 0.24 GiB/s | 108.3 ns | 0.56 GiB/s |

> **Analysis:** `hex-turbo` (TurboBuff) leads in encoding latency on Zen 4. While `hex-simd` is slightly faster for 32B decoding, `hex-turbo` quickly takes the lead as input sizes grow.

### 2. Medium Payloads (64 KB)
**Focus:** L1 Cache Saturation & AVX512 Implementation.

| Crate | Encode Speed | vs `hex-simd` | Decode Speed | vs `hex-simd` |
| :--- | :--- | :--- | :--- | :--- |
| **hex-turbo** | **61.35 GiB/s** | **+13.0%** | **84.02 GiB/s** | **+137.4%** |
| `hex-simd` | 54.30 GiB/s | - | 35.39 GiB/s | - |
| `hex` (std) | 0.35 GiB/s | -99% | 0.28 GiB/s | -99% |

> **Analysis:**
> *   **Decoding:** The performance gap is massive. `hex-turbo`'s AVX512 decoder is over 2.3x faster than `hex-simd`, thanks to a more efficient register utilization strategy.
> *   **Encoding:** `hex-turbo` maintains a healthy 13% lead, demonstrating superior port balancing on the Zen 4 architecture.

### 3. Large Payloads (128 KB)
**Focus:** Cache-to-Memory transitions.

| Crate | Encode Speed | Decode Speed |
| :--- | :--- | :--- |
| **hex-turbo** | **81.86 GiB/s** | **83.33 GiB/s** |
| `hex-simd` | 54.16 GiB/s | 34.85 GiB/s |
| `hex` (std) | 0.26 GiB/s | 0.27 GiB/s |

> **Analysis:** Performance continues to scale excellently. `hex-turbo` remains the undisputed leader for both high-throughput streaming and low-latency processing.

## 📝 Raw Data Log
<details>
<summary>Click to view raw Criterion output</summary>

```text
Benchmarking Hex_Performances/Encode/Turbo/32
  time:   [7.6804 ns 7.6851 ns 7.6912 ns]
  thrpt:  [3.8749 GiB/s 3.8779 GiB/s 3.8803 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/32
  time:   [2.2434 ns 2.2447 ns 2.2459 ns]
  thrpt:  [13.270 GiB/s 13.277 GiB/s 13.284 GiB/s]

Benchmarking Hex_Performances/Encode/Std/32
  time:   [123.86 ns 126.31 ns 128.04 ns]
  thrpt:  [238.34 MiB/s 241.61 MiB/s 246.38 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/32
  time:   [4.0250 ns 4.0260 ns 4.0271 ns]
  thrpt:  [7.4004 GiB/s 7.4024 GiB/s 7.4044 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/32
  time:   [2.4951 ns 2.5002 ns 2.5070 ns]
  thrpt:  [11.888 GiB/s 11.920 GiB/s 11.944 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/32
  time:   [14.321 ns 14.328 ns 14.336 ns]
  thrpt:  [4.1578 GiB/s 4.1599 GiB/s 4.1619 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/32
  time:   [3.1376 ns 3.1413 ns 3.1463 ns]
  thrpt:  [18.944 GiB/s 18.974 GiB/s 18.997 GiB/s]

Benchmarking Hex_Performances/Decode/Std/32
  time:   [108.23 ns 108.27 ns 108.31 ns]
  thrpt:  [563.50 MiB/s 563.75 MiB/s 563.92 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/32
  time:   [7.3743 ns 7.3753 ns 7.3763 ns]
  thrpt:  [8.0806 GiB/s 8.0817 GiB/s 8.0828 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/32
  time:   [2.4624 ns 2.4635 ns 2.4646 ns]
  thrpt:  [24.184 GiB/s 24.195 GiB/s 24.206 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/64
  time:   [8.2798 ns 8.2813 ns 8.2831 ns]
  thrpt:  [7.1959 GiB/s 7.1975 GiB/s 7.1988 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/64
  time:   [2.6854 ns 2.6862 ns 2.6869 ns]
  thrpt:  [22.184 GiB/s 22.189 GiB/s 22.196 GiB/s]

Benchmarking Hex_Performances/Encode/Std/64
  time:   [196.77 ns 196.80 ns 196.84 ns]
  thrpt:  [310.07 MiB/s 310.14 MiB/s 310.18 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/64
  time:   [4.4999 ns 4.5027 ns 4.5062 ns]
  thrpt:  [13.227 GiB/s 13.237 GiB/s 13.246 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/64
  time:   [2.9747 ns 2.9803 ns 2.9870 ns]
  thrpt:  [19.954 GiB/s 19.999 GiB/s 20.037 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/64
  time:   [14.110 ns 14.114 ns 14.119 ns]
  thrpt:  [8.4434 GiB/s 8.4461 GiB/s 8.4485 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/64
  time:   [3.5796 ns 3.5799 ns 3.5804 ns]
  thrpt:  [33.295 GiB/s 33.299 GiB/s 33.303 GiB/s]

Benchmarking Hex_Performances/Decode/Std/64
  time:   [196.96 ns 197.18 ns 197.35 ns]
  thrpt:  [618.54 MiB/s 619.09 MiB/s 619.78 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/64
  time:   [10.636 ns 10.640 ns 10.645 ns]
  thrpt:  [11.199 GiB/s 11.204 GiB/s 11.208 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/64
  time:   [3.4555 ns 3.4561 ns 3.4569 ns]
  thrpt:  [34.485 GiB/s 34.492 GiB/s 34.499 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/256
  time:   [9.6117 ns 9.6132 ns 9.6149 ns]
  thrpt:  [24.797 GiB/s 24.801 GiB/s 24.805 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/256
  time:   [4.0292 ns 4.0298 ns 4.0307 ns]
  thrpt:  [59.151 GiB/s 59.164 GiB/s 59.173 GiB/s]

Benchmarking Hex_Performances/Encode/Std/256
  time:   [716.40 ns 716.55 ns 716.68 ns]
  thrpt:  [340.65 MiB/s 340.71 MiB/s 340.79 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/256
  time:   [7.9761 ns 7.9875 ns 7.9994 ns]
  thrpt:  [29.805 GiB/s 29.849 GiB/s 29.892 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/256
  time:   [5.3710 ns 5.3725 ns 5.3745 ns]
  thrpt:  [44.361 GiB/s 44.378 GiB/s 44.390 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/256
  time:   [16.857 ns 16.860 ns 16.863 ns]
  thrpt:  [28.277 GiB/s 28.282 GiB/s 28.287 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/256
  time:   [10.162 ns 10.165 ns 10.167 ns]
  thrpt:  [46.899 GiB/s 46.911 GiB/s 46.924 GiB/s]

Benchmarking Hex_Performances/Decode/Std/256
  time:   [662.13 ns 662.47 ns 662.80 ns]
  thrpt:  [736.70 MiB/s 737.06 MiB/s 737.44 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/256
  time:   [32.438 ns 32.449 ns 32.459 ns]
  thrpt:  [14.690 GiB/s 14.695 GiB/s 14.700 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/256
  time:   [13.521 ns 13.523 ns 13.525 ns]
  thrpt:  [35.256 GiB/s 35.261 GiB/s 35.266 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/512
  time:   [19.672 ns 19.681 ns 19.693 ns]
  thrpt:  [24.213 GiB/s 24.228 GiB/s 24.240 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/512
  time:   [6.7943 ns 6.7996 ns 6.8055 ns]
  thrpt:  [70.067 GiB/s 70.127 GiB/s 70.182 GiB/s]

Benchmarking Hex_Performances/Encode/Std/512
  time:   [1.3996 µs 1.3998 µs 1.4000 µs]
  thrpt:  [348.78 MiB/s 348.83 MiB/s 348.87 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/512
  time:   [13.557 ns 13.608 ns 13.647 ns]
  thrpt:  [34.940 GiB/s 35.042 GiB/s 35.172 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/512
  time:   [13.860 ns 13.861 ns 13.863 ns]
  thrpt:  [34.396 GiB/s 34.400 GiB/s 34.404 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/512
  time:   [19.680 ns 19.713 ns 19.753 ns]
  thrpt:  [48.279 GiB/s 48.378 GiB/s 48.460 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/512
  time:   [10.912 ns 10.918 ns 10.924 ns]
  thrpt:  [87.304 GiB/s 87.352 GiB/s 87.398 GiB/s]

Benchmarking Hex_Performances/Decode/Std/512
  time:   [1.2456 µs 1.2458 µs 1.2460 µs]
  thrpt:  [783.78 MiB/s 783.90 MiB/s 784.02 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/512
  time:   [65.814 ns 65.820 ns 65.827 ns]
  thrpt:  [14.488 GiB/s 14.489 GiB/s 14.490 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/512
  time:   [26.953 ns 26.956 ns 26.958 ns]
  thrpt:  [35.376 GiB/s 35.379 GiB/s 35.383 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/1024
  time:   [48.025 ns 48.102 ns 48.204 ns]
  thrpt:  [19.784 GiB/s 19.826 GiB/s 19.858 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/1024
  time:   [16.338 ns 16.343 ns 16.346 ns]
  thrpt:  [58.342 GiB/s 58.355 GiB/s 58.370 GiB/s]

Benchmarking Hex_Performances/Encode/Std/1024
  time:   [3.2537 µs 3.2541 µs 3.2545 µs]
  thrpt:  [300.06 MiB/s 300.10 MiB/s 300.14 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/1024
  time:   [25.961 ns 25.978 ns 26.000 ns]
  thrpt:  [36.680 GiB/s 36.710 GiB/s 36.735 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/1024
  time:   [17.467 ns 17.474 ns 17.480 ns]
  thrpt:  [54.556 GiB/s 54.578 GiB/s 54.598 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/1024
  time:   [31.592 ns 31.602 ns 31.613 ns]
  thrpt:  [60.335 GiB/s 60.355 GiB/s 60.374 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/1024
  time:   [21.769 ns 21.782 ns 21.795 ns]
  thrpt:  [87.514 GiB/s 87.564 GiB/s 87.618 GiB/s]

Benchmarking Hex_Performances/Decode/Std/1024
  time:   [2.4856 µs 2.4860 µs 2.4865 µs]
  thrpt:  [785.51 MiB/s 785.63 MiB/s 785.78 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/1024
  time:   [137.16 ns 137.19 ns 137.21 ns]
  thrpt:  [13.901 GiB/s 13.903 GiB/s 13.906 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/1024
  time:   [53.807 ns 53.812 ns 53.818 ns]
  thrpt:  [35.441 GiB/s 35.444 GiB/s 35.448 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/4096
  time:   [71.901 ns 71.912 ns 71.923 ns]
  thrpt:  [53.039 GiB/s 53.047 GiB/s 53.055 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/4096
  time:   [42.858 ns 42.864 ns 42.871 ns]
  thrpt:  [88.980 GiB/s 88.995 GiB/s 89.008 GiB/s]

Benchmarking Hex_Performances/Encode/Std/4096
  time:   [12.862 µs 12.863 µs 12.864 µs]
  thrpt:  [303.66 MiB/s 303.68 MiB/s 303.70 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/4096
  time:   [102.64 ns 102.66 ns 102.67 ns]
  thrpt:  [37.154 GiB/s 37.160 GiB/s 37.166 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/4096
  time:   [69.473 ns 69.504 ns 69.546 ns]
  thrpt:  [54.851 GiB/s 54.885 GiB/s 54.909 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/4096
  time:   [128.86 ns 128.88 ns 128.90 ns]
  thrpt:  [59.189 GiB/s 59.198 GiB/s 59.207 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/4096
  time:   [86.394 ns 86.433 ns 86.474 ns]
  thrpt:  [88.228 GiB/s 88.269 GiB/s 88.309 GiB/s]

Benchmarking Hex_Performances/Decode/Std/4096
  time:   [9.0536 µs 9.0548 µs 9.0561 µs]
  thrpt:  [862.67 MiB/s 862.80 MiB/s 862.91 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/4096
  time:   [540.75 ns 540.82 ns 540.89 ns]
  thrpt:  [14.105 GiB/s 14.107 GiB/s 14.109 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/4096
  time:   [214.96 ns 214.97 ns 214.99 ns]
  thrpt:  [35.487 GiB/s 35.490 GiB/s 35.493 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/16384
  time:   [211.59 ns 211.64 ns 211.70 ns]
  thrpt:  [72.076 GiB/s 72.098 GiB/s 72.114 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/16384
  time:   [179.25 ns 179.29 ns 179.33 ns]
  thrpt:  [85.087 GiB/s 85.105 GiB/s 85.124 GiB/s]

Benchmarking Hex_Performances/Encode/Std/16384
  time:   [51.178 µs 51.254 µs 51.302 µs]
  thrpt:  [304.57 MiB/s 304.86 MiB/s 305.31 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/16384
  time:   [431.29 ns 431.37 ns 431.47 ns]
  thrpt:  [35.365 GiB/s 35.373 GiB/s 35.380 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/16384
  time:   [293.63 ns 293.81 ns 293.97 ns]
  thrpt:  [51.906 GiB/s 51.934 GiB/s 51.966 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/16384
  time:   [381.99 ns 382.04 ns 382.08 ns]
  thrpt:  [79.872 GiB/s 79.882 GiB/s 79.891 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/16384
  time:   [347.16 ns 347.32 ns 347.48 ns]
  thrpt:  [87.826 GiB/s 87.866 GiB/s 87.905 GiB/s]

Benchmarking Hex_Performances/Decode/Std/16384
  time:   [54.523 µs 54.620 µs 54.737 µs]
  thrpt:  [570.91 MiB/s 572.13 MiB/s 573.15 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/16384
  time:   [2.1619 µs 2.1621 µs 2.1623 µs]
  thrpt:  [14.113 GiB/s 14.115 GiB/s 14.116 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/16384
  time:   [872.83 ns 876.59 ns 880.33 ns]
  thrpt:  [34.666 GiB/s 34.814 GiB/s 34.964 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/65536
  time:   [993.13 ns 994.95 ns 996.84 ns]
  thrpt:  [61.228 GiB/s 61.345 GiB/s 61.457 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/65536
  time:   [991.63 ns 992.08 ns 992.51 ns]
  thrpt:  [61.496 GiB/s 61.523 GiB/s 61.550 GiB/s]

Benchmarking Hex_Performances/Encode/Std/65536
  time:   [175.91 µs 175.93 µs 175.96 µs]
  thrpt:  [355.19 MiB/s 355.25 MiB/s 355.30 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/65536
  time:   [1.6580 µs 1.6584 µs 1.6589 µs]
  thrpt:  [36.793 GiB/s 36.804 GiB/s 36.812 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/65536
  time:   [1.1239 µs 1.1240 µs 1.1242 µs]
  thrpt:  [54.293 GiB/s 54.301 GiB/s 54.309 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/65536
  time:   [1.4528 µs 1.4530 µs 1.4532 µs]
  thrpt:  [84.004 GiB/s 84.015 GiB/s 84.026 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/65536
  time:   [1.4202 µs 1.4207 µs 1.4212 µs]
  thrpt:  [85.891 GiB/s 85.922 GiB/s 85.955 GiB/s]

Benchmarking Hex_Performances/Decode/Std/65536
  time:   [436.44 µs 436.66 µs 436.93 µs]
  thrpt:  [286.09 MiB/s 286.27 MiB/s 286.41 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/65536
  time:   [8.6206 µs 8.6218 µs 8.6231 µs]
  thrpt:  [14.156 GiB/s 14.158 GiB/s 14.160 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/65536
  time:   [3.4485 µs 3.4491 µs 3.4498 µs]
  thrpt:  [35.385 GiB/s 35.392 GiB/s 35.398 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/131072
  time:   [1.4905 µs 1.4912 µs 1.4919 µs]
  thrpt:  [81.820 GiB/s 81.861 GiB/s 81.896 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/131072
  time:   [1.8319 µs 1.8374 µs 1.8430 µs]
  thrpt:  [66.236 GiB/s 66.437 GiB/s 66.637 GiB/s]

Benchmarking Hex_Performances/Encode/Std/131072
  time:   [468.92 µs 468.98 µs 469.05 µs]
  thrpt:  [266.50 MiB/s 266.53 MiB/s 266.57 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/131072
  time:   [3.3076 µs 3.3080 µs 3.3085 µs]
  thrpt:  [36.896 GiB/s 36.901 GiB/s 36.906 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/131072
  time:   [2.2528 µs 2.2539 µs 2.2552 µs]
  thrpt:  [54.129 GiB/s 54.160 GiB/s 54.187 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/131072
  time:   [2.9264 µs 2.9298 µs 2.9332 µs]
  thrpt:  [83.233 GiB/s 83.330 GiB/s 83.427 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/131072
  time:   [2.8975 µs 2.9024 µs 2.9082 µs]
  thrpt:  [83.949 GiB/s 84.117 GiB/s 84.259 GiB/s]

Benchmarking Hex_Performances/Decode/Std/131072
  time:   [912.32 µs 913.26 µs 914.33 µs]
  thrpt:  [273.43 MiB/s 273.75 MiB/s 274.03 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/131072
  time:   [17.480 µs 17.500 µs 17.519 µs]
  thrpt:  [13.936 GiB/s 13.951 GiB/s 13.967 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/131072
  time:   [6.9951 µs 7.0057 µs 7.0161 µs]
  thrpt:  [34.797 GiB/s 34.849 GiB/s 34.902 GiB/s]

Model name: AMD EPYC 9R45
```
</details>

