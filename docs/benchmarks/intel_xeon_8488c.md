# ☁️ Benchmark: Intel Xeon Platinum 8488C (AWS)

**Context:** This benchmark represents the **"State of the Art"** for cloud computing. The tests were executed on an AWS `c7i.large` instance powered by Intel Sapphire Rapids.

*   **Processor:** Intel(R) Xeon(R) Platinum 8488C
*   **Instruction Set:** AVX512 (Enabled)
*   **Environment:** AWS Nitro Hypervisor
*   **OS:** Ubuntu latest (Minimal, `cargo` only)

## 📈 Performance Snapshot

![Benchmark Graph](https://github.com/hacer-bark/base64-turbo/blob/main/benches/img/base64_intel.png?raw=true)

**Key Findings:**
1.  **Decode Domination:** `base64-turbo` achieves **21.04 GiB/s** in decoding, more than **2x faster** than the previous Rust standard.
2.  **Low Latency:** For small inputs (32B), the zero-allocation API (`TurboBuff`) offers **~10ns** latency, critical for HFT messaging.
3.  **Memory Saturation:** On large payloads (10MB+), the library effectively saturates the memory bandwidth.

## 🏎️ Detailed Results

### 1. Small Payloads (32 Bytes)
**Focus:** Latency & Branch Prediction.
*Crucial for:** FIX Messages, API Keys, Headers.*

| Crate | Mode | Encode Latency | Encode Throughput | Decode Latency | Decode Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | `TurboBuff` | **10.18 ns** | **2.93 GiB/s** | **13.49 ns** | **3.04 GiB/s** |
| **base64-turbo** | `Standard` | 16.61 ns | 1.79 GiB/s | 19.12 ns | 2.14 GiB/s |
| `base64-simd` | `Standard` | 18.12 ns | 1.64 GiB/s | 15.11 ns | 2.71 GiB/s |
| `base64` (std) | `Standard` | 56.51 ns | 0.54 GiB/s | 49.65 ns | 0.85 GiB/s |

> **Analysis:** `base64-turbo` (TurboBuff) is **1.8x faster** than `base64-simd` on encoding latency. This confirms the efficiency of our runtime feature detection and hot-path optimization.

### 2. Medium Payloads (64 KB)
**Focus:** L1/L2 Cache Saturation & AVX Efficiency.
*Crucial for:** JSON blobs, Binary responses, Images.*

| Crate | Encode Speed | vs `base64-simd` | Decode Speed | vs `base64-simd` |
| :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | **12.48 GiB/s** | **+18.1%** | **21.04 GiB/s** | **+110.1%** |
| `base64-simd` | 10.57 GiB/s | - | 10.01 GiB/s | - |
| `base64` (std) | 2.42 GiB/s | -77% | 2.78 GiB/s | -72% |

> **Analysis:** This is where AVX512 shines.
> *   **Decoding:** The massive **2.1x speedup** comes from our "Logic > Memory" approach, utilizing the full width of AVX512 registers.
> *   **Encoding:** We see a solid ~20% gain, largely due to better port saturation (offloading shuffle operations).

### 3. Large Payloads (10 MB)
**Focus:** RAM Bandwidth & Prefetching.
*Crucial for:** File uploads, Video streams, Backups.*

| Crate | Encode Speed | Decode Speed |
| :--- | :--- | :--- |
| **base64-turbo** | **11.89 GiB/s** | **15.72 GiB/s** |
| `base64-simd` | 10.27 GiB/s | 9.98 GiB/s |
| `base64` (std) | 2.16 GiB/s | 2.60 GiB/s |

> **Analysis:** Even when bottlenecked by main system RAM, `base64-turbo` maintains a significant lead in decoding (`+57%`), proving that our loop unrolling and prefetching strategies effectively hide memory latency.

## 📝 Raw Data Log
<details>
<summary>Click to view raw Criterion output</summary>

```text
Benchmarking Hex_Performances/Encode/Turbo/32
  time:   [9.6015 ns 9.6080 ns 9.6143 ns]
  thrpt:  [3.0998 GiB/s 3.1018 GiB/s 3.1039 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/32
  time:   [3.3378 ns 3.3392 ns 3.3404 ns]
  thrpt:  [8.9217 GiB/s 8.9250 GiB/s 8.9287 GiB/s]

Benchmarking Hex_Performances/Encode/Std/32
  time:   [78.636 ns 79.208 ns 79.747 ns]
  thrpt:  [382.68 MiB/s 385.28 MiB/s 388.09 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/32
  time:   [4.0057 ns 4.0174 ns 4.0313 ns]
  thrpt:  [7.3927 GiB/s 7.4184 GiB/s 7.4400 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/32
  time:   [3.7706 ns 3.7714 ns 3.7723 ns]
  thrpt:  [7.9003 GiB/s 7.9022 GiB/s 7.9038 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/32
  time:   [11.231 ns 11.241 ns 11.254 ns]
  thrpt:  [5.2962 GiB/s 5.3024 GiB/s 5.3072 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/32
  time:   [5.4678 ns 5.4739 ns 5.4794 ns]
  thrpt:  [10.878 GiB/s 10.889 GiB/s 10.901 GiB/s]

Benchmarking Hex_Performances/Decode/Std/32
  time:   [105.76 ns 105.97 ns 106.19 ns]
  thrpt:  [574.77 MiB/s 575.96 MiB/s 577.11 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/32
  time:   [7.7702 ns 7.7724 ns 7.7751 ns]
  thrpt:  [7.6661 GiB/s 7.6687 GiB/s 7.6709 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/32
  time:   [3.8524 ns 3.8682 ns 3.8823 ns]
  thrpt:  [15.353 GiB/s 15.409 GiB/s 15.472 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/64
  time:   [9.5915 ns 9.5956 ns 9.6027 ns]
  thrpt:  [6.2071 GiB/s 6.2117 GiB/s 6.2143 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/64
  time:   [3.4668 ns 3.4678 ns 3.4691 ns]
  thrpt:  [17.182 GiB/s 17.188 GiB/s 17.193 GiB/s]

Benchmarking Hex_Performances/Encode/Std/64
  time:   [134.42 ns 134.93 ns 135.50 ns]
  thrpt:  [450.45 MiB/s 452.33 MiB/s 454.05 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/64
  time:   [4.9228 ns 4.9255 ns 4.9283 ns]
  thrpt:  [12.094 GiB/s 12.101 GiB/s 12.108 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/64
  time:   [4.5342 ns 4.5357 ns 4.5371 ns]
  thrpt:  [13.137 GiB/s 13.141 GiB/s 13.146 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/64
  time:   [12.781 ns 12.786 ns 12.793 ns]
  thrpt:  [9.3186 GiB/s 9.3233 GiB/s 9.3274 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/64
  time:   [6.9342 ns 6.9410 ns 6.9478 ns]
  thrpt:  [17.158 GiB/s 17.175 GiB/s 17.191 GiB/s]

Benchmarking Hex_Performances/Decode/Std/64
  time:   [229.48 ns 229.76 ns 230.02 ns]
  thrpt:  [530.70 MiB/s 531.28 MiB/s 531.94 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/64
  time:   [12.166 ns 12.175 ns 12.183 ns]
  thrpt:  [9.7849 GiB/s 9.7913 GiB/s 9.7984 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/64
  time:   [5.7309 ns 5.7340 ns 5.7396 ns]
  thrpt:  [20.769 GiB/s 20.790 GiB/s 20.801 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/128
  time:   [10.728 ns 10.731 ns 10.735 ns]
  thrpt:  [11.105 GiB/s 11.108 GiB/s 11.112 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/128
  time:   [5.0908 ns 5.0921 ns 5.0935 ns]
  thrpt:  [23.404 GiB/s 23.411 GiB/s 23.417 GiB/s]

Benchmarking Hex_Performances/Encode/Std/128
  time:   [246.58 ns 247.22 ns 247.88 ns]
  thrpt:  [492.46 MiB/s 493.78 MiB/s 495.06 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/128
  time:   [7.5410 ns 7.5441 ns 7.5470 ns]
  thrpt:  [15.796 GiB/s 15.802 GiB/s 15.808 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/128
  time:   [6.3256 ns 6.3284 ns 6.3310 ns]
  thrpt:  [18.829 GiB/s 18.837 GiB/s 18.845 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/128
  time:   [14.063 ns 14.069 ns 14.075 ns]
  thrpt:  [16.939 GiB/s 16.947 GiB/s 16.954 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/128
  time:   [8.5155 ns 8.5231 ns 8.5315 ns]
  thrpt:  [27.946 GiB/s 27.973 GiB/s 27.998 GiB/s]

Benchmarking Hex_Performances/Decode/Std/128
  time:   [425.30 ns 427.40 ns 429.39 ns]
  thrpt:  [568.58 MiB/s 571.23 MiB/s 574.04 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/128
  time:   [25.275 ns 25.277 ns 25.280 ns]
  thrpt:  [9.4313 GiB/s 9.4322 GiB/s 9.4331 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/128
  time:   [9.9674 ns 9.9755 ns 9.9835 ns]
  thrpt:  [23.881 GiB/s 23.900 GiB/s 23.920 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/256
  time:   [13.875 ns 13.878 ns 13.880 ns]
  thrpt:  [17.177 GiB/s 17.180 GiB/s 17.183 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/256
  time:   [8.5422 ns 8.5491 ns 8.5542 ns]
  thrpt:  [27.871 GiB/s 27.888 GiB/s 27.911 GiB/s]

Benchmarking Hex_Performances/Encode/Std/256
  time:   [461.47 ns 462.40 ns 463.31 ns]
  thrpt:  [526.95 MiB/s 527.98 MiB/s 529.05 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/256
  time:   [15.703 ns 15.704 ns 15.706 ns]
  thrpt:  [15.180 GiB/s 15.182 GiB/s 15.183 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/256
  time:   [9.8954 ns 9.8979 ns 9.9001 ns]
  thrpt:  [24.082 GiB/s 24.088 GiB/s 24.094 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/256
  time:   [19.363 ns 19.384 ns 19.405 ns]
  thrpt:  [24.573 GiB/s 24.600 GiB/s 24.626 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/256
  time:   [14.362 ns 14.376 ns 14.389 ns]
  thrpt:  [33.138 GiB/s 33.168 GiB/s 33.201 GiB/s]

Benchmarking Hex_Performances/Decode/Std/256
  time:   [778.91 ns 779.78 ns 780.53 ns]
  thrpt:  [625.58 MiB/s 626.18 MiB/s 626.88 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/256
  time:   [44.319 ns 44.338 ns 44.358 ns]
  thrpt:  [10.750 GiB/s 10.755 GiB/s 10.759 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/256
  time:   [19.725 ns 19.727 ns 19.730 ns]
  thrpt:  [24.168 GiB/s 24.172 GiB/s 24.175 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/512
  time:   [22.018 ns 22.047 ns 22.073 ns]
  thrpt:  [21.602 GiB/s 21.628 GiB/s 21.657 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/512
  time:   [16.874 ns 16.890 ns 16.902 ns]
  thrpt:  [28.212 GiB/s 28.232 GiB/s 28.259 GiB/s]

Benchmarking Hex_Performances/Encode/Std/512
  time:   [895.47 ns 896.46 ns 897.35 ns]
  thrpt:  [544.14 MiB/s 544.68 MiB/s 545.28 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/512
  time:   [26.309 ns 26.313 ns 26.317 ns]
  thrpt:  [18.119 GiB/s 18.122 GiB/s 18.124 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/512
  time:   [20.168 ns 20.171 ns 20.174 ns]
  thrpt:  [23.636 GiB/s 23.639 GiB/s 23.643 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/512
  time:   [30.987 ns 31.012 ns 31.037 ns]
  thrpt:  [30.727 GiB/s 30.752 GiB/s 30.777 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/512
  time:   [26.456 ns 26.485 ns 26.513 ns]
  thrpt:  [35.971 GiB/s 36.009 GiB/s 36.048 GiB/s]

Benchmarking Hex_Performances/Decode/Std/512
  time:   [1.4615 µs 1.4654 µs 1.4693 µs]
  thrpt:  [664.64 MiB/s 666.41 MiB/s 668.20 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/512
  time:   [90.030 ns 90.059 ns 90.111 ns]
  thrpt:  [10.583 GiB/s 10.589 GiB/s 10.593 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/512
  time:   [38.471 ns 38.510 ns 38.566 ns]
  thrpt:  [24.728 GiB/s 24.765 GiB/s 24.790 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/1024
  time:   [59.389 ns 59.417 ns 59.446 ns]
  thrpt:  [16.043 GiB/s 16.051 GiB/s 16.058 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/1024
  time:   [27.862 ns 27.880 ns 27.901 ns]
  thrpt:  [34.180 GiB/s 34.206 GiB/s 34.229 GiB/s]

Benchmarking Hex_Performances/Encode/Std/1024
  time:   [1.7743 µs 1.7755 µs 1.7766 µs]
  thrpt:  [549.68 MiB/s 550.03 MiB/s 550.40 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/1024
  time:   [51.002 ns 51.015 ns 51.031 ns]
  thrpt:  [18.688 GiB/s 18.694 GiB/s 18.699 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/1024
  time:   [32.283 ns 32.304 ns 32.330 ns]
  thrpt:  [29.498 GiB/s 29.522 GiB/s 29.541 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/1024
  time:   [56.078 ns 56.091 ns 56.105 ns]
  thrpt:  [33.996 GiB/s 34.005 GiB/s 34.012 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/1024
  time:   [50.564 ns 50.572 ns 50.581 ns]
  thrpt:  [37.709 GiB/s 37.716 GiB/s 37.721 GiB/s]

Benchmarking Hex_Performances/Decode/Std/1024
  time:   [2.9788 µs 2.9869 µs 2.9947 µs]
  thrpt:  [652.19 MiB/s 653.89 MiB/s 655.66 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/1024
  time:   [176.61 ns 176.64 ns 176.67 ns]
  thrpt:  [10.796 GiB/s 10.798 GiB/s 10.800 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/1024
  time:   [77.935 ns 77.987 ns 78.045 ns]
  thrpt:  [24.439 GiB/s 24.457 GiB/s 24.474 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/2048
  time:   [92.794 ns 92.861 ns 92.926 ns]
  thrpt:  [20.525 GiB/s 20.540 GiB/s 20.555 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/2048
  time:   [54.489 ns 54.581 ns 54.669 ns]
  thrpt:  [34.889 GiB/s 34.945 GiB/s 35.004 GiB/s]

Benchmarking Hex_Performances/Encode/Std/2048
  time:   [3.4995 µs 3.5008 µs 3.5021 µs]
  thrpt:  [557.70 MiB/s 557.92 MiB/s 558.12 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/2048
  time:   [107.72 ns 107.74 ns 107.76 ns]
  thrpt:  [17.701 GiB/s 17.703 GiB/s 17.706 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/2048
  time:   [65.091 ns 65.120 ns 65.153 ns]
  thrpt:  [29.275 GiB/s 29.290 GiB/s 29.303 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/2048
  time:   [131.42 ns 131.57 ns 131.72 ns]
  thrpt:  [28.962 GiB/s 28.994 GiB/s 29.026 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/2048
  time:   [101.79 ns 101.85 ns 101.91 ns]
  thrpt:  [37.432 GiB/s 37.453 GiB/s 37.475 GiB/s]

Benchmarking Hex_Performances/Decode/Std/2048
  time:   [5.7460 µs 5.7535 µs 5.7606 µs]
  thrpt:  [678.09 MiB/s 678.93 MiB/s 679.82 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/2048
  time:   [350.79 ns 350.83 ns 350.87 ns]
  thrpt:  [10.872 GiB/s 10.873 GiB/s 10.874 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/2048
  time:   [156.83 ns 156.87 ns 156.90 ns]
  thrpt:  [24.312 GiB/s 24.318 GiB/s 24.323 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/4096
  time:   [141.26 ns 141.28 ns 141.30 ns]
  thrpt:  [26.996 GiB/s 27.001 GiB/s 27.005 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/4096
  time:   [111.68 ns 111.76 ns 111.84 ns]
  thrpt:  [34.108 GiB/s 34.132 GiB/s 34.157 GiB/s]

Benchmarking Hex_Performances/Encode/Std/4096
  time:   [6.9551 µs 6.9568 µs 6.9586 µs]
  thrpt:  [561.35 MiB/s 561.50 MiB/s 561.64 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/4096
  time:   [201.13 ns 201.15 ns 201.18 ns]
  thrpt:  [18.962 GiB/s 18.964 GiB/s 18.966 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/4096
  time:   [112.55 ns 112.56 ns 112.57 ns]
  thrpt:  [33.887 GiB/s 33.891 GiB/s 33.895 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/4096
  time:   [241.68 ns 241.77 ns 241.85 ns]
  thrpt:  [31.546 GiB/s 31.557 GiB/s 31.568 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/4096
  time:   [202.74 ns 202.83 ns 202.90 ns]
  thrpt:  [37.602 GiB/s 37.616 GiB/s 37.631 GiB/s]

Benchmarking Hex_Performances/Decode/Std/4096
  time:   [14.389 µs 14.497 µs 14.619 µs]
  thrpt:  [534.41 MiB/s 538.90 MiB/s 542.93 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/4096
  time:   [694.61 ns 694.69 ns 694.78 ns]
  thrpt:  [10.981 GiB/s 10.982 GiB/s 10.984 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/4096
  time:   [307.18 ns 307.21 ns 307.25 ns]
  thrpt:  [24.831 GiB/s 24.834 GiB/s 24.837 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/8192
  time:   [240.76 ns 240.81 ns 240.86 ns]
  thrpt:  [31.676 GiB/s 31.682 GiB/s 31.689 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/8192
  time:   [211.13 ns 211.14 ns 211.15 ns]
  thrpt:  [36.132 GiB/s 36.134 GiB/s 36.136 GiB/s]

Benchmarking Hex_Performances/Encode/Std/8192
  time:   [13.837 µs 13.841 µs 13.845 µs]
  thrpt:  [564.27 MiB/s 564.45 MiB/s 564.62 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/8192
  time:   [395.03 ns 395.07 ns 395.10 ns]
  thrpt:  [19.310 GiB/s 19.312 GiB/s 19.313 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/8192
  time:   [205.16 ns 205.21 ns 205.26 ns]
  thrpt:  [37.170 GiB/s 37.178 GiB/s 37.188 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/8192
  time:   [442.83 ns 443.08 ns 443.45 ns]
  thrpt:  [34.409 GiB/s 34.438 GiB/s 34.457 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/8192
  time:   [406.08 ns 406.22 ns 406.34 ns]
  thrpt:  [37.552 GiB/s 37.563 GiB/s 37.576 GiB/s]

Benchmarking Hex_Performances/Decode/Std/8192
  time:   [50.370 µs 50.463 µs 50.585 µs]
  thrpt:  [308.88 MiB/s 309.63 MiB/s 310.21 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/8192
  time:   [1.3941 µs 1.3943 µs 1.3945 µs]
  thrpt:  [10.942 GiB/s 10.944 GiB/s 10.945 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/8192
  time:   [625.28 ns 625.42 ns 625.57 ns]
  thrpt:  [24.392 GiB/s 24.398 GiB/s 24.403 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/16384
  time:   [559.18 ns 559.61 ns 560.04 ns]
  thrpt:  [27.246 GiB/s 27.267 GiB/s 27.288 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/16384
  time:   [438.90 ns 439.24 ns 439.72 ns]
  thrpt:  [34.701 GiB/s 34.739 GiB/s 34.766 GiB/s]

Benchmarking Hex_Performances/Encode/Std/16384
  time:   [27.694 µs 27.699 µs 27.703 µs]
  thrpt:  [564.01 MiB/s 564.11 MiB/s 564.21 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/16384
  time:   [795.17 ns 795.33 ns 795.51 ns]
  thrpt:  [19.181 GiB/s 19.186 GiB/s 19.189 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/16384
  time:   [495.55 ns 495.92 ns 496.16 ns]
  thrpt:  [30.754 GiB/s 30.769 GiB/s 30.792 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/16384
  time:   [846.74 ns 847.18 ns 847.55 ns]
  thrpt:  [36.007 GiB/s 36.023 GiB/s 36.041 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/16384
  time:   [810.99 ns 811.40 ns 811.88 ns]
  thrpt:  [37.589 GiB/s 37.611 GiB/s 37.630 GiB/s]

Benchmarking Hex_Performances/Decode/Std/16384
  time:   [112.77 µs 113.27 µs 113.81 µs]
  thrpt:  [274.59 MiB/s 275.89 MiB/s 277.11 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/16384
  time:   [2.7777 µs 2.7788 µs 2.7802 µs]
  thrpt:  [10.977 GiB/s 10.982 GiB/s 10.986 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/16384
  time:   [1.2615 µs 1.2616 µs 1.2617 µs]
  thrpt:  [24.188 GiB/s 24.190 GiB/s 24.192 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/32768
  time:   [1.4412 µs 1.4416 µs 1.4421 µs]
  thrpt:  [21.162 GiB/s 21.169 GiB/s 21.176 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/32768
  time:   [1.3960 µs 1.3979 µs 1.4012 µs]
  thrpt:  [21.780 GiB/s 21.832 GiB/s 21.860 GiB/s]

Benchmarking Hex_Performances/Encode/Std/32768
  time:   [55.150 µs 55.162 µs 55.175 µs]
  thrpt:  [566.38 MiB/s 566.51 MiB/s 566.64 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/32768
  time:   [1.5864 µs 1.5865 µs 1.5866 µs]
  thrpt:  [19.235 GiB/s 19.236 GiB/s 19.238 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/32768
  time:   [1.1980 µs 1.1981 µs 1.1982 µs]
  thrpt:  [25.470 GiB/s 25.472 GiB/s 25.475 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/32768
  time:   [1.6604 µs 1.6611 µs 1.6616 µs]
  thrpt:  [36.733 GiB/s 36.744 GiB/s 36.758 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/32768
  time:   [1.6196 µs 1.6202 µs 1.6212 µs]
  thrpt:  [37.649 GiB/s 37.670 GiB/s 37.685 GiB/s]

Benchmarking Hex_Performances/Decode/Std/32768
  time:   [245.07 µs 245.78 µs 246.62 µs]
  thrpt:  [253.43 MiB/s 254.29 MiB/s 255.03 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/32768
  time:   [5.6279 µs 5.6284 µs 5.6289 µs]
  thrpt:  [10.843 GiB/s 10.844 GiB/s 10.845 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/32768
  time:   [2.5769 µs 2.5769 µs 2.5770 µs]
  thrpt:  [23.684 GiB/s 23.685 GiB/s 23.686 GiB/s]

Benchmarking Hex_Performances/Encode/Turbo/65536
  time:   [3.7327 µs 3.7365 µs 3.7410 µs]
  thrpt:  [16.315 GiB/s 16.335 GiB/s 16.352 GiB/s]

Benchmarking Hex_Performances/Encode/TurboBuff/65536
  time:   [3.7398 µs 3.7429 µs 3.7453 µs]
  thrpt:  [16.296 GiB/s 16.307 GiB/s 16.320 GiB/s]

Benchmarking Hex_Performances/Encode/Std/65536
  time:   [110.21 µs 110.26 µs 110.31 µs]
  thrpt:  [566.61 MiB/s 566.85 MiB/s 567.08 MiB/s]

Benchmarking Hex_Performances/Encode/FastBuff/65536
  time:   [3.1654 µs 3.1656 µs 3.1658 µs]
  thrpt:  [19.280 GiB/s 19.281 GiB/s 19.282 GiB/s]

Benchmarking Hex_Performances/Encode/SimdBuff/65536
  time:   [3.3977 µs 3.4086 µs 3.4191 µs]
  thrpt:  [17.851 GiB/s 17.906 GiB/s 17.964 GiB/s]

Benchmarking Hex_Performances/Decode/Turbo/65536
  time:   [3.2892 µs 3.2895 µs 3.2899 µs]
  thrpt:  [37.104 GiB/s 37.109 GiB/s 37.113 GiB/s]

Benchmarking Hex_Performances/Decode/TurboBuff/65536
  time:   [3.2531 µs 3.2535 µs 3.2540 µs]
  thrpt:  [37.513 GiB/s 37.520 GiB/s 37.525 GiB/s]

Benchmarking Hex_Performances/Decode/Std/65536
  time:   [524.46 µs 524.81 µs 525.16 µs]
  thrpt:  [238.02 MiB/s 238.18 MiB/s 238.34 MiB/s]

Benchmarking Hex_Performances/Decode/FastBuff/65536
  time:   [11.521 µs 11.522 µs 11.524 µs]
  thrpt:  [10.593 GiB/s 10.594 GiB/s 10.596 GiB/s]

Benchmarking Hex_Performances/Decode/SimdBuff/65536
  time:   [5.3511 µs 5.3516 µs 5.3521 µs]
  thrpt:  [22.808 GiB/s 22.810 GiB/s 22.812 GiB/s]

Model name: Intel(R) Xeon(R) Platinum 8488C
```
</details>
