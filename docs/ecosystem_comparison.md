# ⚖️ Ecosystem Comparison

This project references and benchmarks against several external Hex libraries. Below is an objective analysis of the current landscape, detailing performance characteristics, implementation details, and safety guarantees.

## 📊 Quick Feature Matrix

| Library | Language | SIMD | Verified Safety | Est. Throughput (AVX512) |
| :--- | :--- | :---: | :---: | :--- |
| **hex-turbo** | Rust | ✅ | ✅ (Kani/MIRI/MSan) | **~84.0 GiB/s** |
| [hex-simd](https://crates.io/crates/hex-simd) | Rust | ✅ | ❌ | ~35.4 GiB/s |
| [faster-hex](https://crates.io/crates/faster-hex) | Rust | ✅ | ❌ | ~36.8 GiB/s |
| [hex](https://crates.io/crates/hex) | Rust | ❌ | ✅ (Compiler) | ~0.3 GiB/s |

## The Rust Ecosystem

### 1. [hex](https://crates.io/crates/hex) (Standard)
The de facto standard library for Rust.
*   **Pros:** Rock-solid stability. Uses 100% Safe Rust. Zero `unsafe` blocks.
*   **Cons:** Low performance. Relies on scalar lookup tables.
*   **Verdict:** Use this if you absolutely cannot have `unsafe` code in your dependency tree and do not care about throughput.

### 2. [hex-simd](https://crates.io/crates/hex-simd)
A strong competitor using the `core::simd` (unstable) or hand-rolled intrinsics.
*   **Pros:** Good performance. Native Rust.
*   **Cons:** Slower than `hex-turbo` on AVX512. Uses `unsafe` logic that has not been formally audited by Kani or MIRI.
*   **Verdict:** A strong library, but `hex-turbo` supersedes it in both throughput and verification guarantees on modern hardware.

### 3. [faster-hex](https://crates.io/crates/faster-hex)
A veteran SIMD library for Hex.
*   **Pros:** Very fast on AVX2.
*   **Cons:** Lacks AVX512 support. Not formally verified.
*   **Verdict:** Excellent for AVX2-only hardware, but `hex-turbo` is faster on modern cloud/server CPUs.

---

> **🛡️ Final Safety Note:**
> With the exception of the standard `hex` crate (which uses Safe Rust), **none** of the alternative libraries listed above offer verified guarantees against Undefined Behavior (UB). `hex-turbo` is unique in bridging the gap between C-level speed and Rust-level formal safety.
