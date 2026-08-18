# ❓ Frequently Asked Questions

## 🛡️ Safety & Verification

### Q: The crate uses `unsafe`. How can you claim it is safe?
**A:** We distinguish between "Safe Rust" (compiler-checked) and "Memory Safe" (mathematically proven).
While we use `unsafe` pointers and intrinsics to achieve raw speed, we rely on a **Formal Verification Pipeline**. We have mathematically proven via Kani and MIRI that for the verified paths, **no possible input** (from empty arrays to infinite streams) can trigger a buffer overflow, segfault, or panic via the public API.
**[Read the Verification Report](./verification.md)**

### Q: Can I crash the library by passing garbage data?
**A:** **No.**
The decoder is extremely resilient. If you pass invalid Hex strings, random binary noise, odd-length payloads, or malicious data, the library simply returns a `Result::Err`. It will **never** panic or cause Undefined Behavior (UB) as long as you use the Safe API.

### Q: What happens if I violate safety contracts in the internal `unsafe` API?
**A:** **You are responsible for the crash.**
The `unsafe` internal functions (exposed via the `unstable` feature) are raw tools for those who need to bypass bounds checks for absolute maximum performance. If you pass a null pointer or an invalid length to these functions, you violate the contract. We verify that *our* Safe API never violates these contracts, but we cannot protect you if you bypass the API.

### Q: Is AVX512 enabled by default?
**A:** **Yes.**
We detect CPU features at runtime. If your CPU supports AVX512 (specifically `avx512f` and `avx512bw`), we use it. If not, we gracefully downgrade to AVX2 or our Scalar fallback. There is no need to manually enable feature flags for hardware SIMD support.

## ⚡ Performance & Usage

### Q: Does this work on ARM (Apple Silicon / Raspberry Pi)?
**A:** **Yes, currently via the Scalar backend.**
The library compiles and runs perfectly on ARM.
*   **x86_64:** Automatically uses AVX512 or AVX2.
*   **ARM:** Currently falls back to our highly optimized Scalar implementation.
*   **Roadmap:** Native NEON support is planned for a future release.

### Q: How do I calculate the buffer size for `encode_into`?
**A:** Use the exact length calculators provided by the engine.
If you are avoiding allocations, you cannot guess the size.

```rust
use hex_turbo::LOWER_CASE;

// For Encoding:
let needed = LOWER_CASE.encoded_len(input.len());
let mut buf = vec![0u8; needed];

// For Decoding:
let exact_needed = LOWER_CASE.decoded_len(hex_input.len());
let mut buf = vec![0u8; exact_needed];
```

### Q: Is the Scalar fallback slow?
**A:** **No, it is highly optimized.**
Even without SIMD, our scalar implementation eliminates many bounds checks found in standard libraries by utilizing unrolled loops and clever bitwise arithmetic. While it won't hit the massive throughput of AVX512, it is heavily competitive with other standard Hex implementations.

### Q: Does this work on `no_std` / Embedded systems?
**A:** **Yes.**
Simply disable the default `std` feature in your `Cargo.toml`. The library does not require a heap allocator if you use the `_into` (slice-based) zero-allocation APIs.

```toml
[dependencies]
hex-turbo = { version = "0.1", default-features = false }
```

## 🔌 Compatibility & Ecosystem

### Q: Is the output compatible with the standard `hex` crate?
**A:** **Yes.**
We fully conform to standard Hex encoding (RFC 4648, Section 8).
*   `LOWER_CASE` engine: Standard lowercase hex (`[0-9a-f]`).
*   `UPPER_CASE` engine: Standard uppercase hex (`[0-9A-F]`).
Our decoder is automatically case-insensitive and handles mixed-case inputs flawlessly. You can swap `hex-turbo` into any project using standard Hex without breaking data compatibility.

### Q: Do you support `serde`?
**A:** **Not directly (yet).**
To keep compile times low and dependencies minimal, we do not include `serde` implementations by default. However, you can easily use `hex-turbo` inside a custom `serde` serializer/deserializer wrapper.

### Q: Why should I use this over raw C libraries?
**A:** **Memory Safety.**
Highly optimized C libraries might achieve absolute theoretical maximum speeds, but they are **Unsafe**. They rely on unchecked pointer arithmetic and carry the risk of buffer overflows. `hex-turbo` offers a strategic compromise: saturating memory bandwidth while maintaining **100% Rust Memory Safety guarantees**.

### Q: How can I trust this code?
**A:** **Trust the math, not the author.**
1.  Check the **[GitHub Actions](https://github.com/hacer-bark/hex-turbo/actions)** to see the live Kani/MIRI logs.
2.  Inspect the **GPG Signatures** on our commits.
3.  Read the **[Verification Report](./verification.md)** to understand our audit methodology.
