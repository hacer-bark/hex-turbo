<div align="center">
  <h1>Hex Turbo</h1>
  <p><strong>A Rust hex codec with runtime AVX-512/AVX2 dispatch, whose <code>unsafe</code> kernels are checked by a model checker, not just by review.</strong></p>

  [![Crates.io](https://img.shields.io/crates/v/hex-turbo.svg?style=for-the-badge&color=fc8d62)](https://crates.io/crates/hex-turbo)
  [![License](https://img.shields.io/badge/license-0BSD-8da0cb.svg?style=for-the-badge)](#license)
  [![Kani Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/hex-turbo/verification.yml?label=Kani%20Verified&style=for-the-badge&color=e78ac3)](https://github.com/hacer-bark/hex-turbo/actions/workflows/verification.yml)
  [![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/hex-turbo/miri.yml?label=MIRI%20Verified&style=for-the-badge&color=66c2a5)](https://github.com/hacer-bark/hex-turbo/actions/workflows/miri.yml)
</div>

<br/>

> ⚠️ **Status: pre-0.2, under active development.** The API works and the CI gates below
> are real, but the numbers marked _TBD_ in this README are not measured yet on the
> current code — don't quote them, and don't treat this as a settled release.

`hex-turbo` targets high-throughput systems where CPU cycles are scarce and Undefined
Behavior is unacceptable. It picks the best kernel available at runtime:

* **x86_64:** AVX-512 (`avx512f` + `avx512bw`) or AVX2, via runtime CPU detection.
* **Everything else (including ARM):** a scalar kernel in **100% safe Rust** —
  `#![forbid(unsafe_code)]`, no raw pointers, no `get_unchecked`.

Dispatch also degrades on *size*, not just on CPU flags: inputs under 64 bytes skip
AVX-512 and inputs under 32 bytes skip AVX2, so short payloads don't pay for a vector
setup they can't amortize.

There's no NEON kernel yet, no WASM SIMD, and no alphabet beyond standard hex — if you
need any of those, this isn't that crate. See the [FAQ](#faq).

## Contents

- [Quick start](#quick-start)
- [Zero-allocation API](#zero-allocation-stack--no_std)
- [Feature flags](#feature-flags)
- [Compatibility & stability](#compatibility--stability)
- [Performance & architecture](#performance--architecture)
- [Benchmarks](#benchmarks)
- [Safety & verification](#safety--verification)
- [Ecosystem](#ecosystem)
- [FAQ](#faq)
- [License](#license)

## Quick start

```rust
use hex_turbo::LOWER_CASE;

let data = b"Hello world";
let encoded = LOWER_CASE.encode(data); // String
assert_eq!(encoded, "48656c6c6f20776f726c64");

let decoded = LOWER_CASE.decode(&encoded).unwrap(); // Vec<u8>
assert_eq!(decoded, data);
```

`UPPER_CASE` is the same engine with an uppercase alphabet. Decoding is case-insensitive
on either engine. Free functions `encode`, `encode_upper` and `decode` wrap the two
constants if you don't want to name an engine.

### Zero-Allocation (Stack / `no_std`)

For hot paths where heap allocation is too slow, write directly to stack buffers — the
`_into` APIs need no allocator. Size the buffers with `encoded_len`/`decoded_len` rather
than guessing:

```rust
use hex_turbo::LOWER_CASE;

let input = b"Raw bytes";

let mut enc_buf = [0u8; 64];
let enc_len = LOWER_CASE.encode_into(input, &mut enc_buf).unwrap();
assert_eq!(&enc_buf[..enc_len], b"526177206279746573");

let mut dec_buf = [0u8; 32];
let dec_len = LOWER_CASE.decode_into(&enc_buf[..enc_len], &mut dec_buf).unwrap();
assert_eq!(&dec_buf[..dec_len], input);
```

Both lengths are exact and `const`: `encoded_len(n) == n * 2`, `decoded_len(n) == n / 2`.
A buffer that's too small is an error (`Error::BufferTooSmall`), never a panic.

## Feature flags

| Feature | Default | Description |
| :--- | :---: | :--- |
| `std` | **Yes** | `String`/`Vec` support. Disable for `no_std` (the `_into` APIs need no allocator). |
| `simd` | **Yes** | AVX-512 and AVX2 kernels + runtime detection on x86/x86_64. Implies `std`. |
| `unstable` | **No** | Reserved for exposing the raw internal kernels. Currently a no-op — the kernels are still private. |

Runtime detection gates every SIMD call, so enabling `simd` on a host without the
instructions just falls through to scalar. Disable `simd` and the crate contains no
`unsafe` at all: the whole thing compiles under `#![forbid(unsafe_code)]`, with nothing
left to audit. `serde` support is deliberately not included:
the dependency tree is empty, and the `_into` APIs make a wrapper trivial to write.

## Compatibility & Stability

**MSRV: Rust 1.89.0.** We rely on recently stabilized AVX-512 intrinsics in `core` and do
not plan to lower this. Edition 2024.

The public API (`Engine`, the two engine constants, `Error`, the free functions) is
expected to stay source-compatible through the `0.1.x` line, but this is pre-1.0 software
under active development — treat anything behind `unstable` as free to change without
notice.

Output is standard hex (RFC 4648 §8): `LOWER_CASE` emits `[0-9a-f]`, `UPPER_CASE` emits
`[0-9A-F]`, and both decoders accept mixed case. It is a drop-in data-format match for
the `hex` crate.

## Performance & Architecture

<details>
<summary>Why it should be fast — per-kernel breakdown</summary>

The design goal is maximum throughput *within* Rust's safety guarantees: vectorized data
movement instead of byte-at-a-time lookup tables, with error detection pushed into
bitmasks *after* the vector op so the hot loop stays branchless.

* **Scalar — safe Rust, two different shapes.** The two directions are limited by
  opposite halves of the core, so they don't share a technique. *Encode* is
  load-limited: a 512-byte table maps one input byte straight to the two characters
  it encodes, which saturates both load ports at ~1.0 cycles/byte. *Decode* is
  ALU-limited, so it uses no table at all — it splits nibbles arithmetically inside
  a `u64` and validates by re-encoding the result and comparing, which is one branch
  per 8 characters. Four independent groups per iteration overlap those dependency
  chains. Bounds come from `chunks_exact` and slice splitting, which the optimizer
  turns back into an unchecked pointer walk.
* **AVX2.** Encode widens each input byte to a 16-bit lane with `vpmovzxbw` — the exact
  span of its two output characters — so a single `vpshufb` through the alphabet LUT emits
  both, in order, with no lane fixup. That halves the pressure on port 5, the one port
  every 256-bit shuffle competes for. Short inputs keep the older two-lookup-plus-`vpunpck`
  loop, which spends more port-5 uops but reaches its floor on the first iteration.
  Decode classifies a character and derives its nibble offset from two LUT lookups at once,
  pairs characters with `vpmaddubs`, and packs with `vpackuswb`; validity is folded across
  the whole input with `vpminub` and inspected once at the end, so no iteration carries a
  branch.
* **AVX-512.** Same shape at 512 bits (`vpshufb`/`vpmaddubs`/`vpmovwb`), with 32 `zmm`
  registers keeping the LUTs and constants resident across an unrolled loop. Requires
  `avx512f` + `avx512bw`; both are checked at runtime.
* **Dispatch.** x86 picks AVX-512 (≥64 B) → AVX2 (≥32 B) → scalar, at runtime, guarding
  against `SIGILL`; every other architecture compiles down to scalar only. Feature
  detection runs *once*: each kernel slot starts out pointing at its own resolver, which
  overwrites the slot with the kernel this CPU should use. Steady state is one relaxed
  load and an indirect call — re-running `is_x86_feature_detected!` per call cost ~6 core
  cycles, which a 32-byte payload cannot absorb.

</details>

## Benchmarks

Numbers on the current code are **not measured yet** — the harness exists, the results
don't. Nothing here is a claim until this section carries real output.

| Payload | Encode | Decode | vs `hex-simd` | vs `faster-hex` | vs `hex` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 32 B | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 4 KiB | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 64 KiB | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 10 MB | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Methodology, once numbers land: [criterion.rs](https://github.com/bheisler/criterion.rs),
5 s warm-up, 15 s measurement per group, 50 samples, input sizes spanning 32 B → 10 MB to
cross L1/L2/RAM boundaries, compared against `hex-simd`, `faster-hex` and `hex` on their
default features in the same session on the same box.

Reproduce it:

```bash
git clone https://github.com/hacer-bark/hex-turbo
cd hex-turbo
BENCH_TARGET=all cargo bench
```

Select comparison targets with `BENCH_TARGET` (comma-separated): `turbo` (default,
allocating API), `turbo-buff` (zero-allocation API), `simd` (`hex-simd`), `fast`
(`faster-hex`), `std` (the `hex` crate), `all`.

## Safety & Verification

**Philosophy:** `Safety > Performance > Convenience`. The scalar kernel is plain safe
Rust and carries `#![forbid(unsafe_code)]`, so on a scalar-only build there is no
`unsafe` in the crate to audit. The vector kernels are a different matter — SIMD
intrinsics and raw pointer arithmetic — so rather than rely on review alone we stack
independent layers that cover each other's blind spots.

| Kernel | MIRI | MSan | Kani | Fuzzing |
| :--- | :---: | :---: | :---: | :---: |
| **Scalar (safe Rust)** | ✅ | ✅ | ✅ | ⏳ |
| **AVX2** | ✅ | ✅ | ✅ | ⏳ |
| **AVX-512** | ✅ | ✅ | ✅ | ⏳ |

* **Kani** runs two proofs per kernel: a roundtrip proof (`decode(encode(x)) == x` over a
  fully symbolic input) and a robustness proof (arbitrary garbage in, no panic, no
  out-of-bounds write, `Err` out). Both use an induction length chosen to execute one full
  vector iteration plus the scalar transition — 32+1 for AVX2 — so the block-linear
  structure of hex carries the result to every length. Kani can't execute SIMD, so each
  intrinsic is replaced by a hand-written Rust stub transcribed from the Intel Intrinsics
  Guide; those stubs are the weakest link in the chain (see below).
* **MIRI** catches Undefined Behavior (provenance, alignment, out-of-bounds pointer
  arithmetic, data races) across the test suite in CI, built with
  `-C target-feature=+avx512f,+avx512bw` so the vector paths are actually reached. Branch
  coverage, not exhaustive input coverage.
* **MSan** rebuilds the standard library with instrumentation
  (`-Z build-std -Z sanitizer=memory`) to confirm we never branch on or emit uninitialized
  memory.
* **Fuzzing** is planned and not wired up yet — there is no `cargo-fuzz` target in the
  repo today.

Every merge to `main` must pass Kani, MIRI, MSan and the logic tests; see the
[CI workflows](https://github.com/hacer-bark/hex-turbo/actions).

<details>
<summary>What still rests on human judgment</summary>

1. **The intrinsic stubs.** Kani proves the kernels against hand-written models of
   `_mm256_shuffle_epi8`, `_mm256_maddubs_epi16`, `_mm512_cvtepi16_epi8` and friends, not
   against the silicon. A transcription error in a stub is a hole in the proof. There is
   no stub-vs-hardware equivalence test in this repo yet — that's a known gap.
2. **The induction argument.** The proofs execute one vector iteration plus the transition
   into the scalar tail; the step from "one iteration is safe" to "every length is safe"
   is a structural argument about the loop, made by hand. Restructure the loop and the
   argument needs redoing.
3. **No fuzzing yet**, so nothing is currently probing the paths the proofs' stubs
   abstract away.

Read the `unsafe` blocks themselves — each documents the contract it relies on.

</details>

## Ecosystem

| Library | Lang | SIMD | Verified `unsafe` | Encode | Decode |
| :--- | :---: | :---: | :---: | ---: | ---: |
| **hex-turbo** | Rust | ✅ AVX-512 + AVX2 | ✅ Kani + MIRI + MSan | _TBD_ | _TBD_ |
| [hex-simd](https://crates.io/crates/hex-simd) | Rust | ✅ | ❌ | _TBD_ | _TBD_ |
| [faster-hex](https://crates.io/crates/faster-hex) | Rust | ✅ (AVX2) | ❌ | _TBD_ | _TBD_ |
| [hex](https://crates.io/crates/hex) | Rust | ❌ scalar | ✅ safe Rust | _TBD_ | _TBD_ |

Throughput columns stay empty until we have benchmarks on the current code; the
implementation and verification columns are what we can state today. `hex-simd` and
`faster-hex` are both good crates and neither publishes Kani or MSan coverage of its
`unsafe` as far as we could find — that verification depth, not a speed claim, is what
this crate is currently arguing for. The `hex` crate remains the right answer if you want
zero `unsafe` in your dependency tree and don't care about throughput.

## FAQ

**Is this production-ready?**
Not yet. It's pre-1.0, benchmarks aren't rerun on the current code, and fuzzing isn't set
up. The CI gates (Kani, MIRI, MSan, tests) are real and enforced on every PR.

**Does it work on ARM / Apple Silicon?**
It compiles and runs, on the safe scalar kernel. There is no NEON kernel — a native ARM
path is on the roadmap, not in the crate.

**What happens on an x86 CPU without AVX-512 or AVX2?**
Runtime detection falls back automatically — AVX-512 → AVX2 → scalar, no crash and no
feature gating at the call site. You lose throughput, not correctness.

**Does it work in `no_std` / embedded?**
Yes. Disable default features and use the `_into` APIs; no allocator is required.

```toml
[dependencies]
hex-turbo = { version = "0.1", default-features = false }
```

**Can I crash the decoder with garbage input?**
No. Invalid characters, odd lengths, and arbitrary binary noise return `Err`, never a
panic or UB — that's what the Kani robustness proofs cover, and the decoder's validity
check is exercised against every one of the 256 byte values in every position of a block
by `tests/differential.rs`.

**Why is `unsafe` acceptable here at all?**
Only the vector kernels use it — vectorized hex can't be written in safe Rust, because
the intrinsics themselves require `unsafe`. The scalar kernel needs none and has none.
For the vector paths the answer is to prove the `unsafe` correct with independent tools
rather than ask you to trust a code review; if you'd rather carry none of it, disable
`simd` and the crate is `#![forbid(unsafe_code)]` end to end.

**Do you support `serde`?**
No. It was removed to keep the dependency tree empty; wrap the `_into` APIs in your own
`serialize_with`/`deserialize_with` if you need it.

## License

Licensed under the [0BSD license](https://github.com/hacer-bark/hex-turbo/blob/main/LICENSE).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this crate shall be licensed as above, without any additional terms or conditions.
