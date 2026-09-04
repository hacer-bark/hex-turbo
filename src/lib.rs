//! # Hex Turbo
//!
//! [![Crates.io](https://img.shields.io/crates/v/hex-turbo.svg)](https://crates.io/crates/hex-turbo)
//! [![License](https://img.shields.io/crates/l/hex-turbo.svg)](https://crates.io/crates/hex-turbo)
//! [![Logic Tests](https://img.shields.io/github/actions/workflow/status/hacer-bark/hex-turbo/tests.yml?label=Logic%20Tests)](https://github.com/hacer-bark/hex-turbo/actions/workflows/tests.yml)
//! [![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/hex-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/hex-turbo/actions/workflows/miri.yml)
//!
//! **A Rust hex codec that peaks past 150 GiB/s, with its `unsafe` SIMD checked by MIRI
//! and `MSan`, not just by review.**
//!
//! `hex-turbo` targets high-throughput systems where CPU cycles are scarce and Undefined
//! Behavior is unacceptable. "Memory-safe" here is a specific, bounded claim: the
//! `unsafe` SIMD paths are checked by [MIRI](https://github.com/rust-lang/miri) (a strict
//! UB interpreter) on top of `MemorySanitizer` audits and a `cargo-fuzz` target — see the
//! "Safety & Verification" section below for what each layer does and does not cover.
//! This crate is **not** faster than unchecked C/assembly implementations and does not
//! claim to be.
//!
//! It picks the best kernel available at runtime: **AVX-512 VBMI** or **AVX2** on
//! `x86_64` via runtime CPU detection, and an optimized table-driven scalar kernel
//! everywhere else, in 100% safe Rust. `no_std` environments are supported.
//!
//! Dispatch degrades on *size* as well as on CPU flags: inputs under 64 bytes skip
//! AVX-512 VBMI and inputs under 32 bytes skip AVX2, so short payloads never pay for a
//! vector setup they cannot amortize.
//!
//! ### Basic API (Allocating)
//!
//! Standard usage for general applications. Requires the `std` feature (enabled by default).
//!
//! ```rust
//! # #[cfg(feature = "std")]
//! # {
//! use hex_turbo::LOWER_CASE;
//!
//! let data = b"Hello world";
//!
//! // Encode to String
//! let encoded = LOWER_CASE.encode(data);
//! assert_eq!(encoded, "48656c6c6f20776f726c64");
//!
//! // Decode to Vec<u8>
//! let decoded = LOWER_CASE.decode(&encoded).unwrap();
//! assert_eq!(decoded, data);
//! # }
//! ```
//!
//! ### Zero-Allocation API (Slice-based)
//!
//! For low-latency scenarios or `no_std` environments where heap allocation is undesirable.
//! These methods write directly into a user-provided mutable slice.
//!
//! ```rust
//! use hex_turbo::LOWER_CASE;
//!
//! let input = b"Raw bytes";
//! let mut output = [0u8; 64]; // Pre-allocated stack buffer
//!
//! // Returns Result<usize, Error> indicating bytes written
//! let len = LOWER_CASE.encode_slice(input, &mut output).unwrap();
//!
//! assert_eq!(&output[..len], b"526177206279746573");
//! ```
//!
//! ## Feature Flags
//!
//! Each x86 SIMD kernel is an independent knob, so a target can compile in only
//! what its CPUs are likely to support. Runtime detection still gates every call,
//! so a kernel the host lacks simply falls back to scalar.
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | **`std`** | **Yes** | `String`/`Vec` support. Disable for `no_std` (the slice APIs need no allocator). |
//! | **`avx2`** | **Yes** | AVX2 kernel + runtime detection on `x86`/`x86_64`. Implies `std`. |
//! | **`avx512-vbmi`** | **Yes** | AVX-512 VBMI fast-path kernel on `x86`/`x86_64`. Implies `std`. |
//! | **`simd`** | **Yes** | Convenience meta-feature — turns on `avx2` + `avx512-vbmi` at once. |
//! | **`unstable`** | **No** | Exposes the raw internal kernels (`encode_avx2`, `encode_avx512_vbmi`, …). The `*_scalar` accessors are safe. |
//!
//! If **no** SIMD kernel is enabled, the build is pure scalar Rust and the crate carries
//! `#![forbid(unsafe_code)]` — memory safety then holds by construction, with no `unsafe`
//! anywhere to audit.
//!
//! ## Safety & Verification
//!
//! We use `unsafe` SIMD intrinsics and raw pointer arithmetic, so rather than rely on
//! review alone we stack independent verification layers that cover each other's blind
//! spots:
//!
//! *   **MIRI:** All SIMD paths (AVX512-VBMI, AVX2) and the scalar fallback run under
//!     **MIRI** (an Undefined Behavior interpreter) in CI, covering every distinct code path
//!     at least once. Branch coverage, not exhaustive input coverage.
//! *   **`MemorySanitizer`:** The standard library is rebuilt with instrumentation to confirm
//!     we never branch on or emit uninitialized memory.
//! *   **Fuzzing:** a `cargo-fuzz` target drives both directions across every dispatch tier.
//! *   **Model checking:** not wired up. The Kani proofs that cover the sibling
//!     `base64-turbo` kernels have no counterpart here yet.
//!
//! **[Learn More](https://github.com/hacer-bark/hex-turbo#safety--verification)**: exactly what is proven, and what isn't.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![doc(issue_tracker_base_url = "https://github.com/hacer-bark/hex-turbo/issues/")]
#![cfg_attr(not(unsafe_simd), forbid(unsafe_code))]
#![forbid(elided_lifetimes_in_paths)]
// This crate casts pointers to wider SIMD vector types (`__m128i`, `__m256i`, `__m512i`)
// purely to call `_mm*_loadu_*`/`_mm*_storeu_*` intrinsics, which are explicitly
// documented to work on any alignment ("u" = unaligned).
#![allow(clippy::cast_ptr_alignment)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

#[cfg(all(doctest, feature = "std"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

// Scalar implementation
pub(crate) mod scalar;

// SIMD implementations, compiled when any vectorized kernel is enabled.
#[cfg(unsafe_simd)]
pub(crate) mod simd;

#[cfg(x86_simd)]
pub(crate) mod dispatch {
    //! CPU-feature resolution, done once instead of once per call.
    //!
    //! `std::is_x86_feature_detected!` is cached internally, but reading that
    //! cache is still a load plus a branch, it sits on the critical path
    //! immediately before a call that cannot be inlined (a `#[target_feature]`
    //! function never inlines into a caller that lacks the feature), and it is
    //! an atomic load, so a caller's loop cannot hoist it. Measured on Coffee
    //! Lake it cost ~6 core cycles on every `encode_slice`/`decode_slice` -- about
    //! a quarter of the total for a 32-byte payload.
    //!
    //! Instead each kernel slot starts out pointing at its own resolver. The
    //! first call through a slot picks the kernel for this CPU and overwrites
    //! the slot with it; every later call is one relaxed load and an indirect
    //! call through a target the branch predictor learns immediately. Racing
    //! threads may both resolve, which is harmless: they compute the same
    //! answer and store the same pointer.
    //!
    //! Slots are split by width because the two kernels have different minimum
    //! block sizes -- feeding a 32-byte payload to the AVX-512 kernel would just
    //! fall through to scalar.
    //!
    //! The AVX-512 tier requires `avx512vbmi` on top of `avx512f`/`avx512bw`:
    //! `vpmultishiftqb`, `vpermb` and `vpermi2b` are what make it worth having,
    //! and without them the kernel would be no better than the AVX2 one. A CPU
    //! with AVX-512 but no VBMI (Skylake-SP, Cascade Lake) takes the AVX2 path.

    use crate::{Config, Error, scalar, simd};
    use core::sync::atomic::{AtomicPtr, Ordering};

    type EncodeFn = unsafe fn(Config, &[u8], &mut [u8]);
    type DecodeFn = unsafe fn(&[u8], &mut [u8]) -> Result<(), Error>;

    /// Reads a resolved slot.
    ///
    /// The transmute turns the stored data pointer back into the function
    /// pointer it was made from. Only ever handed values produced by the
    /// matching `store` below, and function and data pointers share a
    /// representation on every target this module compiles for (`x86`/`x86_64`).
    macro_rules! slot {
        ($slot:ident, $ty:ty) => {
            #[inline(always)]
            unsafe fn load() -> $ty {
                unsafe { core::mem::transmute($slot.load(Ordering::Relaxed)) }
            }
        };
    }

    // --- Encode, >= 64 bytes ---

    static ENCODE_WIDE: AtomicPtr<()> = AtomicPtr::new(resolve_encode_wide as *mut ());

    unsafe fn resolve_encode_wide(config: Config, input: &[u8], dst: &mut [u8]) {
        let f: EncodeFn = 'pick: {
            #[cfg(feature = "avx512-vbmi")]
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vbmi")
            {
                break 'pick simd::encode_slice_avx512_vbmi;
            }
            #[cfg(feature = "avx2")]
            if std::is_x86_feature_detected!("avx2") {
                break 'pick simd::encode_slice_avx2;
            }
            scalar_encode
        };
        ENCODE_WIDE.store(f as *mut (), Ordering::Relaxed);
        unsafe { f(config, input, dst) }
    }

    #[inline(always)]
    pub(crate) fn encode_wide() -> EncodeFn {
        slot!(ENCODE_WIDE, EncodeFn);
        unsafe { load() }
    }

    // --- Encode, 32..64 bytes (AVX2 only; the VBMI kernel needs 64) ---

    #[cfg(feature = "avx2")]
    static ENCODE_NARROW: AtomicPtr<()> = AtomicPtr::new(resolve_encode_narrow as *mut ());

    #[cfg(feature = "avx2")]
    unsafe fn resolve_encode_narrow(config: Config, input: &[u8], dst: &mut [u8]) {
        let f: EncodeFn = if std::is_x86_feature_detected!("avx2") {
            simd::encode_slice_avx2
        } else {
            scalar_encode
        };
        ENCODE_NARROW.store(f as *mut (), Ordering::Relaxed);
        unsafe { f(config, input, dst) }
    }

    #[cfg(feature = "avx2")]
    #[inline(always)]
    pub(crate) fn encode_narrow() -> EncodeFn {
        slot!(ENCODE_NARROW, EncodeFn);
        unsafe { load() }
    }

    // --- Decode, >= 64 bytes ---

    static DECODE_WIDE: AtomicPtr<()> = AtomicPtr::new(resolve_decode_wide as *mut ());

    unsafe fn resolve_decode_wide(input: &[u8], dst: &mut [u8]) -> Result<(), Error> {
        let f: DecodeFn = 'pick: {
            #[cfg(feature = "avx512-vbmi")]
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vbmi")
            {
                break 'pick simd::decode_slice_avx512_vbmi;
            }
            #[cfg(feature = "avx2")]
            if std::is_x86_feature_detected!("avx2") {
                break 'pick simd::decode_slice_avx2;
            }
            scalar::decode_slice
        };
        DECODE_WIDE.store(f as *mut (), Ordering::Relaxed);
        unsafe { f(input, dst) }
    }

    #[inline(always)]
    pub(crate) fn decode_wide() -> DecodeFn {
        slot!(DECODE_WIDE, DecodeFn);
        unsafe { load() }
    }

    // --- Decode, 32..64 bytes (AVX2 only; the VBMI kernel needs 64) ---

    #[cfg(feature = "avx2")]
    static DECODE_NARROW: AtomicPtr<()> = AtomicPtr::new(resolve_decode_narrow as *mut ());

    #[cfg(feature = "avx2")]
    unsafe fn resolve_decode_narrow(input: &[u8], dst: &mut [u8]) -> Result<(), Error> {
        let f: DecodeFn = if std::is_x86_feature_detected!("avx2") {
            simd::decode_slice_avx2
        } else {
            scalar::decode_slice
        };
        DECODE_NARROW.store(f as *mut (), Ordering::Relaxed);
        unsafe { f(input, dst) }
    }

    #[cfg(feature = "avx2")]
    #[inline(always)]
    pub(crate) fn decode_narrow() -> DecodeFn {
        slot!(DECODE_NARROW, DecodeFn);
        unsafe { load() }
    }

    /// Adapter so the encode slots have one signature across every tier.
    /// Reached only on an x86 CPU with no AVX2 at all.
    unsafe fn scalar_encode(config: Config, input: &[u8], dst: &mut [u8]) {
        scalar::encode_slice(config, input, dst);
    }
}

// ======================================================================
// Error Definition
// ======================================================================

/// Errors that can occur during Hex encoding or decoding operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The input length is invalid for Hex decoding.
    ///
    /// Hex encoded data must strictly have an even length — two characters per byte.
    /// A truncated or otherwise odd-length input returns this error.
    InvalidLength,

    /// An invalid character was encountered during decoding.
    ///
    /// This occurs if the input contains bytes that do not belong to the
    /// selected Hex alphabet (e.g., symbols not in the standard set).
    InvalidCharacter,

    /// The provided output buffer is too small to hold the result.
    ///
    /// This error is returned by the slice APIs (e.g., `encode_slice`, `decode_slice`)
    /// when the destination slice passed by the user does not have enough capacity
    /// to store the encoded or decoded data.
    BufferTooSmall,
}

// Standard Display implementation for better error messages
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidLength => write!(f, "Invalid Hex input length (must be divisible by 2)"),
            Self::InvalidCharacter => write!(f, "Invalid character found in Hex input"),
            Self::BufferTooSmall => write!(f, "Destination buffer is too small"),
        }
    }
}

// Enable std::error::Error trait when the 'std' feature is active
#[cfg(feature = "std")]
impl std::error::Error for Error {}

// ======================================================================
// Configuration & Types
// ======================================================================

/// Internal configuration for the Hex engine.
///
/// This struct uses `repr(C)` to ensure predictable memory layout.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Config {
    pub uppercase: bool,
}

/// A high-performance, stateless Hex encoder/decoder.
///
/// This struct holds the configuration for encoding/decoding.
/// It is designed to be immutable and thread-safe.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "std")]
/// # {
/// use hex_turbo::LOWER_CASE;
///
/// let data = b"Hello world";
///
/// // Encode to String
/// let encoded = LOWER_CASE.encode(data);
/// assert_eq!(encoded, "48656c6c6f20776f726c64");
///
/// // Decode to Result<Vec<u8>, Error>
/// let decoded = LOWER_CASE.decode(&encoded).unwrap();
/// assert_eq!(decoded, data);
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Engine {
    pub(crate) config: Config,
}

// ======================================================================
// Pre-defined Engines
// ======================================================================

/// Hex encoder with RFC4648 Alphabet. **UPPER CASE**.
pub const UPPER_CASE: Engine = Engine {
    config: Config { uppercase: true },
};

/// Hex encoder with RFC4648 Alphabet. **LOWER CASE**.
pub const LOWER_CASE: Engine = Engine {
    config: Config { uppercase: false },
};

// ======================================================================
// Allocating-API helpers (std only)
//
// These isolate the one place the SIMD and scalar-only builds genuinely differ:
// the SIMD build already contains `unsafe`, so it skips zeroing and validation;
// the scalar-only build forbids `unsafe`, so it pays a linear pass for the same
// result. `encode`/`decode` themselves stay identical across both.
// ======================================================================

/// A `len`-byte buffer for a dispatcher to fill: uninitialized on SIMD builds.
#[cfg(all(feature = "std", unsafe_simd))]
#[inline]
fn spare(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    // SAFETY: the caller passes `out` straight to a dispatcher, which writes
    // every one of the `len` bytes (hex output length is exact in both
    // directions), so no uninitialized byte is ever observed.
    #[allow(clippy::uninit_vec)]
    unsafe {
        out.set_len(len);
    }
    out
}

/// A `len`-byte buffer for a dispatcher to fill: zeroed on the safe scalar build.
#[cfg(all(feature = "std", not(unsafe_simd)))]
#[inline]
fn spare(len: usize) -> Vec<u8> {
    vec![0u8; len]
}

/// Wraps encoder output (guaranteed ASCII) as a `String` without re-validating.
#[cfg(all(feature = "std", unsafe_simd))]
#[inline]
fn into_ascii_string(bytes: Vec<u8>) -> String {
    // SAFETY: the Hex alphabet is strictly ASCII, hence valid UTF-8.
    unsafe { String::from_utf8_unchecked(bytes) }
}

/// Safe-build counterpart: validate on the way out. The bytes are always ASCII,
/// so the happy path reuses the buffer's allocation and the `Err` arm is dead.
#[cfg(all(feature = "std", not(unsafe_simd)))]
#[inline]
fn into_ascii_string(bytes: Vec<u8>) -> String {
    match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    }
}

impl Engine {
    // ======================================================================
    // Length calculations
    // ======================================================================

    /// Calculates the exact buffer size required to encode `input_len` bytes.
    ///
    /// This method computes the size of encoded data.
    ///
    /// # Examples
    ///
    /// ```
    /// use hex_turbo::LOWER_CASE;
    ///
    /// assert_eq!(LOWER_CASE.encoded_len(3), 6);
    /// assert_eq!(LOWER_CASE.encoded_len(2), 4);
    /// ```
    #[inline]
    #[must_use]
    pub const fn encoded_len(&self, input_len: usize) -> usize {
        input_len.saturating_mul(2)
    }

    /// Calculates the **exact** buffer size required to decode `input_len` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use hex_turbo::LOWER_CASE;
    ///
    /// assert_eq!(LOWER_CASE.decoded_len(6), 3);
    /// assert_eq!(LOWER_CASE.decoded_len(4), 2);
    /// ```
    #[inline]
    #[must_use]
    pub const fn decoded_len(&self, input_len: usize) -> usize {
        input_len / 2
    }

    // ======================================================================
    // Slice APIs
    // ======================================================================

    /// Encodes `input` into the provided `output` buffer.
    ///
    /// This is a "Zero-Allocation" API designed for hot paths. It writes directly
    /// into the destination slice without creating intermediate `Vec`.
    ///
    /// # Arguments
    ///
    /// * `input`: The binary data to encode.
    /// * `output`: A mutable slice to write the Hex string into.
    ///
    /// # Returns
    ///
    /// * `Ok(usize)`: The number of bytes written to `output`.
    /// * `Err(Error::BufferTooSmall)`: If `output.len()` is less than [`encoded_len`](Self::encoded_len).
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferTooSmall`] if `output` cannot hold [`encoded_len`](Self::encoded_len)
    /// bytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "std")]
    /// # {
    /// use hex_turbo::LOWER_CASE;
    ///
    /// let data = b"Hello world";
    /// let mut buff = vec![0u8; LOWER_CASE.encoded_len(data.len())];
    ///
    /// // Encode to Result<usize, Error>
    /// LOWER_CASE.encode_slice(data, &mut buff).unwrap();
    /// assert_eq!(buff, b"48656c6c6f20776f726c64");
    /// # }
    /// ```
    #[inline]
    pub fn encode_slice<T: AsRef<[u8]>>(
        &self,
        input: T,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let input = input.as_ref();
        let len = input.len();

        if len == 0 {
            return Ok(0);
        }

        let req_len = Self::encoded_len(self, len);
        let Some(dst) = output.get_mut(..req_len) else {
            return Err(Error::BufferTooSmall);
        };

        Self::encode_dispatch(*self, input, dst);

        Ok(req_len)
    }

    /// Decodes `input` into the provided `output` buffer.
    ///
    /// # Returns
    ///
    /// * `Ok(usize)`: The number of bytes written to `output`.
    /// * `Err(Error)`: If the input is invalid or the buffer is too small.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLength`] if `input.len()` is odd, [`Error::InvalidCharacter`]
    /// if `input` contains a byte outside the Hex alphabet, or [`Error::BufferTooSmall`] if
    /// `output` cannot hold [`decoded_len`](Self::decoded_len) bytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "std")]
    /// # {
    /// use hex_turbo::LOWER_CASE;
    ///
    /// let data = b"Hello world";
    ///
    /// // Encode to String
    /// let encoded = LOWER_CASE.encode(data);
    /// assert_eq!(encoded, "48656c6c6f20776f726c64");
    ///
    /// let mut buff = vec![0u8; LOWER_CASE.decoded_len(encoded.len())];
    ///
    /// // Decode to Result<usize, Error>
    /// LOWER_CASE.decode_slice(&encoded, &mut buff).unwrap();
    /// assert_eq!(buff, data);
    /// # }
    /// ```
    ///
    /// **Note**: Input hex can be uppercase or lowercase.
    #[inline]
    pub fn decode_slice<T: AsRef<[u8]>>(
        &self,
        input: T,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let input = input.as_ref();
        let len = input.len();

        if len == 0 {
            return Ok(0);
        }
        if len % 2 != 0 {
            return Err(Error::InvalidLength);
        }

        let req_len = Self::decoded_len(self, len);
        let Some(dst) = output.get_mut(..req_len) else {
            return Err(Error::BufferTooSmall);
        };

        Self::decode_dispatch(input, dst)?;

        Ok(req_len)
    }

    // ========================================================================
    // Allocating APIs (std)
    // ========================================================================

    /// Allocates a new `String` and encodes the input data into it.
    ///
    /// This is the most convenient method for general usage.
    ///
    /// # Examples
    ///
    /// ```
    /// use hex_turbo::LOWER_CASE;
    /// let hex = LOWER_CASE.encode(b"hello");
    /// assert_eq!(hex, "68656c6c6f");
    /// ```
    #[inline]
    #[cfg(feature = "std")]
    pub fn encode<T: AsRef<[u8]>>(&self, input: T) -> String {
        let input = input.as_ref();

        // Hex encoding is deterministic, so this is the EXACT output size.
        // `spare` hands the dispatcher a full-length buffer (uninitialized on
        // SIMD builds, zeroed on the scalar-only safe build); the dispatcher
        // then overwrites every byte, and the output is pure ASCII.
        let mut out = spare(Self::encoded_len(self, input.len()));
        Self::encode_dispatch(*self, input, &mut out);
        into_ascii_string(out)
    }

    /// Allocates a new `Vec<u8>` and decodes the input data into it.
    ///
    /// # Errors
    /// Returns `Error` if the input contains invalid characters or has an invalid length.
    ///
    /// # Examples
    ///
    /// ```
    /// use hex_turbo::LOWER_CASE;
    /// let bytes = LOWER_CASE.decode("68656c6c6f").unwrap();
    /// assert_eq!(bytes, b"hello");
    /// ```
    ///
    /// **Note**: Input hex can be uppercase or lowercase.
    #[inline]
    #[cfg(feature = "std")]
    pub fn decode<T: AsRef<[u8]>>(&self, input: T) -> Result<Vec<u8>, Error> {
        let input = input.as_ref();

        if input.len() % 2 != 0 {
            return Err(Error::InvalidLength);
        }

        // Decoded length is exact: two characters per byte. On error the whole
        // buffer is dropped without exposing an unwritten byte.
        let mut out = spare(Self::decoded_len(self, input.len()));
        Self::decode_dispatch(input, &mut out)?;
        Ok(out)
    }

    /// Encodes `input` and appends it to `output`.
    #[inline]
    #[cfg(feature = "std")]
    pub fn encode_string<T: AsRef<[u8]>>(&self, input: T, output: &mut String) {
        output.push_str(&Self::encode(self, input));
    }

    /// Decodes `input` and appends the result to `output`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLength`] / [`Error::InvalidCharacter`] if `input` is not
    /// valid Hex.
    #[inline]
    #[cfg(feature = "std")]
    pub fn decode_vec<T: AsRef<[u8]>>(&self, input: T, output: &mut Vec<u8>) -> Result<(), Error> {
        let input = input.as_ref();
        if input.len() % 2 != 0 {
            return Err(Error::InvalidLength);
        }
        let start = output.len();
        let decoded = Self::decoded_len(self, input.len());
        output.resize(start.saturating_add(decoded), 0);
        match Self::decode_dispatch(input, &mut output[start..]) {
            Ok(()) => Ok(()),
            Err(e) => {
                output.truncate(start);
                Err(e)
            }
        }
    }

    // ========================================================================
    // Internal Dispatchers
    // ========================================================================

    #[inline]
    fn encode_dispatch(self, input: &[u8], dst: &mut [u8]) {
        #[cfg(x86_simd)]
        {
            let len = input.len();
            if len >= 64 {
                unsafe { (dispatch::encode_wide())(self.config, input, dst) };
                return;
            }
            #[cfg(feature = "avx2")]
            if len >= 32 {
                unsafe { (dispatch::encode_narrow())(self.config, input, dst) };
                return;
            }
        }

        // Fallback: Scalar / non-SIMD target / short inputs.
        scalar::encode_slice(self.config, input, dst);
    }

    #[inline]
    fn decode_dispatch(input: &[u8], dst: &mut [u8]) -> Result<(), Error> {
        #[cfg(x86_simd)]
        {
            let len = input.len();
            if len >= 64 {
                return unsafe { (dispatch::decode_wide())(input, dst) };
            }
            #[cfg(feature = "avx2")]
            if len >= 32 {
                return unsafe { (dispatch::decode_narrow())(input, dst) };
            }
        }

        // Fallback: Scalar / non-SIMD target / short inputs.
        scalar::decode_slice(input, dst)
    }

    // ========================================================================
    // Raw kernel accessors (`unstable`)
    // ========================================================================

    /// Raw access to the direct AVX2 encoding logic.
    ///
    /// # Safety
    ///
    /// - `dst` must hold at least `input.len() * 2` bytes. Prefer
    ///   [`Engine::encoded_len`] to compute it.
    /// - The caller must ensure the target CPU supports AVX2 at runtime. Running this on a CPU
    ///   without AVX2 causes an illegal instruction crash.
    ///
    /// Prefer the safe higher-level APIs (e.g. [`Engine::encode`]) unless you need this bypass.
    #[cfg(all(x86_simd, feature = "avx2", feature = "unstable"))]
    pub unsafe fn encode_avx2(&self, input: &[u8], dst: &mut [u8]) {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::encode_slice_avx2(self.config, input, dst) }
    }

    /// Raw access to the direct AVX2 decoding logic.
    ///
    /// # Safety
    ///
    /// - `dst` must hold at least `input.len() / 2` bytes. Prefer
    ///   [`Engine::decoded_len`] to compute it.
    /// - The caller must ensure the target CPU supports AVX2 at runtime. Running this on a CPU
    ///   without AVX2 causes an illegal instruction crash.
    ///
    /// Prefer the safe higher-level APIs (e.g. [`Engine::decode`]) unless you need this bypass.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCharacter`] if `input` is not valid Hex.
    #[cfg(all(x86_simd, feature = "avx2", feature = "unstable"))]
    pub unsafe fn decode_avx2(&self, input: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::decode_slice_avx2(input, dst)? };
        Ok(input.len() / 2)
    }

    /// Raw access to the direct AVX-512-VBMI encoding logic, the fastest kernel in the crate.
    ///
    /// # Safety
    ///
    /// - `dst` must hold at least `input.len() * 2` bytes. Prefer
    ///   [`Engine::encoded_len`] to compute it.
    /// - The caller must ensure the target CPU supports the `avx512f`, `avx512bw` and
    ///   `avx512vbmi` subsets at runtime. Running this without all three causes an illegal
    ///   instruction crash.
    ///
    /// Prefer the safe higher-level APIs (e.g. [`Engine::encode`]) unless you need this bypass.
    #[cfg(all(x86_simd, feature = "avx512-vbmi", feature = "unstable"))]
    pub unsafe fn encode_avx512_vbmi(&self, input: &[u8], dst: &mut [u8]) {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::encode_slice_avx512_vbmi(self.config, input, dst) }
    }

    /// Raw access to the direct AVX-512-VBMI decoding logic.
    ///
    /// # Safety
    ///
    /// - `dst` must hold at least `input.len() / 2` bytes. Prefer
    ///   [`Engine::decoded_len`] to compute it.
    /// - The caller must ensure the target CPU supports the `avx512f`, `avx512bw` and
    ///   `avx512vbmi` subsets at runtime. Running this without all three causes an illegal
    ///   instruction crash.
    ///
    /// Prefer the safe higher-level APIs (e.g. [`Engine::decode`]) unless you need this bypass.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCharacter`] if `input` is not valid Hex.
    #[cfg(all(x86_simd, feature = "avx512-vbmi", feature = "unstable"))]
    pub unsafe fn decode_avx512_vbmi(&self, input: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::decode_slice_avx512_vbmi(input, dst)? };
        Ok(input.len() / 2)
    }

    /// Raw access to the direct scalar encoding logic.
    ///
    /// Unlike the SIMD accessors, this is a **safe** function: the scalar kernel uses no
    /// `unsafe`, so every write is bounds-checked.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is smaller than `input.len() * 2` (a bounds check, not memory
    /// corruption). Size it with [`Engine::encoded_len`].
    #[cfg(feature = "unstable")]
    pub fn encode_scalar(&self, input: &[u8], dst: &mut [u8]) {
        scalar::encode_slice(self.config, input, dst);
    }

    /// Raw access to the direct scalar decoding logic.
    ///
    /// Like [`Engine::encode_scalar`], this is a **safe** function — the scalar kernel
    /// contains no `unsafe`, so a too-small `dst` panics on a bounds check rather than
    /// corrupting memory. Size `dst` with [`Engine::decoded_len`].
    ///
    /// # Panics
    ///
    /// Panics if `dst` is too small to hold the decoded output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCharacter`] if `input` is not valid Hex.
    #[cfg(feature = "unstable")]
    pub fn decode_scalar(&self, input: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
        scalar::decode_slice(input, dst)?;
        Ok(input.len() / 2)
    }
}

// ========================================================================
// Free functions
// ========================================================================

/// Encodes `data` with the lowercase RFC 4648 §8 alphabet.
#[cfg(feature = "std")]
#[inline]
pub fn encode<T: AsRef<[u8]>>(data: T) -> String {
    LOWER_CASE.encode(data)
}

/// Encodes `data` with the uppercase RFC 4648 §8 alphabet.
#[cfg(feature = "std")]
#[inline]
pub fn encode_upper<T: AsRef<[u8]>>(data: T) -> String {
    UPPER_CASE.encode(data)
}

/// Decodes `data`, accepting either case.
///
/// # Errors
///
/// Returns [`Error::InvalidLength`] / [`Error::InvalidCharacter`] if `data` is not
/// valid Hex.
#[cfg(feature = "std")]
#[inline]
pub fn decode<T: AsRef<[u8]>>(data: T) -> Result<Vec<u8>, Error> {
    LOWER_CASE.decode(data)
}
