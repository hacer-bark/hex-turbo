//! # Hex Turbo
//!
//! [![Crates.io](https://img.shields.io/crates/v/hex-turbo.svg)](https://crates.io/crates/hex-turbo)
//! [![Documentation](https://docs.rs/hex-turbo/badge.svg)](https://docs.rs/hex-turbo)
//! [![License](https://img.shields.io/crates/l/hex-turbo.svg)](https://crates.io/crates/hex-turbo)
//! [![Tests](https://img.shields.io/github/actions/workflow/status/hacer-bark/hex-turbo/tests.yml?label=Tests)](https://github.com/hacer-bark/hex-turbo/actions/workflows/tests.yml)
//! [![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/hex-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/hex-turbo/actions/workflows/miri.yml)
//!
//! **A Rust hex codec that peaks past 150 GiB/s, with its `unsafe` SIMD checked by MIRI
//! and `MSan`, not just by review.**
//!
//! `hex-turbo` is a production-grade library engineered for **High Frequency Trading (HFT)**, **Mission-Critical Servers**, and **Embedded Systems** where CPU cycles are scarce and Undefined Behavior (UB) is unacceptable.
//!
//! This crate provides runtime CPU detection to utilize **AVX512** and **AVX2** intrinsics.
//! It includes a highly optimized scalar fallback for non-SIMD targets and supports `no_std` environments.
//!
//! > ⚠️ **Minimum Supported Rust Version (MSRV):** This crate requires **Rust 1.89.0 or newer** due to reliance on stabilized AVX512 intrinsics in the standard library.
//!
//! ## Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! hex-turbo = "0.1"
//! ```
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
//! // Decode to Result<Vec<u8>, Error>
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
//! // Encode to String
//! let len = LOWER_CASE.encode_into(input, &mut output).unwrap();
//!
//! assert_eq!(&output[..len], b"526177206279746573");
//! ```
//!
//! ## Feature Flags
//!
//! This crate is highly configurable via Cargo features:
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | **`std`** | **Yes** | Enables `String` and `Vec` support. Disable this for `no_std` environments. |
//! | **`simd`** | **Yes** | Enables runtime detection for **AVX512** and **AVX2** intrinsics. If disabled or unsupported by hardware, the crate falls back to scalar logic. |
//! | **`unstable`** | **No** | Reserved for exposing the raw internal SIMD kernels. Currently a no-op. |
//!
//! ## Safety & Verification
//!
//! The SIMD kernels use `unsafe` for intrinsics and pointer arithmetic to achieve maximum
//! performance. The scalar kernel does not: it is plain safe Rust under
//! `#![forbid(unsafe_code)]`, and a build with `simd` disabled forbids `unsafe` crate-wide.
//! To ensure safety, we employ a "Swiss Cheese" model of verification layers:
//!
//! * **MIRI Audited:** All SIMD paths (AVX512, AVX2) and Scalar fallbacks are verified with **MIRI** (Undefined Behavior checker) in CI to ensure strict memory safety.
//! * **`MemorySanitizer`:** The codebase is audited with `MSan` to prevent logic errors derived from reading uninitialized memory.
//!
//! **[Learn More](https://github.com/hacer-bark/hex-turbo#safety--verification)**: Details on our threat model and formal verification strategy.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![doc(issue_tracker_base_url = "https://github.com/hacer-bark/hex-turbo/issues/")]
#![cfg_attr(not(unsafe_simd), forbid(unsafe_code))]
#![forbid(elided_lifetimes_in_paths)]
// This crate casts pointers to wider SIMD vector types (`__m128i`, `__m256i`, `__m512i`)
// purely to call `_mm*_loadu_*`/`_mm*_storeu_*` intrinsics, which are explicitly
// documented to work on any alignment ("u" = unaligned).
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::redundant_pub_crate)]
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
    //! Lake it cost ~6 core cycles on every `encode_into`/`decode_into` -- about
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
        let f: EncodeFn = if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
        {
            simd::encode_slice_avx512_vbmi
        } else if std::is_x86_feature_detected!("avx2") {
            simd::encode_slice_avx2
        } else {
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

    // --- Encode, 32..64 bytes ---

    static ENCODE_NARROW: AtomicPtr<()> = AtomicPtr::new(resolve_encode_narrow as *mut ());

    unsafe fn resolve_encode_narrow(config: Config, input: &[u8], dst: &mut [u8]) {
        let f: EncodeFn = if std::is_x86_feature_detected!("avx2") {
            simd::encode_slice_avx2
        } else {
            scalar_encode
        };
        ENCODE_NARROW.store(f as *mut (), Ordering::Relaxed);
        unsafe { f(config, input, dst) }
    }

    #[inline(always)]
    pub(crate) fn encode_narrow() -> EncodeFn {
        slot!(ENCODE_NARROW, EncodeFn);
        unsafe { load() }
    }

    // --- Decode, >= 64 bytes ---

    static DECODE_WIDE: AtomicPtr<()> = AtomicPtr::new(resolve_decode_wide as *mut ());

    unsafe fn resolve_decode_wide(input: &[u8], dst: &mut [u8]) -> Result<(), Error> {
        let f: DecodeFn = if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
        {
            simd::decode_slice_avx512_vbmi
        } else if std::is_x86_feature_detected!("avx2") {
            simd::decode_slice_avx2
        } else {
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

    // --- Decode, 32..64 bytes ---

    static DECODE_NARROW: AtomicPtr<()> = AtomicPtr::new(resolve_decode_narrow as *mut ());

    unsafe fn resolve_decode_narrow(input: &[u8], dst: &mut [u8]) -> Result<(), Error> {
        let f: DecodeFn = if std::is_x86_feature_detected!("avx2") {
            simd::decode_slice_avx2
        } else {
            scalar::decode_slice
        };
        DECODE_NARROW.store(f as *mut (), Ordering::Relaxed);
        unsafe { f(input, dst) }
    }

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
// ERROR DEFINITION
// ======================================================================

/// Errors that can occur during Hex encoding or decoding operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The input length is invalid for Hex decoding.
    ///
    /// Hex encoded data (with padding) must strictly have a length divisible by 4.
    /// If the input string is truncated or has incorrect padding length, this error is returned.
    InvalidLength,

    /// An invalid character was encountered during decoding.
    ///
    /// This occurs if the input contains bytes that do not belong to the
    /// selected Hex alphabet (e.g., symbols not in the standard set).
    InvalidCharacter,

    /// The provided output buffer is too small to hold the result.
    ///
    /// This error is returned by the zero-allocation APIs (e.g., `encode_into`, `decode_into`)
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
/// ## Examples
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
    // Length Calculators
    // ======================================================================

    /// Calculates the exact buffer size required to encode `input_len` bytes.
    ///
    /// This method computes the size of encoded data.
    ///
    /// ## Examples
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
    /// ## Examples
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
    // Zero-Allocation APIs
    // ======================================================================

    /// Encodes `input` into the provided `output` buffer.
    ///
    /// This is a "Zero-Allocation" API designed for hot paths. It writes directly
    /// into the destination slice without creating intermediate `Vec`.
    ///
    /// ## Arguments
    ///
    /// * `input`: The binary data to encode.
    /// * `output`: A mutable slice to write the Hex string into.
    ///
    /// ## Returns
    ///
    /// * `Ok(usize)`: The number of bytes written to `output`.
    /// * `Err(Error::BufferTooSmall)`: If `output.len()` is less than [`encoded_len`](Self::encoded_len).
    ///
    /// ## Errors
    ///
    /// Returns [`Error::BufferTooSmall`] if `output` cannot hold [`encoded_len`](Self::encoded_len)
    /// bytes.
    ///
    /// ## Examples
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
    /// LOWER_CASE.encode_into(data, &mut buff).unwrap();
    /// assert_eq!(buff, b"48656c6c6f20776f726c64");
    /// # }
    /// ```
    #[inline]
    pub fn encode_into<T: AsRef<[u8]>>(&self, input: T, output: &mut [u8]) -> Result<usize, Error> {
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
    /// ## Returns
    ///
    /// * `Ok(usize)`: The number of bytes written to `output`.
    /// * `Err(Error)`: If the input is invalid or the buffer is too small.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidLength`] if `input.len()` is odd, [`Error::InvalidCharacter`]
    /// if `input` contains a byte outside the Hex alphabet, or [`Error::BufferTooSmall`] if
    /// `output` cannot hold [`decoded_len`](Self::decoded_len) bytes.
    ///
    /// ## Examples
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
    /// LOWER_CASE.decode_into(&encoded, &mut buff).unwrap();
    /// assert_eq!(buff, data);
    /// # }
    /// ```
    ///
    /// **Note**: Input hex can be uppercase or lowercase.
    #[inline]
    pub fn decode_into<T: AsRef<[u8]>>(&self, input: T, output: &mut [u8]) -> Result<usize, Error> {
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
    /// ## Examples
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
    /// ## Errors
    /// Returns `Error` if the input contains invalid characters or has an invalid length.
    ///
    /// ## Examples
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
            if len >= 32 {
                return unsafe { (dispatch::decode_narrow())(input, dst) };
            }
        }

        // Fallback: Scalar / non-SIMD target / short inputs.
        scalar::decode_slice(input, dst)
    }
}

// ========================================================================
// Simple API
// ========================================================================

/// Simplified API which calls to `LOWER_CASE.encode(data)`
#[cfg(feature = "std")]
pub fn encode<T: AsRef<[u8]>>(data: T) -> String {
    LOWER_CASE.encode(data)
}

/// Simplified API which calls to `UPPER_CASE.encode(data)`
#[cfg(feature = "std")]
pub fn encode_upper<T: AsRef<[u8]>>(data: T) -> String {
    UPPER_CASE.encode(data)
}

/// Simplified API which calls to `LOWER_CASE.decode(data)`
///
/// ## Errors
///
/// See [`Engine::decode_into`].
///
/// **Note**: Input hex can be uppercase or lowercase.
#[cfg(feature = "std")]
pub fn decode<T: AsRef<[u8]>>(data: T) -> Result<Vec<u8>, Error> {
    LOWER_CASE.decode(data)
}
