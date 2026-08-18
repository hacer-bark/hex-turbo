//! # Hex Turbo
//! 
//! [![Crates.io](https://img.shields.io/crates/v/hex-turbo.svg)](https://crates.io/crates/hex-turbo)
//! [![Documentation](https://docs.rs/hex-turbo/badge.svg)](https://docs.rs/hex-turbo)
//! [![License](https://img.shields.io/crates/l/hex-turbo.svg)](https://crates.io/crates/hex-turbo)
//! [![Kani Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/hex-turbo/verification.yml?label=Kani%20Verified)](https://github.com/hacer-bark/hex-turbo/actions/workflows/verification.yml)
//! [![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/hex-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/hex-turbo/actions/workflows/miri.yml)
//! 
//! **The fastest memory-safe Hex implementation.**
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
//! | **`serde`** | **No** | Enables support for serde and adds `hex_turbo::serde` functions. |
//! | **`unstable`** | **No** | Enables access to the raw, unsafe internal functions (e.g. `encode_avx2`). |
//! 
//! ## Safety & Verification
//! 
//! This crate utilizes `unsafe` code for SIMD intrinsics and pointer arithmetic to achieve maximum performance.
//! To ensure safety, we employ a "Swiss Cheese" model of verification layers:
//! 
//! * **Formal Verification (Kani):** Mathematical proofs ensure the kernels never read out of bounds or panic on any input (0..∞ bytes).
//! * **MIRI Audited:** All SIMD paths (AVX512, AVX2) and Scalar fallbacks are verified with **MIRI** (Undefined Behavior checker) in CI to ensure strict memory safety.
//! * **MemorySanitizer:** The codebase is audited with MSan to prevent logic errors derived from reading uninitialized memory.
//! * **Fuzzing:** The codebase is fuzz-tested via `cargo-fuzz` (2.5B+ iterations).
//! 
//! **[Learn More](https://github.com/hacer-bark/hex-turbo/blob/main/docs/verification.md)**: Details on our threat model and formal verification strategy.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![doc(issue_tracker_base_url = "https://github.com/hacer-bark/hex-turbo/issues/")]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![warn(unused_qualifications)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Use `serde` when enabled
#[cfg(feature = "serde")]
use ::serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
pub mod serde;

// Scalar implementation
mod scalar;

// SIMD implementation (x86/x86_64 only)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "simd")]
mod simd;

// ======================================================================
// ERROR DEFINITION
// ======================================================================

/// Errors that can occur during Hex encoding or decoding operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
            Error::InvalidLength => write!(f, "Invalid Hex input length (must be divisible by 2)"),
            Error::InvalidCharacter => write!(f, "Invalid character found in Hex input"),
            Error::BufferTooSmall => write!(f, "Destination buffer is too small"),
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    pub const fn encoded_len(&self, input_len: usize) -> usize { input_len * 2 }

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
    // TODO: MUST CHECK INPUT WHICH NOT DIVISIBLE BY 2!
    #[inline]
    #[must_use]
    pub const fn decoded_len(&self, input_len: usize) -> usize { input_len / 2 }

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
    pub fn encode_into<T: AsRef<[u8]>>(
        &self,
        input: T,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let input = input.as_ref();
        let len = input.len();

        if len == 0 { return Ok(0); }

        let req_len = Self::encoded_len(self, len);
        if output.len() < req_len { return Err(Error::BufferTooSmall); }

        // Pass the raw pointer to the dispatcher. 
        // SAFETY: We checked output.len() >= req_len above.
        unsafe { Self::encode_dispatch(self, input, output[..req_len].as_mut_ptr()) };

        Ok(req_len)
    }

    /// Decodes `input` into the provided `output` buffer.
    ///
    /// ## Returns
    ///
    /// * `Ok(usize)`: The number of bytes written to `output`.
    /// * `Err(Error)`: If the input is invalid or the buffer is too small.
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
    pub fn decode_into<T: AsRef<[u8]>>(
        &self,
        input: T,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let input = input.as_ref();
        let len = input.len();

        if len == 0 { return Ok(0); }
        if len % 2 != 0 { return Err(Error::InvalidLength); }

        let req_len = Self::decoded_len(self, len);
        if output.len() < req_len { return Err(Error::BufferTooSmall); }

        // SAFETY: We pass only verified data.
        unsafe { Self::decode_dispatch(self, input, output[..req_len].as_mut_ptr())? };

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

        // 1. Calculate EXACT required size. Hex encoding is deterministic.
        let len = Self::encoded_len(self, input.len());

        // 2. Allocate uninitialized buffer
        let mut out = Vec::with_capacity(len);

        unsafe {
            Self::encode_dispatch(self, input, out.as_mut_ptr());
            out.set_len(len);
            String::from_utf8_unchecked(out)
        }
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

        // 1. Calculate required size
        let len = Self::decoded_len(self, input.len());

        // 2. Allocate buffer
        let mut out = Vec::with_capacity(len);

        // 3. Decode
        match unsafe { Self::decode_dispatch(self, input, out.as_mut_ptr()) } {
            Ok(_) => {
                // 4. Shrink to fit the real data
                unsafe { out.set_len(len); }
                Ok(out)
            }
            Err(e) => {
                Err(e)
            }
        }
    }

    // ========================================================================
    // Internal Dispatchers
    // ========================================================================

    #[inline(always)]
    unsafe fn encode_dispatch(&self, input: &[u8], dst: *mut u8) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(feature = "simd")]
        {
            let len = input.len();

            // Smart degrade: If len < 64, skip AVX512.
            if len >= 64 
                && std::is_x86_feature_detected!("avx512f") 
                && std::is_x86_feature_detected!("avx512bw") 
            {
                unsafe { simd::encode_slice_avx512(&self.config, input, dst); }
                return;
            }

            // Smart degrade: If len < 32, skip AVX2.
            if len >= 32 && std::is_x86_feature_detected!("avx2") {
                unsafe { simd::encode_slice_avx2(&self.config, input, dst); }
                return;
            }
        }

        // Fallback: Scalar / Non-x86 / Short inputs
        unsafe { scalar::encode_slice_unsafe(&self.config, input, dst); }
    }

    #[inline(always)]
    unsafe fn decode_dispatch(&self, input: &[u8], dst: *mut u8) -> Result<(), Error> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(feature = "simd")]
        {
            let len = input.len();

            // Smart degrade: If len < 64, skip AVX512.
            if len >= 64 
                && std::is_x86_feature_detected!("avx512f") 
                && std::is_x86_feature_detected!("avx512bw") 
            {
                return unsafe { simd::decode_slice_avx512(input, dst) };
            }

            // Smart degrade: If len < 32, skip AVX2.
            if len >= 32 && std::is_x86_feature_detected!("avx2") {
                return unsafe { simd::decode_slice_avx2(input, dst) };
            }
        }

        // Fallback: Scalar / Non-x86 / Short inputs
        unsafe { scalar::decode_slice_unsafe(input, dst) }
    }
}

// ========================================================================
// Simple API
// ========================================================================

/// Simplified API which calls to `LOWER_CASE.encode(data)`
pub fn encode<T: AsRef<[u8]>>(data: T) -> String {
    LOWER_CASE.encode(data)
}

/// Simplified API which calls to `UPPER_CASE.encode(data)`
pub fn encode_upper<T: AsRef<[u8]>>(data: T) -> String {
    UPPER_CASE.encode(data)
}

/// Simplified API which calls to `LOWER_CASE.decode(data)`
/// 
/// **Note**: Input hex can be uppercase or lowercase.
pub fn decode<T: AsRef<[u8]>>(data: T) -> Result<Vec<u8>, Error> {
    LOWER_CASE.decode(data)
}
