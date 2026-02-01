#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![doc(issue_tracker_base_url = "https://github.com/hacer-bark/hex-turbo/issues/")]
#![deny(unsafe_op_in_unsafe_fn)]
// #![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![warn(unused_qualifications)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Scalar implementation
mod scalar;
// SIMD implementation (x86/x86_64 only)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "simd")]
mod simd;

// ======================================================================
// ERROR DEFINITION
// ======================================================================

/// Errors that can occur during Base64 encoding or decoding operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The input length is invalid for Base64 decoding.
    ///
    /// Base64 encoded data (with padding) must strictly have a length divisible by 4.
    /// If the input string is truncated or has incorrect padding length, this error is returned.
    InvalidLength,

    /// An invalid character was encountered during decoding.
    ///
    /// This occurs if the input contains bytes that do not belong to the
    /// selected Base64 alphabet (e.g., symbols not in the standard set) or
    /// if padding characters (`=`) appear in invalid positions.
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
            Error::InvalidLength => write!(f, "Invalid Base64 input length (must be divisible by 4)"),
            Error::InvalidCharacter => write!(f, "Invalid character found in Base64 input"),
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

/// Internal configuration for the Base64 engine.
///
/// This struct uses `repr(C)` to ensure predictable memory layout.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Config {
    pub uppercase: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Engine {
    pub(crate) config: Config,
}

// ======================================================================
// Pre-defined Engines
// ======================================================================

pub const UPPER_CASE: Engine = Engine {
    config: Config {
        uppercase: true,
    },
};

pub const LOWER_CASE: Engine = Engine {
    config: Config {
        uppercase: false,
    },
};

impl Engine {
    // ======================================================================
    // Length Calculators
    // ======================================================================

    /// Calculates the exact buffer size required to encode `input_len` bytes.
    ///
    /// This method computes the size based on the current configuration (padding vs. no padding).
    ///
    /// # Examples
    ///
    /// ```
    /// use base64_turbo::STANDARD;
    ///
    /// assert_eq!(STANDARD.encoded_len(3), 4);
    /// assert_eq!(STANDARD.encoded_len(1), 4); // With padding
    /// ```
    #[inline]
    #[must_use]
    pub const fn encoded_len(&self, input_len: usize) -> usize { input_len * 2 }

    /// Calculates the **maximum** buffer size required to decode `input_len` bytes.
    ///
    /// # Note
    /// This is an upper-bound estimate. The actual number of bytes written during
    /// decoding will likely be smaller.
    ///
    /// You should rely on the `usize` returned by [`decode_into`](Self::decode_into)
    /// to determine the actual valid slice of the output buffer.
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
    /// # Parallelism
    /// If the `parallel` feature is enabled and the input size exceeds the
    /// internal threshold (default: 512KB), this method automatically uses
    /// Rayon to process chunks in parallel, saturating memory bandwidth.
    ///
    /// # Arguments
    ///
    /// * `input`: The binary data to encode.
    /// * `output`: A mutable slice to write the Base64 string into.
    ///
    /// # Returns
    ///
    /// * `Ok(usize)`: The actual number of bytes written to `output`.
    /// * `Err(Error::BufferTooSmall)`: If `output.len()` is less than [`encoded_len`](Self::encoded_len).
    #[inline]
    pub fn encode_into<T: AsRef<[u8]> + Sync>(
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
        if output.len() < req_len {
            return Err(Error::BufferTooSmall);
        }

        // --- Serial Path ---
        // Pass the raw pointer to the dispatcher. 
        // SAFETY: We checked output.len() >= req_len above.
        unsafe { Self::encode_dispatch(self, input, output[..req_len].as_mut_ptr()) };

        Ok(req_len)
    }

    /// Decodes `input` into the provided `output` buffer.
    ///
    /// # Performance
    /// Like encoding, this method supports automatic parallelization for large payloads.
    /// It verifies the validity of the Base64 input while decoding.
    ///
    /// # Returns
    ///
    /// * `Ok(usize)`: The actual number of bytes written to `output`.
    /// * `Err(Error)`: If the input is invalid or the buffer is too small.
    #[inline]
    pub fn decode_into<T: AsRef<[u8]> + Sync>(
        &self,
        input: T,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let input = input.as_ref();
        let len = input.len();

        if len == 0 {
            return Ok(0);
        }

        let req_len = Self::decoded_len(self, len);
        if output.len() < req_len {
            return Err(Error::BufferTooSmall);
        }

        // --- Serial Path ---
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
    /// # Examples
    ///
    /// ```
    /// use base64_turbo::STANDARD;
    /// let b64 = STANDARD.encode(b"hello");
    /// assert_eq!(b64, "aGVsbG8=");
    /// ```
    #[inline]
    #[cfg(feature = "std")]
    pub fn encode<T: AsRef<[u8]> + Sync>(&self, input: T) -> String {
        let input = input.as_ref();

        // 1. Calculate EXACT required size. Base64 encoding is deterministic.
        let len = Self::encoded_len(self, input.len());

        // 2. Allocate uninitialized buffer
        let mut out = Vec::with_capacity(len);

        // 3. Set length immediately
        // SAFETY: We are about to overwrite the entire buffer in `encode_into`.
        // We require a valid `&mut [u8]` slice for the internal logic (especially Rayon) to work.
        // Since `encode_into` guarantees it writes exactly `len` bytes or fails (and we panic on fail),
        // we won't expose uninitialized memory.
        #[allow(clippy::uninit_vec)]
        unsafe { out.set_len(len); }

        // 4. Encode
        // We trust our `encoded_len` math completely.
        Self::encode_into(self, input, &mut out).expect("Base64 logic error: buffer size mismatch");

        // 5. Convert to String
        // SAFETY: The Base64 alphabet consists strictly of ASCII characters,
        // which are valid UTF-8.
        unsafe { String::from_utf8_unchecked(out) }
    }

    /// Allocates a new `Vec<u8>` and decodes the input data into it.
    ///
    /// # Errors
    /// Returns `Error` if the input contains invalid characters or has an invalid length.
    ///
    /// # Examples
    ///
    /// ```
    /// use base64_turbo::STANDARD;
    /// let bytes = STANDARD.decode("aGVsbG8=").unwrap();
    /// assert_eq!(bytes, b"hello");
    /// ```
    #[inline]
    #[cfg(feature = "std")]
    pub fn decode<T: AsRef<[u8]> + Sync>(&self, input: T) -> Result<Vec<u8>, Error> {
        let input = input.as_ref();

        // 1. Calculate required size
        let max_len = Self::decoded_len(self, input.len());

        // 2. Allocate buffer
        let mut out = Vec::with_capacity(max_len);

        // 3. Set length to MAX
        // SAFETY: We temporarily expose uninitialized memory to the `decode_into` function
        // so it can write into the slice. We strictly sanitize the length in step 5.
        #[allow(clippy::uninit_vec)]
        unsafe { out.set_len(max_len); }

        // 4. Decode
        // `decode_into` handles parallel/serial dispatch and returns the `actual_len`.
        match Self::decode_into(self, input, &mut out) {
            Ok(actual_len) => {
                // 5. Shrink to fit the real data
                // SAFETY: `decode_into` reported it successfully wrote `actual_len` valid bytes.
                // We truncate the Vec to this length, discarding any trailing garbage/uninitialized memory.
                unsafe { out.set_len(actual_len); }
                Ok(out)
            }
            Err(e) => {
                // SAFETY: If an error occurred, we force the length to 0.
                // This prevents the caller from accidentally inspecting uninitialized memory
                // if they were to (incorrectly) reuse the Vec from a partial result.
                unsafe { out.set_len(0); }
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

            #[cfg(feature = "avx512")]
            // Smart degrade: If len < 64, don't bother checking AVX512 features or setting up ZMM register
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

            // // Smart degrade: If len < 16, skip SSE4.1 and go straight to scalar.
            // if len >= 16 && std::is_x86_feature_detected!("sse4.1")  {
            //     unsafe { simd::encode_slice_simd(&self.config, input, dst); }
            //     return;
            // }
        }

        // Fallback: Scalar / Non-x86 / Short inputs
        // Safety: Pointers verified by caller
        unsafe { scalar::encode_slice_unsafe(&self.config, input, dst); }
    }

    #[inline(always)]
    unsafe fn decode_dispatch(&self, input: &[u8], dst: *mut u8) -> Result<(), Error> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(feature = "simd")]
        {
            let len = input.len();

            #[cfg(feature = "avx512")]
            // Smart degrade: Don't enter AVX512 path if we don't have a full vector of input.
            if len >= 64 
                && std::is_x86_feature_detected!("avx512f") 
                && std::is_x86_feature_detected!("avx512bw") 
            {
                return unsafe { simd::decode_slice_avx512(input, dst) };
            }

            // Smart degrade: Fallback to AVX2 if len is between 32 and 64, or if AVX512 is missing.
            if len >= 32 && std::is_x86_feature_detected!("avx2") {
                return unsafe { simd::decode_slice_avx2(input, dst) };
            }

            // // Smart degrade: Fallback to SSE4.1 if len is between 16 and 32.
            // if len >= 16 && std::is_x86_feature_detected!("sse4.1")  {
            //     return unsafe { simd::decode_slice_simd(input, dst) };
            // }
        }

        // Fallback: Scalar / Non-x86 / Short inputs
        // Safety: Pointers verified by caller
        unsafe { scalar::decode_slice_unsafe(input, dst) }
    }
}
