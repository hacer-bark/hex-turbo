use crate::{Config, Error, scalar};

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m256i, __m512i, _mm256_storeu_si256, _mm512_add_epi8, _mm512_and_si512,
    _mm512_cmpeq_epi8_mask, _mm512_cvtepi16_epi8, _mm512_loadu_si512, _mm512_maddubs_epi16,
    _mm512_set1_epi8, _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_shuffle_i32x4,
    _mm512_srli_epi16, _mm512_storeu_si512, _mm512_unpackhi_epi8, _mm512_unpacklo_epi8,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, __m512i, _mm256_storeu_si256, _mm512_add_epi8, _mm512_and_si512,
    _mm512_cmpeq_epi8_mask, _mm512_cvtepi16_epi8, _mm512_loadu_si512, _mm512_maddubs_epi16,
    _mm512_set1_epi8, _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_shuffle_i32x4,
    _mm512_srli_epi16, _mm512_storeu_si512, _mm512_unpackhi_epi8, _mm512_unpacklo_epi8,
};

// --- CONSTANTS ---

// Duplicated 16-byte tables for AVX512 pshufb (Encoding)
const HEX_TABLE_UPPER: [u8; 64] =
    *b"0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF";

const HEX_TABLE_LOWER: [u8; 64] =
    *b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

// Duplicated 16-byte LUTs and weights for AVX512 (Decoding)
const LUT_HI: [u8; 64] = [
    0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 73, 0, 73, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
];
const LUT_LO: [u8; 64] = [
    128, 192, 192, 192, 192, 192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0, 128, 192, 192, 192, 192,
    192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0, 128, 192, 192, 192, 192, 192, 192, 128, 128, 128, 0,
    0, 0, 0, 0, 0, 128, 192, 192, 192, 192, 192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0,
];
const WEIGHTS: [u8; 64] = [
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16,
    1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
];

// --- ENCODING ---

/// # Safety
/// `dst_slice` must hold at least `input.len() * 2` bytes.
// `input[processed_len..]` is guarded by `processed_len < len` just above,
// and `dst_slice[dst_off..]` is proven in bounds by the SAFETY comment at
// its call site (`dst` walked forward from `dst_start` within `dst_slice`).
#[allow(clippy::indexing_slicing)]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn encode_slice_avx512(config: Config, input: &[u8], dst_slice: &mut [u8]) {
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let len = input.len();
    let mut src = input.as_ptr();
    let start_ptr = input.as_ptr();

    // Select and load the appropriate 32-byte shuffle table
    let table_ptr = if config.uppercase {
        HEX_TABLE_UPPER.as_ptr()
    } else {
        HEX_TABLE_LOWER.as_ptr()
    };
    let table = unsafe { _mm512_loadu_si512(table_ptr.cast::<__m512i>()) };
    let mask_0f = _mm512_set1_epi8(0x0F);

    macro_rules! encode_vec {
        ($in_vec:expr) => {{
            // Extract nibbles
            let low_idx = _mm512_and_si512($in_vec, mask_0f);
            let high_idx = _mm512_and_si512(_mm512_srli_epi16($in_vec, 4), mask_0f);

            // LUT lookup
            let low_chars = _mm512_shuffle_epi8(table, low_idx);
            let high_chars = _mm512_shuffle_epi8(table, high_idx);

            // Unpack 8-bit to 16-bit (High char, Low char)
            let inter_lo = _mm512_unpacklo_epi8(high_chars, low_chars);
            let inter_hi = _mm512_unpackhi_epi8(high_chars, low_chars);

            (inter_lo, inter_hi)
        }};
    }

    macro_rules! store_128_bytes {
        ($dst:expr, $inter_lo:expr, $inter_hi:expr) => {{
            // Reorder the data using 128-bit lane shuffles
            let tmp1 = _mm512_shuffle_i32x4::<0x44>($inter_lo, $inter_hi);
            let ordered_1 = _mm512_shuffle_i32x4::<0xD8>(tmp1, tmp1);

            let tmp2 = _mm512_shuffle_i32x4::<0xEE>($inter_lo, $inter_hi);
            let ordered_2 = _mm512_shuffle_i32x4::<0xD8>(tmp2, tmp2);

            unsafe {
                _mm512_storeu_si512($dst.cast::<_>(), ordered_1);
                _mm512_storeu_si512($dst.add(64).cast::<_>(), ordered_2);
            }
        }};
    }

    // --- Large unrolled loop: 256 input bytes (512 output bytes) ---
    let limit_256 = (len / 256) * 256;
    let src_end_256 = unsafe { start_ptr.add(limit_256) };

    while src < src_end_256 {
        let v0 = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let v1 = unsafe { _mm512_loadu_si512(src.add(64).cast::<__m512i>()) };
        let v2 = unsafe { _mm512_loadu_si512(src.add(128).cast::<__m512i>()) };
        let v3 = unsafe { _mm512_loadu_si512(src.add(192).cast::<__m512i>()) };

        let (lo0, hi0) = encode_vec!(v0);
        let (lo1, hi1) = encode_vec!(v1);
        let (lo2, hi2) = encode_vec!(v2);
        let (lo3, hi3) = encode_vec!(v3);

        store_128_bytes!(dst, lo0, hi0);
        store_128_bytes!(dst.add(128), lo1, hi1);
        store_128_bytes!(dst.add(256), lo2, hi2);
        store_128_bytes!(dst.add(384), lo3, hi3);

        src = unsafe { src.add(256) };
        dst = unsafe { dst.add(512) };
    }

    // --- Small loop: 64 input bytes (128 output bytes) ---
    let limit_64 = (len / 64) * 64;
    let src_end_64 = unsafe { start_ptr.add(limit_64) };

    while src < src_end_64 {
        let v = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let (lo, hi) = encode_vec!(v);

        store_128_bytes!(dst, lo, hi);

        src = unsafe { src.add(64) };
        dst = unsafe { dst.add(128) };
    }

    // --- Scalar Fallback ---
    let processed_len = unsafe { src.offset_from(start_ptr) }.cast_unsigned();
    if processed_len < len {
        // SAFETY: `dst` walked forward from `dst_start` within `dst_slice`.
        let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
        scalar::encode_slice(config, &input[processed_len..], &mut dst_slice[dst_off..]);
    }
}

// --- DECODING ---

/// # Safety
/// `dst_slice` must hold at least `input.len() / 2` bytes.
// Same reasoning as `encode_slice_avx512`.
#[allow(clippy::indexing_slicing)]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn decode_slice_avx512(input: &[u8], dst_slice: &mut [u8]) -> Result<(), Error> {
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let len = input.len();
    let mut src = input.as_ptr();

    // Load 64-byte LUTs into 512-bit registers
    let lut_hi = unsafe { _mm512_loadu_si512(LUT_HI.as_ptr().cast::<__m512i>()) };
    let lut_lo = unsafe { _mm512_loadu_si512(LUT_LO.as_ptr().cast::<__m512i>()) };
    let weights = unsafe { _mm512_loadu_si512(WEIGHTS.as_ptr().cast::<__m512i>()) };

    let mask_0f = _mm512_set1_epi8(0x0F);
    let zero = _mm512_setzero_si512();

    macro_rules! decode_hex_vec {
        ($input:expr) => {{
            // Split Nibbles
            let lo = _mm512_and_si512($input, mask_0f);
            let hi = _mm512_and_si512(_mm512_srli_epi16($input, 4), mask_0f);

            // LUT Lookups
            let hi_props = _mm512_shuffle_epi8(lut_hi, hi);
            let lo_props = _mm512_shuffle_epi8(lut_lo, lo);

            // Validation
            let valid_flags = _mm512_and_si512(hi_props, lo_props);
            let err_mask = _mm512_cmpeq_epi8_mask(valid_flags, zero);

            // Value Calculation
            let offset = _mm512_and_si512(hi_props, mask_0f);
            let nibbles = _mm512_add_epi8(lo, offset);

            // 5. Pack to Bytes
            let pairs_i16 = _mm512_maddubs_epi16(nibbles, weights);
            let result_256 = _mm512_cvtepi16_epi8(pairs_i16);

            (result_256, err_mask)
        }};
    }

    // --- Large unrolled loop: 256 input bytes (128 output bytes) ---
    let limit_256 = (len / 256) * 256;
    let src_end_256 = unsafe { input.as_ptr().add(limit_256) };

    while src < src_end_256 {
        let v0 = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let v1 = unsafe { _mm512_loadu_si512(src.add(64).cast::<__m512i>()) };
        let v2 = unsafe { _mm512_loadu_si512(src.add(128).cast::<__m512i>()) };
        let v3 = unsafe { _mm512_loadu_si512(src.add(192).cast::<__m512i>()) };

        let (r0, e0) = decode_hex_vec!(v0);
        let (r1, e1) = decode_hex_vec!(v1);
        let (r2, e2) = decode_hex_vec!(v2);
        let (r3, e3) = decode_hex_vec!(v3);

        // Accumulate error masks
        if (e0 | e1 | e2 | e3) != 0 {
            return Err(Error::InvalidCharacter);
        }

        unsafe {
            _mm256_storeu_si256(dst.cast::<__m256i>(), r0);
            _mm256_storeu_si256(dst.add(32).cast::<__m256i>(), r1);
            _mm256_storeu_si256(dst.add(64).cast::<__m256i>(), r2);
            _mm256_storeu_si256(dst.add(96).cast::<__m256i>(), r3);

            src = src.add(256);
            dst = dst.add(128);
        }
    }

    // --- Small loop: 64 input bytes (32 output bytes) ---
    let safe_len = len.saturating_sub(63);
    let src_end = unsafe { input.as_ptr().add(safe_len) };

    while src < src_end {
        let v = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let (res, err) = decode_hex_vec!(v);

        if err != 0 {
            return Err(Error::InvalidCharacter);
        }

        unsafe {
            _mm256_storeu_si256(dst.cast::<__m256i>(), res);
            src = src.add(64);
            dst = dst.add(32);
        }
    }

    // --- Scalar fallback ---
    let processed_len = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
    if processed_len < len {
        // SAFETY: `dst` walked forward from `dst_start` within `dst_slice`.
        let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
        scalar::decode_slice(&input[processed_len..], &mut dst_slice[dst_off..])?;
    }

    Ok(())
}

// Verification: Kani proofs, intrinsic stubs, and the Miri coverage suite.
#[cfg(any(kani, test))]
mod verify;
