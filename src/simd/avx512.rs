use crate::{Error, Config, scalar};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const HEX_TABLE_UPPER: [u8; 64] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
];

const HEX_TABLE_LOWER: [u8; 64] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
];

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn encode_slice_avx512(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();
    let start_ptr = input.as_ptr();

    // Select and load the appropriate 32-byte shuffle table
    let table_ptr = if config.uppercase { HEX_TABLE_UPPER.as_ptr() } else { HEX_TABLE_LOWER.as_ptr() };
    let table = unsafe { _mm512_loadu_si512(table_ptr as *const __m512i) };

    let mask_0f = _mm512_set1_epi8(0x0F);

    // 2. Permutation Indices
    // These shuffle 64-bit blocks to restore linear order from the 128-bit interleaved lanes.
    // Indices 0-7 refer to 'inter_lo', 8-15 refer to 'inter_hi'.
    let perm_idx_1 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let perm_idx_2 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    macro_rules! encode_vec {
        ($in_vec:expr) => {{
            // Extract nibbles
            let low_idx  = _mm512_and_si512($in_vec, mask_0f);
            let high_idx = _mm512_and_si512(_mm512_srli_epi16($in_vec, 4), mask_0f);

            // LUT lookup
            let low_chars  = _mm512_shuffle_epi8(table, low_idx);
            let high_chars = _mm512_shuffle_epi8(table, high_idx);

            // Unpack 8-bit to 16-bit (High char, Low char)
            let inter_lo = _mm512_unpacklo_epi8(high_chars, low_chars);
            let inter_hi = _mm512_unpackhi_epi8(high_chars, low_chars);

            (inter_lo, inter_hi)
        }};
    }

    macro_rules! store_128_bytes {
        ($dst:expr, $inter_lo:expr, $inter_hi:expr) => {{
            // Reorder the data using the permutation indices
            let ordered_1 = _mm512_permutex2var_epi64($inter_lo, perm_idx_1, $inter_hi);
            let ordered_2 = _mm512_permutex2var_epi64($inter_lo, perm_idx_2, $inter_hi);

            unsafe {
                _mm512_storeu_si512($dst as *mut _, ordered_1);
                _mm512_storeu_si512($dst.add(64) as *mut _, ordered_2);
            }
        }};
    }

    // --- Loop: Process 64 input bytes (128 output bytes) ---
    let limit_64 = (len / 64) * 64;
    let src_end_64 = unsafe { start_ptr.add(limit_64) };

    while src < src_end_64 {
        let v = unsafe { _mm512_loadu_si512(src as *const __m512i) };
        let (lo, hi) = encode_vec!(v);

        store_128_bytes!(dst, lo, hi);

        src = unsafe { src.add(64) };
        dst = unsafe { dst.add(128) };
    }

    // --- Scalar Fallback ---
    let processed_len = unsafe { src.offset_from(start_ptr) } as usize;
    if processed_len < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[processed_len..], dst) };
    }
}

// Duplicated 64-byte LUTs and weights for AVX512
const LUT_HI: [u8; 64] = [
    0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const LUT_LO: [u8; 64] = [
    128, 192, 192, 192, 192, 192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0,
    128, 192, 192, 192, 192, 192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0,
    128, 192, 192, 192, 192, 192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0,
    128, 192, 192, 192, 192, 192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0,
];
const WEIGHTS: [u8; 64] = [
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
];

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn decode_slice_avx512(input: &[u8], mut dst: *mut u8) -> Result<(), Error> {
    let len = input.len();
    let mut src = input.as_ptr();

    // Load 64-byte LUTs into 512-bit registers
    let lut_hi = unsafe { _mm512_loadu_si512(LUT_HI.as_ptr() as *const __m512i) };
    let lut_lo = unsafe { _mm512_loadu_si512(LUT_LO.as_ptr() as *const __m512i) };
    let weights = unsafe { _mm512_loadu_si512(WEIGHTS.as_ptr() as *const __m512i) };

    let mask_0f = _mm512_set1_epi8(0x0F);
    let zero = _mm512_setzero_si512();

    macro_rules! decode_hex_vec {
        ($input:expr) => {{
            // 1. Split Nibbles
            let lo = _mm512_and_si512($input, mask_0f);
            let hi = _mm512_and_si512(_mm512_srli_epi16($input, 4), mask_0f);

            // 2. LUT Lookups (Shuffle)
            let hi_props = _mm512_shuffle_epi8(lut_hi, hi);
            let lo_props = _mm512_shuffle_epi8(lut_lo, lo);

            // 3. Validation
            // Combine properties. If valid, result != 0.
            let valid_flags = _mm512_and_si512(hi_props, lo_props);
            // Compare with zero to find errors. Returns a bitmask (1 = error).
            let err_mask = _mm512_cmpeq_epi8_mask(valid_flags, zero);

            // 4. Value Calculation
            let offset = _mm512_and_si512(hi_props, mask_0f);
            let nibbles = _mm512_add_epi8(lo, offset);

            // 5. Pack to Bytes
            // maddubs computes: High*16 + Low*1. Result is 32x i16 integers in a 512-bit register.
            let pairs_i16 = _mm512_maddubs_epi16(nibbles, weights);
            
            // Convert packed 16-bit integers to packed 8-bit integers.
            // Input: 512-bit (32 x i16). Output: 256-bit (32 x i8).
            let result_256 = _mm512_cvtepi16_epi8(pairs_i16);

            (result_256, err_mask)
        }};
    }

    // --- Large unrolled loop: 256 input bytes (128 output bytes) ---
    // We process 4 vectors of 64 bytes each.
    let limit_256 = (len / 256) * 256;
    let src_end_256 = unsafe { input.as_ptr().add(limit_256) };

    while src < src_end_256 {
        let v0 = unsafe { _mm512_loadu_si512(src as *const __m512i) };
        let v1 = unsafe { _mm512_loadu_si512(src.add(64) as *const __m512i) };
        let v2 = unsafe { _mm512_loadu_si512(src.add(128) as *const __m512i) };
        let v3 = unsafe { _mm512_loadu_si512(src.add(192) as *const __m512i) };

        let (r0, e0) = decode_hex_vec!(v0);
        let (r1, e1) = decode_hex_vec!(v1);
        let (r2, e2) = decode_hex_vec!(v2);
        let (r3, e3) = decode_hex_vec!(v3);

        // Accumulate error masks
        if (e0 | e1 | e2 | e3) != 0 {
            return Err(Error::InvalidCharacter);
        }

        unsafe {
            _mm256_storeu_si256(dst as *mut __m256i, r0);
            _mm256_storeu_si256(dst.add(32) as *mut __m256i, r1);
            _mm256_storeu_si256(dst.add(64) as *mut __m256i, r2);
            _mm256_storeu_si256(dst.add(96) as *mut __m256i, r3);

            src = src.add(256);
            dst = dst.add(128);
        }
    }

    // --- Small loop: 64 input bytes (32 output bytes) ---
    // Process remaining full AVX-512 vectors.
    let safe_len = len.saturating_sub(63);
    let src_end = unsafe { input.as_ptr().add(safe_len) };

    while src < src_end {
        let v = unsafe { _mm512_loadu_si512(src as *const __m512i) };
        let (res, err) = decode_hex_vec!(v);

        if err != 0 {
            return Err(Error::InvalidCharacter);
        }

        unsafe {
            _mm256_storeu_si256(dst as *mut __m256i, res);
            src = src.add(64);
            dst = dst.add(32);
        }
    }

    // --- Scalar fallback ---
    let processed_len = unsafe { src.offset_from(input.as_ptr()) } as usize;
    if processed_len < len {
        unsafe { scalar::decode_slice_unsafe(&input[processed_len..], dst)? };
    }

    Ok(())
}
