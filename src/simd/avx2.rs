use crate::{Config, Error, scalar};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// --- CONSTANTS ---

// Duplicated 16-byte tables for AVX2 pshufb (Encoding)
const HEX_TABLE_UPPER: [u8; 32] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
];

const HEX_TABLE_LOWER: [u8; 32] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
];

// Duplicated 16-byte LUTs and weights for AVX2 (Decoding)
const LUT_HI: [u8; 32] = [
    0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0,
    0, 0, 0,
];
const LUT_LO: [u8; 32] = [
    128, 192, 192, 192, 192, 192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0, 128, 192, 192, 192, 192,
    192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0,
];
const WEIGHTS: [u8; 32] = [
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16,
    1, 16, 1, 16, 1,
];

// --- ENCODING ---

#[target_feature(enable = "avx2")]
pub unsafe fn encode_slice_avx2(config: &Config, input: &[u8], mut dst: *mut u8) {
    let table_ptr = if config.uppercase {
        HEX_TABLE_UPPER.as_ptr()
    } else {
        HEX_TABLE_LOWER.as_ptr()
    };
    let table = unsafe { _mm256_loadu_si256(table_ptr as *const __m256i) };
    let mask_0f = _mm256_set1_epi8(0x0F);

    macro_rules! encode_chunk {
        ($in_vec:expr, $dst_ptr:expr) => {{
            let low_idx = _mm256_and_si256($in_vec, mask_0f);
            let high_idx = _mm256_and_si256(_mm256_srli_epi16($in_vec, 4), mask_0f);

            let low_chars = _mm256_shuffle_epi8(table, low_idx);
            let high_chars = _mm256_shuffle_epi8(table, high_idx);

            let inter_lo = _mm256_unpacklo_epi8(high_chars, low_chars);
            let inter_hi = _mm256_unpackhi_epi8(high_chars, low_chars);

            let out0 = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x20);
            let out1 = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x31);

            unsafe { _mm256_storeu_si256($dst_ptr as *mut __m256i, out0) };
            unsafe { _mm256_storeu_si256($dst_ptr.add(32) as *mut __m256i, out1) };
        }};
    }

    let mut src = input;

    while src.len() >= 128 {
        let v0 = unsafe { _mm256_loadu_si256(src.as_ptr() as *const __m256i) };
        let v1 = unsafe { _mm256_loadu_si256(src.as_ptr().add(32) as *const __m256i) };
        let v2 = unsafe { _mm256_loadu_si256(src.as_ptr().add(64) as *const __m256i) };
        let v3 = unsafe { _mm256_loadu_si256(src.as_ptr().add(96) as *const __m256i) };

        encode_chunk!(v0, dst);
        encode_chunk!(v1, dst.add(64));
        encode_chunk!(v2, dst.add(128));
        encode_chunk!(v3, dst.add(192));

        src = &src[128..];
        dst = unsafe { dst.add(256) };
    }

    while src.len() >= 32 {
        let v = unsafe { _mm256_loadu_si256(src.as_ptr() as *const __m256i) };
        encode_chunk!(v, dst);

        src = &src[32..];
        dst = unsafe { dst.add(64) };
    }

    if !src.is_empty() {
        unsafe { scalar::encode_slice_unsafe(config, src, dst) };
    }
}

// --- DECODING ---

#[target_feature(enable = "avx2")]
pub unsafe fn decode_slice_avx2(input: &[u8], mut dst: *mut u8) -> Result<(), Error> {
    let lut_hi = unsafe { _mm256_loadu_si256(LUT_HI.as_ptr() as *const __m256i) };
    let lut_lo = unsafe { _mm256_loadu_si256(LUT_LO.as_ptr() as *const __m256i) };
    let weights = unsafe { _mm256_loadu_si256(WEIGHTS.as_ptr() as *const __m256i) };

    let mask_0f = _mm256_set1_epi8(0x0F);
    let zero = _mm256_setzero_si256();

    macro_rules! decode_chunk {
        ($input:expr) => {{
            let lo = _mm256_and_si256($input, mask_0f);
            let hi = _mm256_and_si256(_mm256_srli_epi16($input, 4), mask_0f);

            let hi_props = _mm256_shuffle_epi8(lut_hi, hi);
            let lo_props = _mm256_shuffle_epi8(lut_lo, lo);

            let valid = _mm256_and_si256(hi_props, lo_props);

            let offset = _mm256_and_si256(hi_props, mask_0f);
            let nibbles = _mm256_add_epi8(lo, offset);

            let pairs = _mm256_maddubs_epi16(nibbles, weights);

            (pairs, valid)
        }};
    }

    let mut src = input;

    while src.len() >= 128 {
        let v0 = unsafe { _mm256_loadu_si256(src.as_ptr() as *const __m256i) };
        let v1 = unsafe { _mm256_loadu_si256(src.as_ptr().add(32) as *const __m256i) };
        let v2 = unsafe { _mm256_loadu_si256(src.as_ptr().add(64) as *const __m256i) };
        let v3 = unsafe { _mm256_loadu_si256(src.as_ptr().add(96) as *const __m256i) };

        let (r0, v0_val) = decode_chunk!(v0);
        let (r1, v1_val) = decode_chunk!(v1);
        let (r2, v2_val) = decode_chunk!(v2);
        let (r3, v3_val) = decode_chunk!(v3);

        let v01 = _mm256_min_epu8(v0_val, v1_val);
        let v23 = _mm256_min_epu8(v2_val, v3_val);
        let valid_all = _mm256_min_epu8(v01, v23);

        let err = _mm256_cmpeq_epi8(valid_all, zero);
        if _mm256_movemask_epi8(err) != 0 {
            return Err(Error::InvalidCharacter);
        }

        let packed01 = _mm256_packus_epi16(r0, r1);
        let ordered01 = _mm256_permute4x64_epi64(packed01, 0xD8);
        unsafe { _mm256_storeu_si256(dst as *mut __m256i, ordered01) };

        let packed23 = _mm256_packus_epi16(r2, r3);
        let ordered23 = _mm256_permute4x64_epi64(packed23, 0xD8);
        unsafe { _mm256_storeu_si256(dst.add(32) as *mut __m256i, ordered23) };

        src = &src[128..];
        dst = unsafe { dst.add(64) };
    }

    while src.len() >= 32 {
        let v = unsafe { _mm256_loadu_si256(src.as_ptr() as *const __m256i) };
        let (pairs, valid) = decode_chunk!(v);

        let err = _mm256_cmpeq_epi8(valid, zero);
        if _mm256_movemask_epi8(err) != 0 {
            return Err(Error::InvalidCharacter);
        }

        let low = _mm256_castsi256_si128(pairs);
        let high = _mm256_extracti128_si256(pairs, 1);
        let res = _mm_packus_epi16(low, high);

        unsafe { _mm_storeu_si128(dst as *mut __m128i, res) };

        src = &src[32..];
        dst = unsafe { dst.add(16) };
    }

    if !src.is_empty() {
        (unsafe { scalar::decode_slice_unsafe(src, dst) })?;
    }

    Ok(())
}

// --- KANI (FORMAL VERIFICATION) ---

#[cfg(kani)]
mod kani_verification_avx2 {
    use super::*;
    use crate::Config;
    use core::mem::transmute;

    // --- CONSTANTS ---

    // Encoder Induction Size: 32 (1 AVX2 Loop) + 1 (Scalar Transition)
    const ENC_INDUCTION_LEN: usize = 33;

    // Decoder Induction Size: 32 (1 AVX2 Loop) + 1 (Scalar Transition)
    const DEC_INDUCTION_LEN: usize = 33;

    // --- STUBS ---

    // STUB: _mm256_shuffle_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shuffle_epi8
    unsafe fn _mm256_shuffle_epi8_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [u8; 32] = unsafe { transmute(b) };
        let mut dst = [0u8; 32];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*8
            // (In Rust we access bytes 'j' so '*8' offset is not needed)
            let i = j;

            // IF b[i+7] == 1
            if (b[i] & 0x80) != 0 {
                // dst[i+7:i] := 0
                dst[i] = 0;
            } else {
                // index[3:0] := b[i+3:i]
                let index = b[i] & 0x0F;
                // dst[i+7:i] := a[index*8+7:index*8]
                dst[i] = a[index as usize];
            }
            // FI

            // IF b[128+i+7] == 1
            if (b[16 + i] & 0x80) != 0 {
                // dst[128+i+7:128+i] := 0
                dst[16 + i] = 0;
            } else {
                // index[3:0] := b[128+i+3:128+i]
                let index = b[16 + i] & 0x0F;
                // dst[128+i+7:128+i] := a[128+index*8+7:128+index*8]
                dst[16 + i] = a[(16 + index) as usize];
            }
            // FI
        }
        // ENDFOR

        // dst[MAX:256] := 0
        // (__m256i is exactly 256 bits. There are no bits beyond 256 to zero out)

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_maddubs_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maddubs_epi16
    unsafe fn _mm256_maddubs_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [i8; 32] = unsafe { transmute(b) };
        let mut dst = [0i16; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*16
            let i = j * 2;

            // dst[i+15:i] := Saturate16( a[i+15:i+8]*b[i+15:i+8] + a[i+7:i]*b[i+7:i] )
            dst[j] = ((a[i + 1] as i16) * (b[i + 1] as i16))
                .saturating_add((a[i] as i16) * (b[i] as i16));
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_testz_si256
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_testz_si256
    // Note: in this logic added complexity as Rust do not support 256 bits values.
    unsafe fn _mm256_testz_si256_stub(a: __m256i, b: __m256i) -> i32 {
        let a: [u64; 4] = unsafe { transmute(a) };
        let b: [u64; 4] = unsafe { transmute(b) };
        let zf: i32;
        let _cf: i32;

        // Perform 256 bit AND
        let res_and = [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]];

        // IF ((a[255:0] AND b[255:0]) == 0)
        if res_and[0] == 0 && res_and[1] == 0 && res_and[2] == 0 && res_and[3] == 0 {
            // ZF := 1
            zf = 1;
        } else {
            // ZF := 0
            zf = 0;
        }
        // FI

        // Perform 256 bit (NOT a) AND b
        let res_not_and = [
            (!a[0]) & b[0],
            (!a[1]) & b[1],
            (!a[2]) & b[2],
            (!a[3]) & b[3],
        ];

        // IF (((NOT a[255:0]) AND b[255:0]) == 0)
        if res_not_and[0] == 0 && res_not_and[1] == 0 && res_not_and[2] == 0 && res_not_and[3] == 0
        {
            // CF := 1
            _cf = 1;
        } else {
            // CF := 0
            _cf = 0;
        }
        // FI

        // RETURN ZF
        return zf;
    }

    // STUB: _mm_packus_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_packus_epi16
    unsafe fn _mm_packus_epi16_stub(a: __m128i, b: __m128i) -> __m128i {
        let a: [i16; 8] = unsafe { transmute(a) };
        let b: [i16; 8] = unsafe { transmute(b) };
        let mut dst = [0u8; 16];

        // dst[7:0] := SaturateU8(a[15:0])
        dst[0] = a[0].max(0).min(255) as u8;
        // dst[15:8] := SaturateU8(a[31:16])
        dst[1] = a[1].max(0).min(255) as u8;
        // dst[23:16] := SaturateU8(a[47:32])
        dst[2] = a[2].max(0).min(255) as u8;
        // dst[31:24] := SaturateU8(a[63:48])
        dst[3] = a[3].max(0).min(255) as u8;
        // dst[39:32] := SaturateU8(a[79:64])
        dst[4] = a[4].max(0).min(255) as u8;
        // dst[47:40] := SaturateU8(a[95:80])
        dst[5] = a[5].max(0).min(255) as u8;
        // dst[55:48] := SaturateU8(a[111:96])
        dst[6] = a[6].max(0).min(255) as u8;
        // dst[63:56] := SaturateU8(a[127:112])
        dst[7] = a[7].max(0).min(255) as u8;
        // dst[71:64] := SaturateU8(b[15:0])
        dst[8] = b[0].max(0).min(255) as u8;
        // dst[79:72] := SaturateU8(b[31:16])
        dst[9] = b[1].max(0).min(255) as u8;
        // dst[87:80] := SaturateU8(b[47:32])
        dst[10] = b[2].max(0).min(255) as u8;
        // dst[95:88] := SaturateU8(b[63:48])
        dst[11] = b[3].max(0).min(255) as u8;
        // dst[103:96] := SaturateU8(b[79:64])
        dst[12] = b[4].max(0).min(255) as u8;
        // dst[111:104] := SaturateU8(b[95:80])
        dst[13] = b[5].max(0).min(255) as u8;
        // dst[119:112] := SaturateU8(b[111:96])
        dst[14] = b[6].max(0).min(255) as u8;
        // dst[127:120] := SaturateU8(b[127:112])
        dst[15] = b[7].max(0).min(255) as u8;

        unsafe { transmute(dst) }
    }

    // --- REAL TESTS ---

    /// **Proof 1: Roundtrip Correctness (The Logic Check)**
    ///
    /// Verifies that `Decode(Encode(X)) == X`.
    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, _mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_maddubs_epi16, _mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_testz_si256, _mm256_testz_si256_stub)]
    #[kani::stub(_mm_packus_epi16, _mm_packus_epi16_stub)]
    fn check_avx2_roundtrip_correctness() {
        let config = Config {
            uppercase: kani::any(),
        };
        let input: [u8; ENC_INDUCTION_LEN] = kani::any();
        let input_len = input.len();

        // Buffers
        let mut enc_buf = [0u8; 128];
        let mut dec_buf = [0u8; 128];

        unsafe {
            // 1. Encode
            encode_slice_avx2(&config, &input, enc_buf.as_mut_ptr());

            // Calculate actual encoded length for slicing
            let encoded_slice = &enc_buf[..input_len * 2];

            // 2. Decode
            // This MUST succeed for valid encoded output
            decode_slice_avx2(encoded_slice, dec_buf.as_mut_ptr())
                .expect("Valid encoding failed to decode");

            // 3. Verify
            assert_eq!(&dec_buf[..input_len], &input, "Roundtrip mismatch");
        }
    }

    /// **Proof 2: Decoder Robustness & Induction**
    ///
    /// Verifies that `decode_slice_avx2`:
    /// 1. Accepts ANY `N` bytes of garbage input.
    /// 2. Never Segfaults, Panics, or causes UB.
    /// 3. Safely handles the SIMD->Scalar pointer transition.
    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, _mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_maddubs_epi16, _mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_testz_si256, _mm256_testz_si256_stub)]
    #[kani::stub(_mm_packus_epi16, _mm_packus_epi16_stub)]
    fn check_avx2_decode_robustness() {
        // Input: `N` bytes of unrestricted symbolic data (garbage)
        let input: [u8; DEC_INDUCTION_LEN] = kani::any();

        // Output Buffer: Max estimated size
        let mut output = [0u8; 128];

        unsafe {
            // We ignore the Result. We only care that this function call
            // returns safely (Ok or Err) and does not crash.
            let _ = decode_slice_avx2(&input, output.as_mut_ptr());
        }
    }
}

// --- MIRI (FORMAL VERIFICATION) ---

#[cfg(all(test, miri))]
mod avx2_miri_tests {
    use super::{decode_slice_avx2, encode_slice_avx2};
    use crate::{Config, Error};

    // Reference crate
    use hex::encode as ref_encode_lower;

    // --- Fast Deterministic Generator ---
    // Generating random numbers in Miri is extremely slow.
    // Sequential bytes cover 100% of the bitwise logic just as effectively.
    fn get_data(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 256) as u8).collect()
    }

    // --- Helpers ---
    fn verify_roundtrip(config: &Config, input: &[u8]) {
        let len = input.len();

        // --- Encoding ---
        let expected_lower = ref_encode_lower(input);
        let expected = if config.uppercase {
            expected_lower.to_ascii_uppercase()
        } else {
            expected_lower
        };

        let mut enc_buf = vec![0u8; len * 2];
        unsafe {
            encode_slice_avx2(config, input, enc_buf.as_mut_ptr());
        }

        assert_eq!(
            &enc_buf[..],
            expected.as_bytes(),
            "AVX2 Encoding mismatch (len={})",
            len
        );

        // --- Decoding (own output) ---
        let mut dec_buf = vec![0u8; len];
        unsafe {
            decode_slice_avx2(&enc_buf, dec_buf.as_mut_ptr())
                .expect("AVX2 decoder failed on valid own output")
        };

        assert_eq!(&dec_buf[..], input, "AVX2 round-trip failed (len={})", len);
    }

    fn run_avx2_tests(uppercase: bool) {
        let config = Config { uppercase };

        // MIRI is slow. We don't need random lengths.
        // We only need to test boundary conditions to achieve 100% path coverage.
        // 0..16: Scalar tails
        // 32: AVX2 boundaries (AVX2 processes 32 bytes / 64 hex chars per chunk)
        // 64: Multiple AVX2 chunks
        let boundaries = [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128];

        for &len in &boundaries {
            let input = get_data(len);
            verify_roundtrip(&config, &input);
        }
    }

    // --- Tests ---

    #[test]
    fn miri_avx2_lower_roundtrip() {
        run_avx2_tests(false);
    }

    #[test]
    fn miri_avx2_upper_roundtrip() {
        run_avx2_tests(true);
    }

    #[test]
    fn miri_avx2_decode_mixed_case() {
        // 64 length ensures we trigger exactly 2 full AVX2 loops (64 bytes = 128 hex chars)
        let input = get_data(64);
        let hex_lower = ref_encode_lower(&input).into_bytes();

        // Deterministically mix case (avoids heavy `rand` in Miri)
        let mixed_hex: Vec<u8> = hex_lower
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                if i % 2 == 0 {
                    b.to_ascii_uppercase()
                } else {
                    b
                }
            })
            .collect();

        let mut dec_buf = vec![0u8; 64];
        unsafe {
            decode_slice_avx2(&mixed_hex, dec_buf.as_mut_ptr())
                .expect("AVX2 decoder failed on valid mixed-case input")
        };

        assert_eq!(&dec_buf[..], input);
    }

    #[test]
    fn miri_avx2_decode_errors() {
        let mut out = [0u8; 128];

        // 1. Invalid character in SIMD region (64 hex chars = 32 bytes = 1 AVX2 chunk)
        let mut invalid_simd = vec![b'0'; 64];
        invalid_simd[33] = b'g'; // 'g' is strictly invalid in hex
        let res = unsafe { decode_slice_avx2(&invalid_simd, out.as_mut_ptr()) };
        assert_eq!(res, Err(Error::InvalidCharacter));

        // 2. Invalid character in Scalar/Tail region (66 chars = 64 SIMD + 2 scalar)
        let mut invalid_tail = vec![b'0'; 66];
        invalid_tail[65] = b'g';
        let res = unsafe { decode_slice_avx2(&invalid_tail, out.as_mut_ptr()) };
        assert_eq!(res, Err(Error::InvalidCharacter));
    }
}
