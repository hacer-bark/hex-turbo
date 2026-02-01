use crate::{Error, Config, scalar};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Duplicated 16-byte tables for AVX2 pshufb
const HEX_TABLE_UPPER: [u8; 32] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
];

const HEX_TABLE_LOWER: [u8; 32] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
];

// Duplicated 32-byte LUTs and weights for AVX2
const LUT_HI: [u8; 32] = [
    0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 128, 73, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const LUT_LO: [u8; 32] = [
    128, 192, 192, 192, 192, 192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0,
    128, 192, 192, 192, 192, 192, 192, 128, 128, 128, 0, 0, 0, 0, 0, 0,
];
const WEIGHTS: [u8; 32] = [
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
    16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1,
];

#[target_feature(enable = "avx2")]
pub unsafe fn encode_slice_avx2(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();
    let start_ptr = input.as_ptr();

    // Select and load the appropriate 32-byte shuffle table
    let table_ptr = if config.uppercase { HEX_TABLE_UPPER.as_ptr() } else { HEX_TABLE_LOWER.as_ptr() };
    let table = unsafe { _mm256_loadu_si256(table_ptr as *const __m256i) };

    let mask_0f = _mm256_set1_epi8(0x0F);

    macro_rules! encode_vec {
        ($in_vec:expr) => {{
            // Extract nibble indices (0-15) into low bits of each byte
            let low_idx  = _mm256_and_si256($in_vec, mask_0f);
            let high_idx = _mm256_and_si256(_mm256_srli_epi16($in_vec, 4), mask_0f);

            // LUT lookup via pshufb
            let low_chars  = _mm256_shuffle_epi8(table, low_idx);
            let high_chars = _mm256_shuffle_epi8(table, high_idx);

            // Interleave high nibble char first, then low
            let inter_lo = _mm256_unpacklo_epi8(high_chars, low_chars);
            let inter_hi = _mm256_unpackhi_epi8(high_chars, low_chars);

            (inter_lo, inter_hi)
        }};
    }

    macro_rules! store_64_bytes {
        ($dst:expr, $inter_lo:expr, $inter_hi:expr) => {{
            unsafe {
                _mm_storeu_si128($dst as *mut __m128i, _mm256_castsi256_si128($inter_lo));
                _mm_storeu_si128($dst.add(16) as *mut __m128i, _mm256_castsi256_si128($inter_hi));
                _mm_storeu_si128($dst.add(32) as *mut __m128i, _mm256_extracti128_si256($inter_lo, 1));
                _mm_storeu_si128($dst.add(48) as *mut __m128i, _mm256_extracti128_si256($inter_hi, 1));
            }
        }};
    }

    // --- Big Loop: Process 128 input bytes (256 output bytes) ---
    let limit_128 = (len / 128) * 128;
    let src_end_128 = unsafe { start_ptr.add(limit_128) };

    while src < src_end_128 {
        // Load 128 input bytes (4 vectors)
        let v0 = unsafe { _mm256_loadu_si256(src as *const __m256i) };
        let v1 = unsafe { _mm256_loadu_si256(src.add(32) as *const __m256i) };
        let v2 = unsafe { _mm256_loadu_si256(src.add(64) as *const __m256i) };
        let v3 = unsafe { _mm256_loadu_si256(src.add(96) as *const __m256i) };

        let (lo0, hi0) = encode_vec!(v0);
        let (lo1, hi1) = encode_vec!(v1);
        let (lo2, hi2) = encode_vec!(v2);
        let (lo3, hi3) = encode_vec!(v3);

        store_64_bytes!(dst, lo0, hi0);
        store_64_bytes!(dst.add(64), lo1, hi1);
        store_64_bytes!(dst.add(128), lo2, hi2);
        store_64_bytes!(dst.add(192), lo3, hi3);

        src = unsafe { src.add(128) };
        dst = unsafe { dst.add(256) };
    }

    // --- Small Loop: Process 32 input bytes (64 output bytes) ---
    let limit_32 = (len / 32) * 32;
    let src_end_32 = unsafe { start_ptr.add(limit_32) };

    while src < src_end_32 {
        let v = unsafe { _mm256_loadu_si256(src as *const __m256i) };
        let (lo, hi) = encode_vec!(v);

        store_64_bytes!(dst, lo, hi);

        src = unsafe { src.add(32) };
        dst = unsafe { dst.add(64) };
    }

    // --- Scalar Fallback ---
    let processed_len = unsafe { src.offset_from(start_ptr) } as usize;
    if processed_len < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[processed_len..], dst) };
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn decode_slice_avx2(input: &[u8], mut dst: *mut u8) -> Result<(), Error> {
    let len = input.len();
    let mut src = input.as_ptr();

    let lut_hi = unsafe { _mm256_loadu_si256(LUT_HI.as_ptr() as *const __m256i) };
    let lut_lo = unsafe { _mm256_loadu_si256(LUT_LO.as_ptr() as *const __m256i) };
    let weights = unsafe { _mm256_loadu_si256(WEIGHTS.as_ptr() as *const __m256i) };
    let mask_0f = _mm256_set1_epi8(0x0F);

    macro_rules! decode_hex_vec {
        ($input:expr) => {{
            // Split nibbles
            let lo = _mm256_and_si256($input, mask_0f);
            let hi = _mm256_and_si256(_mm256_srli_epi16($input, 4), mask_0f);

            // LUT lookups
            let hi_props = _mm256_shuffle_epi8(lut_hi, hi);
            let lo_props = _mm256_shuffle_epi8(lut_lo, lo);

            // Validation + error mask
            let valid_flags = _mm256_and_si256(hi_props, lo_props);
            let err = _mm256_cmpeq_epi8(valid_flags, _mm256_setzero_si256());

            // Value calculation
            let offset = _mm256_and_si256(hi_props, mask_0f);
            let nibbles = _mm256_add_epi8(lo, offset);

            // Pack to bytes (high * 16 + low)
            let pairs_i16 = _mm256_maddubs_epi16(nibbles, weights);
            let low = _mm256_castsi256_si128(pairs_i16);
            let high = _mm256_extracti128_si256(pairs_i16, 1);
            let result_128 = _mm_packus_epi16(low, high);

            (result_128, err)
        }};
    }

    // --- Large unrolled loop: 128 input bytes (64 output bytes) ---
    let limit_128 = (len / 128) * 128;
    let src_end_128 = unsafe { input.as_ptr().add(limit_128) };
    while src < src_end_128 {
        let v0 = unsafe { _mm256_loadu_si256(src as *const __m256i) };
        let v1 = unsafe { _mm256_loadu_si256(src.add(32) as *const __m256i) };
        let v2 = unsafe { _mm256_loadu_si256(src.add(64) as *const __m256i) };
        let v3 = unsafe { _mm256_loadu_si256(src.add(96) as *const __m256i) };

        let (r0, e0) = decode_hex_vec!(v0);
        let (r1, e1) = decode_hex_vec!(v1);
        let (r2, e2) = decode_hex_vec!(v2);
        let (r3, e3) = decode_hex_vec!(v3);

        let err_any = _mm256_or_si256(
            _mm256_or_si256(e0, e1),
            _mm256_or_si256(e2, e3),
        );

        if _mm256_movemask_epi8(err_any) != 0 {
            return Err(Error::InvalidCharacter);
        }

        unsafe {
            _mm_storeu_si128(dst as *mut __m128i, r0);
            _mm_storeu_si128(dst.add(16) as *mut __m128i, r1);
            _mm_storeu_si128(dst.add(32) as *mut __m128i, r2);
            _mm_storeu_si128(dst.add(48) as *mut __m128i, r3);
            src = src.add(128);
            dst = dst.add(64);
        }
    }

    // --- Small loop: 32 input bytes (16 output bytes) ---
    let safe_len = len.saturating_sub(31);
    let src_end = unsafe { input.as_ptr().add(safe_len) };
    while src < src_end {
        let v = unsafe { _mm256_loadu_si256(src as *const __m256i) };
        let (res, err) = decode_hex_vec!(v);

        if _mm256_movemask_epi8(err) != 0 {
            return Err(Error::InvalidCharacter);
        }

        unsafe {
            _mm_storeu_si128(dst as *mut __m128i, res);
            src = src.add(32);
            dst = dst.add(16);
        }
    }

    // --- Scalar fallback ---
    let processed_len = unsafe { src.offset_from(input.as_ptr()) } as usize;
    if processed_len < len {
        unsafe { scalar::decode_slice_unsafe(&input[processed_len..], dst)? };
    }

    Ok(())
}

#[cfg(kani)]
mod kani_verification_avx2 {
    use super::*;
    use crate::Config;
    use core::mem::transmute;

    // Magic number: 32
    const INPUT_LEN: usize = 32;

    // --- HELPERS AND STUBS ---

    // STUB: _mm256_shuffle_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shuffle_epi8
    #[allow(dead_code)]
    unsafe fn mm256_shuffle_epi8_stub(a: __m256i, b: __m256i) -> __m256i {
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
    #[allow(dead_code)]
    unsafe fn mm256_maddubs_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [i8; 32] = unsafe { transmute(b) };
        let mut dst = [0i16; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*16
            let i = j * 2;

            // dst[i+15:i] := Saturate16( a[i+15:i+8]*b[i+15:i+8] + a[i+7:i]*b[i+7:i] )
            dst[j] = ((a[i+1] as i16) * (b[i+1] as i16)).saturating_add((a[i] as i16) * (b[i] as i16));
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm_packus_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_packus_epi16
    #[allow(dead_code)]
    unsafe fn mm_packus_epi16_stub(a: __m128i, b: __m128i) -> __m128i {
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

    // -- REAL LOGIC --- 

    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_maddubs_epi16, mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm_packus_epi16, mm_packus_epi16_stub)]
    fn check_roundtrip_safety() {
        // Symbolic Config
        let config = Config { uppercase: kani::any() };

        // Symbolic Input
        let input: [u8; INPUT_LEN] = kani::any();

        // Setup Buffers
        let mut enc_buf = [0u8; INPUT_LEN * 2];
        let mut dec_buf = [0u8; INPUT_LEN];

        unsafe {
            // Encode
            encode_slice_avx2(&config, &input, enc_buf.as_mut_ptr());

            // Decode
            decode_slice_avx2(&enc_buf, dec_buf.as_mut_ptr()).expect("Decoder failed");

            // Verification
            assert_eq!(&dec_buf, &input, "AVX2 Roundtrip Failed");
        }
    }

    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_maddubs_epi16, mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm_packus_epi16, mm_packus_epi16_stub)]
    fn check_decoder_robustness() {
        // Symbolic Input (Random Garbage)
        let input: [u8; INPUT_LEN] = kani::any();

        // Setup Buffer
        let mut dec_buf = [0u8; 64];

        unsafe {
            // We verify what function NEVER panics/crashes
            let _ = decode_slice_avx2(&input, dec_buf.as_mut_ptr());
        }
    }
}

#[cfg(all(test, miri))]
mod avx2_miri_tests {
    use super::{encode_slice_avx2, decode_slice_avx2};
    use crate::{Config, Error};

    // Reference crate (hex crate: lowercase encode, case-insensitive decode)
    use hex::{encode as ref_encode_lower, encode_upper as ref_encode_upper};

    use rand::{rng, Rng};

    // --- Helpers ---
    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    fn verify_roundtrip(config: &Config, input: &[u8], ref_encode: fn(Vec<u8>) -> String) {
        let len = input.len();

        // --- Encoding ---
        let expected = ref_encode(input.to_vec());

        let mut enc_buf = vec![0u8; len * 2];
        unsafe {
            encode_slice_avx2(config, input, enc_buf.as_mut_ptr());
        }

        assert_eq!(&enc_buf[..], expected.as_bytes(), "AVX2 Encoding mismatch (len={})", len);

        // --- Decoding (own output) ---
        let mut dec_buf = vec![0u8; len];
        unsafe { decode_slice_avx2(&enc_buf, dec_buf.as_mut_ptr()).expect("AVX2 decoder failed on valid own output") };

        assert_eq!(&dec_buf[..], input, "AVX2 round-trip failed (len={})", len);
    }

    fn run_avx2_tests(uppercase: bool) {
        let config = Config { uppercase };
        let ref_encode_fn: fn(Vec<u8>) -> String = if uppercase { ref_encode_upper } else { ref_encode_lower };

        // --- Deterministic boundary tests (hit all loop transitions) ---
        for len in 0..64 {
            let input = random_bytes(len);
            verify_roundtrip(&config, &input, ref_encode_fn);
        }

        // --- Small random fuzz ---
        let mut rng = rng();
        for _ in 0..15 {
            let len = rng.random_range(65..=512);
            let input = random_bytes(len);
            verify_roundtrip(&config, &input, ref_encode_fn);
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
        let mut rng = rng();
        for _ in 0..5 {
            let original_len = rng.random_range(1..=256);
            let input = random_bytes(original_len);

            // Start from valid lowercase encoding
            let mut hex_chars = ref_encode_lower(&input).into_bytes();

            // Randomly convert some chars to uppercase
            for b in &mut hex_chars {
                if rng.random_bool(0.5) {
                    *b = b.to_ascii_uppercase();
                }
            }

            let mut dec_buf = vec![0u8; original_len];
            unsafe { decode_slice_avx2(&hex_chars, dec_buf.as_mut_ptr()).expect("AVX2 decoder failed on valid mixed-case input") };

            assert_eq!(&dec_buf[..], input);
        }
    }

    #[test]
    fn miri_avx2_decode_errors() {
        let mut out = vec![0u8; 128];

        // Invalid character in middle region
        let mut invalid_simd = vec![b'0'; 64];
        invalid_simd[15] = b'g';
        let res = unsafe { decode_slice_avx2(&invalid_simd, out.as_mut_ptr()) };
        assert!(matches!(res, Err(Error::InvalidCharacter)));

        // Force small length with invalid char near end
        let invalid_tail = b"00112233445566778899abcg";
        let res = unsafe { decode_slice_avx2(invalid_tail, out.as_mut_ptr()) };
        assert!(matches!(res, Err(Error::InvalidCharacter)));
    }
}
