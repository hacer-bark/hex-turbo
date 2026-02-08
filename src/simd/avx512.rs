use crate::{Error, Config, scalar};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// --- CONSTANTS ---

// Duplicated 16-byte tables for AVX512 pshufb (Encoding)
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

// Duplicated 16-byte LUTs and weights for AVX512 (Decoding)
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

// --- MIRI STUBS ---

// STUB: _mm512_permutex2var_epi64
// REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_permutex2var_epi64
#[cfg(all(miri, test))]
fn mm512_permutex2var_epi64(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    use core::mem::transmute;

    let a: [u64; 8] = unsafe { transmute(a) };
    let idx: [u64; 8] = unsafe { transmute(idx) };
    let b: [u64; 8] = unsafe { transmute(b) };
    let mut dst = [0u64; 8];

    // FOR j := 0 to 7
    for j in 0..8 {
        // i := j*64
        let i = j;
        // off := idx[i+2:i]*64
        let off = (idx[i] & 0x7) as usize;
        // dst[i+63:i] := idx[i+3] ? b[off+63:off] : a[off+63:off]
        dst[i] = if (idx[i] >> 3) & 1 != 0 { b[off] } else { a[off] };
    // ENDFOR
    }
    // dst[MAX:512] := 0

    unsafe { transmute(dst) }
}

#[cfg(not(all(miri, test)))]
#[target_feature(enable = "avx512f,avx512bw")]
fn mm512_permutex2var_epi64(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    _mm512_permutex2var_epi64(a, idx, b)
}

// --- ENCODING ---

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn encode_slice_avx512(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();
    let start_ptr = input.as_ptr();

    // Select and load the appropriate 32-byte shuffle table
    let table_ptr = if config.uppercase { HEX_TABLE_UPPER.as_ptr() } else { HEX_TABLE_LOWER.as_ptr() };
    let table = unsafe { _mm512_loadu_si512(table_ptr as *const __m512i) };
    let mask_0f = _mm512_set1_epi8(0x0F);

    // Permutation Indices
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
            let ordered_1 = mm512_permutex2var_epi64($inter_lo, perm_idx_1, $inter_hi);
            let ordered_2 = mm512_permutex2var_epi64($inter_lo, perm_idx_2, $inter_hi);

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

// --- DECODING ---

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

// --- KANI (FORMAL VERIFICATION) ---

#[cfg(kani)]
mod kani_verification_avx512 {
    use super::*;
    use crate::Config;
    use core::mem::transmute;

    const INPUT_LEN: usize = 65;

    // --- KANI STUBS ---

    // STUB: _mm512_shuffle_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shuffle_epi8
    #[allow(dead_code)]
    unsafe fn mm512_shuffle_epi8_stub(a: __m512i, b: __m512i) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [u8; 64] = unsafe { transmute(b) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // i := j*8
            // (In Rust we access bytes 'j' so '*8' offset is not needed)
            let i = j;

            // IF b[i+7] == 1
            if (b[i] & 0x80) != 0 {
                // dst[i+7:i] := 0
                dst[i] = 0;
            // ELSE
            } else {
                // index[5:0] := b[i+3:i] + (j & 0x30)
                let index: u8 = (b[i] & 0x0F) + (j as u8 & 0x30);
                // dst[i+7:i] := a[index*8+7:index*8]
                dst[i] = a[index as usize];
            // FI
            }
        // ENDFOR
        }
        // dst[MAX:512] := 0
        // (No extra bits beyond 512 in __m512i)

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_permutex2var_epi64
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_permutex2var_epi64
    #[allow(dead_code)]
    unsafe fn mm512_permutex2var_epi64_stub(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
        let a: [u64; 8] = unsafe { transmute(a) };
        let idx: [u64; 8] = unsafe { transmute(idx) };
        let b: [u64; 8] = unsafe { transmute(b) };
        let mut dst = [0u64; 8];

        // FOR j := 0 to 7
        for j in 0..8 {
            // i := j*64
            let i = j;
            // off := idx[i+2:i]*64
            let off = (idx[i] & 0x7) as usize;
            // dst[i+63:i] := idx[i+3] ? b[off+63:off] : a[off+63:off]
            dst[i] = if (idx[i] >> 3) & 1 != 0 { b[off] } else { a[off] };
        // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_maddubs_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maddubs_epi16
    #[allow(dead_code)]
    unsafe fn mm512_maddubs_epi16_stub(a: __m512i, b: __m512i) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [i8; 64] = unsafe { transmute(b) };
        let mut dst = [0i16; 32];

        // FOR j := 0 to 31
        for j in 0..32 {
            // i := j*16
            let i = j * 2;
            // dst[i+15:i] := Saturate16( a[i+15:i+8]*b[i+15:i+8] + a[i+7:i]*b[i+7:i] )
            dst[j] = ((a[i+1] as i16) * (b[i+1] as i16)).saturating_add((a[i] as i16) * (b[i] as i16));
        // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_cvtepi16_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepi16_epi8
    #[allow(dead_code)]
    unsafe fn mm512_cvtepi16_epi8_stub(a: __m512i) -> __m256i {
        let a: [u16; 32] = unsafe { transmute(a) };
        let mut dst = [0u8; 32];

        // FOR j := 0 to 31
        for j in 0..32 {
            // i := 16*j
            let i = j;
            // l := 8*j
            let l = j;
            // dst[l+7:l] := Truncate8(a[i+15:i])
            dst[l] = a[i] as u8;
        // ENDFOR
        }
        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // --- REAL TESTS --- 

    #[kani::proof]
    #[kani::stub(_mm512_shuffle_epi8, mm512_shuffle_epi8_stub)]
    #[kani::stub(_mm512_permutex2var_epi64, mm512_permutex2var_epi64_stub)]
    #[kani::stub(_mm512_maddubs_epi16, mm512_maddubs_epi16_stub)]
    #[kani::stub(_mm512_cvtepi16_epi8, mm512_cvtepi16_epi8_stub)]
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
            encode_slice_avx512(&config, &input, enc_buf.as_mut_ptr());

            // Decode
            decode_slice_avx512(&enc_buf, dec_buf.as_mut_ptr()).expect("Decoder failed");

            // Verification
            assert_eq!(&dec_buf, &input, "AVX512 Roundtrip Failed");
        }
    }

    #[kani::proof]
    #[kani::stub(_mm512_shuffle_epi8, mm512_shuffle_epi8_stub)]
    #[kani::stub(_mm512_permutex2var_epi64, mm512_permutex2var_epi64_stub)]
    #[kani::stub(_mm512_maddubs_epi16, mm512_maddubs_epi16_stub)]
    #[kani::stub(_mm512_cvtepi16_epi8, mm512_cvtepi16_epi8_stub)]
    fn check_decoder_robustness() {
        // Symbolic Input (Random Garbage)
        let input: [u8; INPUT_LEN] = kani::any();

        // Setup Buffer
        let mut dec_buf = [0u8; INPUT_LEN * 2];

        unsafe {
            // We verify what function NEVER panics/crashes
            let _ = decode_slice_avx512(&input, dec_buf.as_mut_ptr());
        }
    }
}

// --- MIRI (FORMAL VERIFICATION) ---

#[cfg(all(test, miri))]
mod avx512_miri_tests {
    use super::{encode_slice_avx512, decode_slice_avx512};
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
        unsafe { encode_slice_avx512(config, input, enc_buf.as_mut_ptr()); }

        assert_eq!(&enc_buf[..], expected.as_bytes(), "AVX2 Encoding mismatch (len={})", len);

        // --- Decoding (own output) ---
        let mut dec_buf = vec![0u8; len];
        unsafe { decode_slice_avx512(&enc_buf, dec_buf.as_mut_ptr()).expect("AVX2 decoder failed on valid own output") };

        assert_eq!(&dec_buf[..], input, "AVX2 round-trip failed (len={})", len);
    }

    fn run_avx512_tests(uppercase: bool) {
        let config = Config { uppercase };
        let ref_encode_fn: fn(Vec<u8>) -> String = if uppercase { ref_encode_upper } else { ref_encode_lower };

        // --- Deterministic boundary tests (hit all loop transitions) ---
        for len in 64..128 {
            let input = random_bytes(len);
            verify_roundtrip(&config, &input, ref_encode_fn);
        }

        // --- Small random fuzz ---
        let mut rng = rng();
        for _ in 0..15 {
            let len = rng.random_range(0..=512);
            let input = random_bytes(len);
            verify_roundtrip(&config, &input, ref_encode_fn);
        }
    }

    // --- Tests ---

    #[test]
    fn miri_avx512_lower_roundtrip() {
        run_avx512_tests(false);
    }

    #[test]
    fn miri_avx512_upper_roundtrip() {
        run_avx512_tests(true);
    }

    #[test]
    fn miri_avx512_decode_mixed_case() {
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
            unsafe { decode_slice_avx512(&hex_chars, dec_buf.as_mut_ptr()).expect("avx512 decoder failed on valid mixed-case input") };

            assert_eq!(&dec_buf[..], input);
        }
    }

    #[test]
    fn miri_avx512_decode_errors() {
        let mut out = vec![0u8; 128];

        // Invalid character in middle region
        let mut invalid_simd = vec![b'0'; 64];
        invalid_simd[15] = b'g';
        let res = unsafe { decode_slice_avx512(&invalid_simd, out.as_mut_ptr()) };
        assert!(matches!(res, Err(Error::InvalidCharacter)));

        // Force small length with invalid char near end
        let invalid_tail = b"00112233445566778899abcg";
        let res = unsafe { decode_slice_avx512(invalid_tail, out.as_mut_ptr()) };
        assert!(matches!(res, Err(Error::InvalidCharacter)));
    }
}
