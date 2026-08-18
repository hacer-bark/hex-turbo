//! AVX2 verification: Kani proofs, Intel-pseudocode intrinsic stubs, and the
//! Miri coverage suite. Split out of the production module purely to keep it
//! lean; nothing here is compiled into a normal build.

// Re-exported to the `kani`/`miri` submodules below; neither is compiled in a
// plain `cargo test`, so the glob looks unused there.
#[allow(unused_imports)]
use super::*;

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
            encode_slice_avx2(&config, &input, &mut enc_buf);

            // Calculate actual encoded length for slicing
            let encoded_slice = &enc_buf[..input_len * 2];

            // 2. Decode
            // This MUST succeed for valid encoded output
            decode_slice_avx2(encoded_slice, &mut dec_buf)
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
            let _ = decode_slice_avx2(&input, &mut output);
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
            encode_slice_avx2(config, input, &mut enc_buf);
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
            decode_slice_avx2(&enc_buf, &mut dec_buf)
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
            decode_slice_avx2(&mixed_hex, &mut dec_buf)
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
        let res = unsafe { decode_slice_avx2(&invalid_simd, &mut out) };
        assert_eq!(res, Err(Error::InvalidCharacter));

        // 2. Invalid character in Scalar/Tail region (66 chars = 64 SIMD + 2 scalar)
        let mut invalid_tail = vec![b'0'; 66];
        invalid_tail[65] = b'g';
        let res = unsafe { decode_slice_avx2(&invalid_tail, &mut out) };
        assert_eq!(res, Err(Error::InvalidCharacter));
    }
}
