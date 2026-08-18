//! AVX-512 verification: Kani proofs, Intel-pseudocode intrinsic stubs, and the
//! Miri coverage suite. Split out of the production module purely to keep it
//! lean; nothing here is compiled into a normal build.

// Re-exported to the `kani`/`miri` submodules below; neither is compiled in a
// plain `cargo test`, so the glob looks unused there.
#[allow(unused_imports)]
use super::*;

// --- KANI (FORMAL VERIFICATION) ---

#[cfg(kani)]
mod kani_verification_avx512 {
    use super::*;
    use crate::Config;
    use core::mem::transmute;

    // --- CONSTANTS ---

    // Encoder Induction Size: 64 (1 AVX512 Loop) + 1 (Scalar Transition)
    const ENC_INDUCTION_LEN: usize = 65;

    // Decoder Induction Size: 64 (1 AVX512 Loop) + 1 (Scalar Transition)
    const DEC_INDUCTION_LEN: usize = 65;

    // --- STUBS ---

    // STUB: _mm512_shuffle_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shuffle_epi8
    unsafe fn _mm512_shuffle_epi8_stub(a: __m512i, b: __m512i) -> __m512i {
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
    unsafe fn _mm512_permutex2var_epi64_stub(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
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
            dst[i] = if (idx[i] >> 3) & 1 != 0 {
                b[off]
            } else {
                a[off]
            };
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_maddubs_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maddubs_epi16
    unsafe fn _mm512_maddubs_epi16_stub(a: __m512i, b: __m512i) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [i8; 64] = unsafe { transmute(b) };
        let mut dst = [0i16; 32];

        // FOR j := 0 to 31
        for j in 0..32 {
            // i := j*16
            let i = j * 2;
            // dst[i+15:i] := Saturate16( a[i+15:i+8]*b[i+15:i+8] + a[i+7:i]*b[i+7:i] )
            dst[j] = ((a[i + 1] as i16) * (b[i + 1] as i16))
                .saturating_add((a[i] as i16) * (b[i] as i16));
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_cvtepi16_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepi16_epi8
    unsafe fn _mm512_cvtepi16_epi8_stub(a: __m512i) -> __m256i {
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

    /// **Proof 1: Roundtrip Correctness (The Logic Check)**
    ///
    /// Verifies that `Decode(Encode(X)) == X`.
    #[kani::proof]
    #[kani::stub(_mm512_shuffle_epi8, _mm512_shuffle_epi8_stub)]
    #[kani::stub(_mm512_maddubs_epi16, _mm512_maddubs_epi16_stub)]
    #[kani::stub(_mm512_cvtepi16_epi8, _mm512_cvtepi16_epi8_stub)]
    #[kani::stub(_mm512_permutex2var_epi64, _mm512_permutex2var_epi64_stub)]
    fn check_avx512_roundtrip_correctness() {
        let config = Config {
            uppercase: kani::any(),
        };
        let input: [u8; ENC_INDUCTION_LEN] = kani::any();
        let input_len = input.len();

        // Buffers
        let mut enc_buf = [0u8; 256];
        let mut dec_buf = [0u8; 256];

        unsafe {
            // 1. Encode
            encode_slice_avx512(&config, &input, &mut enc_buf);

            // Calculate actual encoded length for slicing
            let encoded_slice = &enc_buf[..input_len * 2];

            // 2. Decode
            // This MUST succeed for valid encoded output
            decode_slice_avx512(encoded_slice, &mut dec_buf)
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
    #[kani::stub(_mm512_shuffle_epi8, _mm512_shuffle_epi8_stub)]
    #[kani::stub(_mm512_maddubs_epi16, _mm512_maddubs_epi16_stub)]
    #[kani::stub(_mm512_cvtepi16_epi8, _mm512_cvtepi16_epi8_stub)]
    fn check_avx512_decode_robustness() {
        // Input: `N` bytes of unrestricted symbolic data (garbage)
        let input: [u8; DEC_INDUCTION_LEN] = kani::any();

        // Output Buffer: Max estimated size
        let mut output = [0u8; 256];

        unsafe {
            // We ignore the Result. We only care that this function call
            // returns safely (Ok or Err) and does not crash.
            let _ = decode_slice_avx512(&input, &mut output);
        }
    }
}

// --- MIRI (FORMAL VERIFICATION) ---

#[cfg(all(test, miri))]
mod avx512_miri_tests {
    use super::{decode_slice_avx512, encode_slice_avx512};
    use crate::{Config, Error};

    // Reference crate
    use hex::encode as ref_encode_lower;

    // --- Deterministic Generator ---
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
            encode_slice_avx512(config, input, &mut enc_buf);
        }

        assert_eq!(
            &enc_buf[..],
            expected.as_bytes(),
            "AVX512 Encoding mismatch (len={})",
            len
        );

        // --- Decoding (own output) ---
        let mut dec_buf = vec![0u8; len];
        unsafe {
            decode_slice_avx512(&enc_buf, &mut dec_buf)
                .expect("AVX512 decoder failed on valid own output")
        };

        assert_eq!(
            &dec_buf[..],
            input,
            "AVX512 round-trip failed (len={})",
            len
        );
    }

    fn run_avx512_tests(uppercase: bool) {
        let config = Config { uppercase };

        // MIRI is slow. We don't need random lengths.
        // We only need to test boundary conditions to achieve 100% path coverage.
        let boundaries = [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128];

        for &len in &boundaries {
            let input = get_data(len);
            verify_roundtrip(&config, &input);
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
        // 128 length ensures we trigger 2 full AVX512 loops
        let input = get_data(128);
        let hex_lower = ref_encode_lower(&input).into_bytes();

        // Deterministically mix case
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

        let mut dec_buf = vec![0u8; 128];
        unsafe {
            decode_slice_avx512(&mixed_hex, &mut dec_buf)
                .expect("AVX512 decoder failed on valid mixed-case input")
        };

        assert_eq!(&dec_buf[..], input);
    }

    #[test]
    fn miri_avx512_decode_errors() {
        let mut out = [0u8; 128];

        // 1. Invalid character in SIMD region (128 hex chars = 64 bytes)
        let mut invalid_simd = vec![b'0'; 128];
        invalid_simd[65] = b'g'; // 'g' is strictly invalid in hex
        let res = unsafe { decode_slice_avx512(&invalid_simd, &mut out) };
        assert_eq!(res, Err(Error::InvalidCharacter));

        // 2. Invalid character in Scalar/Tail region (66 chars = 64 SIMD + 2 scalar)
        let mut invalid_tail = vec![b'0'; 66];
        invalid_tail[65] = b'g';
        let res = unsafe { decode_slice_avx512(&invalid_tail, &mut out) };
        assert_eq!(res, Err(Error::InvalidCharacter));
    }
}
