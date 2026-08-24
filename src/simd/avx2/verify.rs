//! AVX2 verification: the Miri coverage suite. Split out of the production
//! module purely to keep it lean; nothing here is compiled into a normal
//! build.

// Re-exported to the `miri` submodule below; not compiled in a plain
// `cargo test`, so the glob looks unused there.
#[allow(unused_imports)]
use super::*;

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
    fn verify_roundtrip(config: Config, input: &[u8]) {
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
            verify_roundtrip(config, &input);
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
