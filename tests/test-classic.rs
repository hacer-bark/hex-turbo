//! Integration tests verifying `hex-turbo`'s output against the reference `hex` crate.

#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

#[cfg(not(miri))]
mod classic {
    // Reference crate for oracle verification.
    use hex::encode as ref_encode;
    use hex_turbo::{Error, LOWER_CASE, UPPER_CASE};
    #[cfg(feature = "std")]
    use hex_turbo::{decode, encode, encode_upper};
    use rand::RngExt;

    // ======================================================================
    // Helpers
    // ======================================================================

    fn random_bytes(len: usize) -> Vec<u8> {
        let mut bytes = vec![0; len];
        rand::rng().fill(&mut bytes);
        bytes
    }

    /// Checks the whole public API for one payload in a single pass.
    #[track_caller]
    fn check_engine(input: &[u8]) {
        let expected_lower = ref_encode(input);
        let expected_upper = expected_lower.to_ascii_uppercase();

        // 1. Check Length Calculators
        assert_eq!(LOWER_CASE.encoded_len(input.len()), expected_lower.len());
        assert_eq!(UPPER_CASE.encoded_len(input.len()), expected_upper.len());
        assert_eq!(LOWER_CASE.decoded_len(expected_lower.len()), input.len());

        // 2. Zero-Allocation API: Encode (Lower & Upper)
        let mut enc_lower = vec![0u8; LOWER_CASE.encoded_len(input.len())];
        assert_eq!(
            LOWER_CASE.encode_slice(input, &mut enc_lower).unwrap(),
            enc_lower.len()
        );
        assert_eq!(
            enc_lower,
            expected_lower.as_bytes(),
            "Lower encode_slice mismatch"
        );

        let mut enc_upper = vec![0u8; UPPER_CASE.encoded_len(input.len())];
        assert_eq!(
            UPPER_CASE.encode_slice(input, &mut enc_upper).unwrap(),
            enc_upper.len()
        );
        assert_eq!(
            enc_upper,
            expected_upper.as_bytes(),
            "Upper encode_slice mismatch"
        );

        // 3. Zero-Allocation API: Decode (Case Insensitive)
        let mut dec_buf = vec![0u8; LOWER_CASE.decoded_len(enc_lower.len())];
        assert_eq!(
            LOWER_CASE.decode_slice(&enc_lower, &mut dec_buf).unwrap(),
            input.len()
        );
        assert_eq!(dec_buf, input, "Decode lower mismatch");

        let mut dec_buf_upper = vec![0u8; UPPER_CASE.decoded_len(enc_upper.len())];
        assert_eq!(
            UPPER_CASE
                .decode_slice(&enc_upper, &mut dec_buf_upper)
                .unwrap(),
            input.len()
        );
        assert_eq!(dec_buf_upper, input, "Decode upper mismatch");

        // 4. Allocating APIs (`std` only)
        #[cfg(feature = "std")]
        {
            assert_eq!(LOWER_CASE.encode(input), expected_lower);
            assert_eq!(UPPER_CASE.encode(input), expected_upper);
            assert_eq!(LOWER_CASE.decode(&expected_lower).unwrap(), input);
            assert_eq!(UPPER_CASE.decode(&expected_upper).unwrap(), input);
        }
    }

    // ======================================================================
    // Tests
    // ======================================================================

    #[test]
    fn exhaustive_small_and_simd_boundaries() {
        let data = random_bytes(256);

        // Every length from 0 to 256 saturates the scalar fallback, the AVX2
        // thresholds (32-byte bounds & tails) and the AVX-512 VBMI ones
        // (64-byte bounds & tails).
        for len in 0..=256 {
            check_engine(&data[..len]);
        }
    }

    #[test]
    fn large_payloads() {
        let data = random_bytes(65536);

        // Jump straight to large power-of-two blocks to test SIMD unrolling limits
        for &size in &[1024, 2048, 4096, 16384, 65536] {
            check_engine(&data[..size]);
        }
    }

    #[test]
    fn public_api_errors() {
        let mut small_buf = [0u8; 2];
        let mut dec_buf = [0u8; 10];

        // A. Invalid Length (Must be divisible by 2)
        assert_eq!(
            LOWER_CASE.decode_slice(b"123", &mut dec_buf),
            Err(Error::InvalidLength)
        );
        #[cfg(feature = "std")]
        assert_eq!(LOWER_CASE.decode("123"), Err(Error::InvalidLength));

        // B. Buffer Too Small (Encode) - needs 4 bytes, given 2
        assert_eq!(
            LOWER_CASE.encode_slice(b"ab", &mut small_buf),
            Err(Error::BufferTooSmall)
        );

        // C. Buffer Too Small (Decode) - needs 2 bytes, given 1
        assert_eq!(
            LOWER_CASE.decode_slice(b"1234", &mut small_buf[..1]),
            Err(Error::BufferTooSmall)
        );

        // D. Invalid Characters (Out of [0-9a-fA-F] range)
        let bad_inputs = [
            b"g1".as_slice(),  // 'g' is strictly invalid in Hex
            b"1G".as_slice(),  // 'G' is strictly invalid in Hex
            b" a".as_slice(),  // Space
            b"a\n".as_slice(), // Control Char / Newline
            b"a-".as_slice(),  // Symbol
        ];

        for bad in bad_inputs {
            assert_eq!(
                LOWER_CASE.decode_slice(bad, &mut dec_buf),
                Err(Error::InvalidCharacter)
            );
            #[cfg(feature = "std")]
            assert_eq!(LOWER_CASE.decode(bad), Err(Error::InvalidCharacter));
        }

        // E. Verify Display implementation formats correctly
        #[cfg(feature = "std")]
        {
            assert_eq!(
                Error::InvalidLength.to_string(),
                "Invalid Hex input length (must be divisible by 2)"
            );
            assert_eq!(
                Error::InvalidCharacter.to_string(),
                "Invalid character found in Hex input"
            );
            assert_eq!(
                Error::BufferTooSmall.to_string(),
                "Destination buffer is too small"
            );
        }
    }

    #[test]
    fn empty_input() {
        let empty: [u8; 0] = [];
        let mut buf = [0u8; 0];

        // Zero-Alloc API
        assert_eq!(LOWER_CASE.encode_slice(empty, &mut buf), Ok(0));
        assert_eq!(LOWER_CASE.decode_slice(empty, &mut buf), Ok(0));

        // Allocating API
        #[cfg(feature = "std")]
        {
            assert_eq!(LOWER_CASE.encode(empty), "");
            assert_eq!(LOWER_CASE.decode("").unwrap(), empty);
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn free_functions_wrap_the_engine_constants() {
        let data = b"hello";
        assert_eq!(encode(data), "68656c6c6f");
        assert_eq!(encode_upper(data), "68656C6C6F");
        assert_eq!(decode(encode(data)).unwrap(), data);
    }

    /// One bad character has to be caught by whichever loop the length lands in:
    /// the scalar 8-byte group, the AVX2 32-byte body, and its 128-byte unrolled
    /// body all accumulate validity separately.
    #[test]
    fn invalid_character_rejected_in_every_decode_loop() {
        for len in [8usize, 32, 128] {
            for pos in [0, 1, len - 1] {
                let mut input = vec![b'0'; len];
                input[pos] = b'g';
                let mut out = vec![0u8; len / 2];
                assert_eq!(
                    LOWER_CASE.decode_slice(&input, &mut out),
                    Err(Error::InvalidCharacter),
                    "accepted 'g' at position {pos} of a {len}-character input"
                );
            }
        }
    }
}
