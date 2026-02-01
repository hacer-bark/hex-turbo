#[cfg(all(test, not(miri)))]
mod exhaustive_tests {
    // --- Imports ---
    use hex_turbo::{LOWER_CASE, UPPER_CASE};

    // Reference Crate
    use hex::{encode as ref_encode, decode as ref_decode};
    use rand::{Rng, rng};

    // --- Oracle Helpers ---
    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    /// The "Oracle" Test - compares against the `hex` crate and verifies round-trip.
    #[track_caller]
    fn assert_oracle_match_lower(input: &[u8]) {
        // 1. Truth (Reference crate - lowercase hex)
        let expected_encoded = ref_encode(input);

        // 2. Turbo Encode (LOWER_CASE)
        let mut enc_buf = vec![0u8; LOWER_CASE.encoded_len(input.len())];
        LOWER_CASE.encode_into(input, &mut enc_buf).expect("Turbo encode_into failed");

        // Verify content match (lowercase)
        assert!(enc_buf.as_slice() == expected_encoded.as_bytes(), "Encode content mismatch (lowercase). Input Len: {}", input.len());

        // 3. Turbo Decode (case-insensitive, but input is lowercase)
        let mut dec_buf = vec![0u8; LOWER_CASE.decoded_len(expected_encoded.len())];
        let written_dec = LOWER_CASE.decode_into(expected_encoded.as_bytes(), &mut dec_buf)
            .expect("Turbo decode_into failed on valid reference output");

        // Verify Round Trip
        let turbo_decoded_slice = &dec_buf[..written_dec];
        assert!(turbo_decoded_slice == input, "Decode content mismatch / Round trip failed (lowercase input). Input Len: {}", input.len());
    }

    #[track_caller]
    fn assert_oracle_match_upper(input: &[u8]) {
        // Encode with UPPER_CASE for mixed/uppercase testing
        let mut enc_buf_upper = vec![0u8; UPPER_CASE.encoded_len(input.len())];
        UPPER_CASE.encode_into(input, &mut enc_buf_upper).expect("Turbo encode_into (upper) failed");

        let expected_upper = ref_encode(input).to_ascii_uppercase();

        assert_eq!(
            enc_buf_upper.as_slice(), expected_upper.as_bytes(),
            "Encode content mismatch (uppercase). Input Len: {}",
            input.len()
        );

        // Decode uppercase output with LOWER_CASE engine (must be case-insensitive)
        let mut dec_buf = vec![0u8; LOWER_CASE.decoded_len(enc_buf_upper.len())];
        let written_dec = LOWER_CASE
            .decode_into(&enc_buf_upper, &mut dec_buf)
            .expect("Turbo decode_into failed on valid uppercase hex");

        let turbo_decoded_slice = &dec_buf[..written_dec];
        assert_eq!(
            turbo_decoded_slice, input,
            "Decode content mismatch / Round trip failed (uppercase input)"
        );
    }

    // --- 1. Basic Correctness ---
    #[test]
    fn test_oracle_exhaustive_small() {
        // Test 0 to 1024 bytes inclusive (covers all small corner cases)
        for i in 0..=1024 {
            let data = random_bytes(i);
            assert_oracle_match_lower(&data);
            assert_oracle_match_upper(&data);
        }
    }

    #[test]
    fn test_oracle_fuzz_medium() {
        let mut rng = rng();
        // 1,000 iterations of random sizes up to 64KB
        for _ in 0..1_000 {
            let len = rng.random_range(1025..=65536);
            let data = random_bytes(len);
            assert_oracle_match_lower(&data);
            assert_oracle_match_upper(&data);
        }
    }

    // --- SIMD / Scalar Transition Boundary Tests ---
    #[test]
    fn test_simd_scalar_transition_boundaries() {
        // Explicitly test lengths around SIMD chunk boundaries
        // AVX2: 32 input bytes -> 64 output
        // AVX512: 64 input bytes -> 128 output
        let boundaries = [
            0, 1, 2, 3, // Tiny inputs (scalar only)
            15, 16, 17, // Pre-AVX2
            31, 32, 33, // AVX2 boundary
            63, 64, 65, // AVX512 boundary
            127, 128, 129, // Double chunk
            255, 256, 257, // Edge cases near u8 overflow / cache lines
            1023, 1024, 1025, // Around 1KB
        ];

        for &len in &boundaries {
            let data = random_bytes(len);
            assert_oracle_match_lower(&data);
            assert_oracle_match_upper(&data);
        }
    }

    // --- Case-Insensitivity Explicit Tests ---
    #[test]
    fn test_decode_mixed_case() {
        // Mixed case input that reference `hex` crate accepts
        let mixed_hex = b"4a9fFfcC3e";
        let expected_bytes = ref_decode(mixed_hex).unwrap();

        let mut dec_buf = vec![0u8; LOWER_CASE.decoded_len(mixed_hex.len())];
        let written = LOWER_CASE
            .decode_into(mixed_hex, &mut dec_buf)
            .expect("Failed to decode valid mixed-case hex");

        assert_eq!(&dec_buf[..written], expected_bytes.as_slice());
    }

    // --- Negative / Error Path Tests ---
    #[test]
    fn test_reject_invalid_characters() {
        let invalid_inputs = vec![
            "gg", // 'g' invalid
            "A@", // '@' invalid
            "1Z", // 'Z' invalid
            "abc\n", // newline
            "abc ", // space
            "abc\0", // null
            "abc!", // garbage
            "abcdefgh", // 'g' and 'h' invalid in lower, but 'h' is valid, wait - 'h' is valid (0-9a-f)
            "abG", // 'G' is valid (case-insensitive)
            "abG!", // '!' invalid
            "ab\x7f", // DEL
            "ab\x1f", // control char
        ];

        let mut buf = [0u8; 100];
        for inp in invalid_inputs {
            assert!(
                LOWER_CASE.decode_into(inp.as_bytes(), &mut buf).is_err(),
                "Failed to reject invalid character in input: {:?}",
                inp
            );
        }
    }

    #[test]
    fn test_buffer_too_small_encode() {
        let data = random_bytes(10);
        let mut small_buf = vec![0u8; 19];
        assert!(
            LOWER_CASE.encode_into(&data, &mut small_buf).is_err(),
            "Encode did not reject too-small buffer"
        );
    }

    #[test]
    fn test_buffer_too_small_decode() {
        let valid_hex = b"ffffffffffffffffffff";
        let mut small_buf = vec![0u8; 9];
        assert!(
            LOWER_CASE.decode_into(valid_hex, &mut small_buf).is_err(),
            "Decode did not reject too-small buffer"
        );
    }

    #[test]
    fn test_buffer_overflow_protection_safe_api() {
        // Feed garbage that could theoretically overflow if logic is wrong
        let garbage = vec![b'f'; 1024];
        let mut buf = vec![0u8; 512];

        // Should either succeed (if valid) or Err, but never panic/write OOB
        let _ = LOWER_CASE.decode_into(&garbage, &mut buf);
    }

    #[test]
    fn test_empty_input() {
        let empty: [u8; 0] = [];
        let mut enc_buf = vec![0u8; 0];
        assert_eq!(LOWER_CASE.encode_into(&empty, &mut enc_buf).unwrap(), 0);

        let mut dec_buf = vec![0u8; 0];
        assert_eq!(LOWER_CASE.decode_into(&empty, &mut dec_buf).unwrap(), 0);
    }
}
