#[cfg(all(test, not(miri)))]
mod exhaustive_tests {
    // --- Imports ---
    use hex_turbo::{Error, LOWER_CASE, UPPER_CASE};
    use hex::{encode as ref_encode};

    // --- High-Speed Deterministic Generator ---
    // Hex algorithm is a 1-to-1 byte mapping. High entropy (randomness) provides zero 
    // extra algorithmic coverage compared to sequential byte arrays, but sequential 
    // generation is exponentially faster for unit testing.
    fn get_deterministic_data(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 256) as u8).collect()
    }

    // --- Unified Oracle ---
    // Checks 100% of the Public API for a given payload in a single pass.
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
        assert_eq!(LOWER_CASE.encode_into(input, &mut enc_lower).unwrap(), enc_lower.len());
        assert_eq!(enc_lower, expected_lower.as_bytes(), "Lower encode_into mismatch");

        let mut enc_upper = vec![0u8; UPPER_CASE.encoded_len(input.len())];
        assert_eq!(UPPER_CASE.encode_into(input, &mut enc_upper).unwrap(), enc_upper.len());
        assert_eq!(enc_upper, expected_upper.as_bytes(), "Upper encode_into mismatch");

        // 3. Zero-Allocation API: Decode (Case Insensitive)
        let mut dec_buf = vec![0u8; LOWER_CASE.decoded_len(enc_lower.len())];
        assert_eq!(LOWER_CASE.decode_into(&enc_lower, &mut dec_buf).unwrap(), input.len());
        assert_eq!(dec_buf, input, "Decode lower mismatch");

        let mut dec_buf_upper = vec![0u8; UPPER_CASE.decoded_len(enc_upper.len())];
        assert_eq!(UPPER_CASE.decode_into(&enc_upper, &mut dec_buf_upper).unwrap(), input.len());
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

    // --- 1. Exhaustive Boundary Tests ---
    #[test]
    fn test_exhaustive_small_and_simd_boundaries() {
        let data = get_deterministic_data(256);

        // Exhaustively tests 0 to 256. This fully saturates:
        // - Scalar fallbacks
        // - SSE4.1 thresholds (16-byte bounds)
        // - AVX2 thresholds (32-byte bounds & tails)
        // - AVX512 thresholds (64-byte bounds & tails)
        for len in 0..=256 {
            check_engine(&data[..len]);
        }
    }

    // --- 2. High-Throughput / Large Chunk Tests ---
    #[test]
    fn test_large_payloads() {
        let data = get_deterministic_data(65536);
        
        // Jump straight to large power-of-two blocks to test SIMD unrolling limits
        for &size in &[1024, 2048, 4096, 16384, 65536] {
            check_engine(&data[..size]);
        }
    }

    // --- 3. Negative Paths & Error Handling ---
    #[test]
    fn test_public_api_errors() {
        let mut small_buf = [0u8; 2];
        let mut dec_buf = [0u8; 10];

        // A. Invalid Length (Must be divisible by 2)
        assert_eq!(LOWER_CASE.decode_into(b"123", &mut dec_buf), Err(Error::InvalidLength));
        #[cfg(feature = "std")]
        assert_eq!(LOWER_CASE.decode("123"), Err(Error::InvalidLength));

        // B. Buffer Too Small (Encode) - needs 4 bytes, given 2
        assert_eq!(LOWER_CASE.encode_into(b"ab", &mut small_buf), Err(Error::BufferTooSmall));

        // C. Buffer Too Small (Decode) - needs 2 bytes, given 1
        assert_eq!(LOWER_CASE.decode_into(b"1234", &mut small_buf[..1]), Err(Error::BufferTooSmall));

        // D. Invalid Characters (Out of [0-9a-fA-F] range)
        let bad_inputs = [
            b"g1".as_slice(), // 'g' is strictly invalid in Hex
            b"1G".as_slice(), // 'G' is strictly invalid in Hex
            b" a".as_slice(), // Space
            b"a\n".as_slice(),// Control Char / Newline
            b"a-".as_slice(), // Symbol
        ];

        for bad in bad_inputs {
            assert_eq!(LOWER_CASE.decode_into(bad, &mut dec_buf), Err(Error::InvalidCharacter));
            #[cfg(feature = "std")]
            assert_eq!(LOWER_CASE.decode(bad), Err(Error::InvalidCharacter));
        }

        // E. Verify Display implementation formats correctly
        #[cfg(feature = "std")]
        {
            assert_eq!(Error::InvalidLength.to_string(), "Invalid Hex input length (must be divisible by 2)");
            assert_eq!(Error::InvalidCharacter.to_string(), "Invalid character found in Hex input");
            assert_eq!(Error::BufferTooSmall.to_string(), "Destination buffer is too small");
        }
    }

    // --- 4. Empty Input Edge Case ---
    #[test]
    fn test_empty_input() {
        let empty: [u8; 0] = [];
        let mut buf = [0u8; 0];

        // Zero-Alloc API
        assert_eq!(LOWER_CASE.encode_into(&empty, &mut buf), Ok(0));
        assert_eq!(LOWER_CASE.decode_into(&empty, &mut buf), Ok(0));

        // Allocating API
        #[cfg(feature = "std")]
        {
            assert_eq!(LOWER_CASE.encode(&empty), "");
            assert_eq!(LOWER_CASE.decode("").unwrap(), empty);
        }
    }
}
