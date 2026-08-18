#[cfg(feature = "std")]
mod coverage_tests {
    use hex_turbo::{Error, LOWER_CASE, decode, encode, encode_upper};

    #[test]
    fn test_top_level_functions() {
        let data = b"hello";
        let enc = encode(data);
        assert_eq!(enc, "68656c6c6f");
        let enc_u = encode_upper(data);
        assert_eq!(enc_u, "68656C6C6F");
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_scalar_8byte_error() {
        // Scalar 8-byte loop is for len >= 8.
        // We use len = 8 to trigger it but not AVX2 (which needs >= 32).
        let mut input = *b"0123456g";
        let mut out = [0u8; 4];
        assert_eq!(
            LOWER_CASE.decode_into(&input, &mut out),
            Err(Error::InvalidCharacter)
        );

        input = *b"g1234567";
        assert_eq!(
            LOWER_CASE.decode_into(&input, &mut out),
            Err(Error::InvalidCharacter)
        );
    }

    #[test]
    fn test_avx2_errors() {
        // AVX2 128-byte loop
        let mut input = vec![b'0'; 128];
        input[1] = b'g';
        let mut out = vec![0u8; 64];
        assert_eq!(
            LOWER_CASE.decode_into(&input, &mut out),
            Err(Error::InvalidCharacter)
        );

        // AVX2 32-byte loop
        let mut input = vec![b'0'; 32];
        input[1] = b'g';
        let mut out = vec![0u8; 16];
        assert_eq!(
            LOWER_CASE.decode_into(&input, &mut out),
            Err(Error::InvalidCharacter)
        );
    }
}
