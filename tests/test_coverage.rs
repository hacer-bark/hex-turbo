#[cfg(feature = "serde")]
mod coverage_tests {
    use hex_turbo::{Error, LOWER_CASE, decode, encode, encode_upper};
    use serde::{Deserialize, Serialize};

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

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_coverage() {
        use hex_turbo::serde::{hex, hex_upper, hex_upper32, hex32};
        use serde_test::{Token, assert_de_tokens, assert_de_tokens_error, assert_tokens};

        #[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
        struct TestStruct {
            #[serde(with = "hex")]
            bytes: Vec<u8>,
            #[serde(with = "hex_upper")]
            bytes_u: Vec<u8>,
        }

        let s = TestStruct {
            bytes: vec![0x01, 0x02],
            bytes_u: vec![0x03, 0x04],
        };

        assert_tokens(
            &s,
            &[
                Token::Struct {
                    name: "TestStruct",
                    len: 2,
                },
                Token::Str("bytes"),
                Token::Str("0102"),
                Token::Str("bytes_u"),
                Token::Str("0304"),
                Token::StructEnd,
            ],
        );

        assert_de_tokens(
            &s,
            &[
                Token::Struct {
                    name: "TestStruct",
                    len: 2,
                },
                Token::Str("bytes"),
                Token::String("0102"),
                Token::Str("bytes_u"),
                Token::Bytes(b"0304"),
                Token::StructEnd,
            ],
        );

        // Test fixed size 32
        #[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
        struct Test32 {
            #[serde(with = "hex32")]
            val: [u8; 32],
            #[serde(with = "hex_upper32")]
            val_u: [u8; 32],
        }

        let val = [0u8; 32];
        let s32 = Test32 { val, val_u: val };

        assert_tokens(
            &s32,
            &[
                Token::Struct {
                    name: "Test32",
                    len: 2,
                },
                Token::Str("val"),
                Token::Str("0000000000000000000000000000000000000000000000000000000000000000"),
                Token::Str("val_u"),
                Token::Str("0000000000000000000000000000000000000000000000000000000000000000"),
                Token::StructEnd,
            ],
        );

        // Test hex32 error - wrong length
        assert_de_tokens_error::<Test32>(
            &[
                Token::Struct {
                    name: "Test32",
                    len: 2,
                },
                Token::Str("val"),
                Token::Str("00"), // too short
            ],
            "expected 32 bytes, got 1",
        );

        // Test hex_upper32 error - wrong length (from bytes)
        assert_de_tokens_error::<Test32>(
            &[
                Token::Struct {
                    name: "Test32",
                    len: 2,
                },
                Token::Str("val"),
                Token::Str("0000000000000000000000000000000000000000000000000000000000000000"),
                Token::Str("val_u"),
                Token::Bytes(b"00"), // too short
            ],
            "expected 32 bytes, got 1",
        );

        // Test serialize error for hex32
        #[derive(Serialize)]
        struct Bad32 {
            #[serde(with = "hex32")]
            val: Vec<u8>,
        }
        let bad = Bad32 { val: vec![0u8; 31] };
        let res = serde_json::to_string(&bad);
        assert!(res.is_err());
        assert!(
            res.unwrap_err()
                .to_string()
                .contains("expected 32 bytes, got 31")
        );

        #[derive(Serialize)]
        struct BadUpper32 {
            #[serde(with = "hex_upper32")]
            val: Vec<u8>,
        }
        let bad_u = BadUpper32 { val: vec![0u8; 31] };
        let res_u = serde_json::to_string(&bad_u);
        assert!(res_u.is_err());
        assert!(
            res_u
                .unwrap_err()
                .to_string()
                .contains("expected 32 bytes, got 31")
        );
    }
}
