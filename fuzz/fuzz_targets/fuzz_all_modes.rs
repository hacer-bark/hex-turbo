#![no_main]
use libfuzzer_sys::fuzz_target;

use hex_turbo::{Engine, Error, LOWER_CASE as TURBO_LOWER, UPPER_CASE as TURBO_UPPER};

/// True when the host implements every subset the VBMI kernel issues. All three
/// are required: `vpermb`/`vpermi2b`/`vpmultishiftqb` are VBMI, the masked
/// `vmovdqu8` tiers are BW, and the 512-bit registers are F.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn has_avx512_vbmi() -> bool {
    std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vbmi")
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // The first byte picks the alphabet; the rest is the payload.
    let uppercase = data[0] & 1 == 1;
    let engine: &Engine = if uppercase { &TURBO_UPPER } else { &TURBO_LOWER };
    let payload = &data[1..];

    // ======================================================================
    // 1. Safe allocating APIs (.encode / .decode)
    // ======================================================================
    let encoded_ref = if uppercase {
        hex::encode_upper(payload)
    } else {
        hex::encode(payload)
    };
    let encoded_turbo = engine.encode(payload);
    assert_eq!(encoded_ref, encoded_turbo);

    let decoded = engine.decode(&encoded_turbo).unwrap();
    assert_eq!(decoded.as_slice(), payload);

    // Decoding is case-insensitive on either engine.
    let flipped = if uppercase {
        encoded_turbo.to_ascii_lowercase()
    } else {
        encoded_turbo.to_ascii_uppercase()
    };
    assert_eq!(engine.decode(&flipped).unwrap().as_slice(), payload);

    // ======================================================================
    // 2. Slice APIs (.encode_slice / .decode_slice)
    // ======================================================================
    let enc_len = engine.encoded_len(payload.len());
    let mut enc_buf = vec![0u8; enc_len.max(1)];

    let written_enc = engine.encode_slice(payload, &mut enc_buf[..enc_len]).unwrap();
    assert_eq!(written_enc, enc_len);
    assert_eq!(&enc_buf[..written_enc], encoded_turbo.as_bytes());

    // Insufficient buffer for encoding (must return error, no panic/UB)
    if enc_len > 0 {
        let mut small_enc = vec![0u8; enc_len - 1];
        assert!(matches!(
            engine.encode_slice(payload, &mut small_enc),
            Err(Error::BufferTooSmall)
        ));
    }

    let dec_len = engine.decoded_len(written_enc);
    let mut dec_buf = vec![0u8; dec_len.max(payload.len() + 16)];

    let written_dec = engine.decode_slice(&enc_buf[..written_enc], &mut dec_buf).unwrap();
    assert_eq!(&dec_buf[..written_dec], payload);

    // Arbitrary/invalid data must return `Err`, never panic or corrupt memory.
    let _ = engine.decode_slice(payload, &mut dec_buf);

    if !payload.is_empty() {
        let mut small_dec = vec![0u8; 1];
        let res = engine.decode_slice(payload, &mut small_dec);
        assert!(matches!(
            res,
            Err(Error::BufferTooSmall) | Err(Error::InvalidCharacter) | Err(Error::InvalidLength)
        ));
    }

    // ======================================================================
    // 3. Raw kernels (`unstable` feature)
    //
    //    Every buffer below is sized to *exactly* the capacity the kernel's
    //    safety contract asks for -- `encoded_len` to encode, `decoded_len` to
    //    decode -- and not a byte more. Slack here would hide the one bug class
    //    this section exists to find: a kernel whose overlapping or masked
    //    stores reach past the bound it documents. With ASan on, an overrun of
    //    these allocations is a hard failure.
    //
    //    Both valid and arbitrary input go through the decoders. The kernels
    //    fold validation into an accumulator they only test after their loops,
    //    so they may write garbage for invalid input -- but that garbage must
    //    still land inside `decoded_len`, and the call must report `Err` rather
    //    than panic.
    // ======================================================================

    let valid_encoded = &enc_buf[..written_enc];
    // The kernels take the input length at face value; an odd length is
    // rejected by the public API before dispatch, so mirror that here.
    let arbitrary = if payload.len() % 2 == 0 { payload } else { &payload[..payload.len() - 1] };
    let arbitrary_dec_len = engine.decoded_len(arbitrary.len());

    // Runs one kernel pair over: encode(payload), decode(valid), decode(arbitrary).
    macro_rules! exercise_kernel {
        ($name:literal, $encode:ident, $decode:ident) => {{
            if enc_len > 0 {
                let mut out_enc = vec![0u8; enc_len];
                unsafe { engine.$encode(payload, &mut out_enc) };
                assert_eq!(&out_enc[..], valid_encoded, concat!($name, ": encode mismatch"));
            }

            if !valid_encoded.is_empty() {
                let mut out_dec = vec![0u8; dec_len];
                let written = unsafe { engine.$decode(valid_encoded, &mut out_dec) }
                    .expect(concat!($name, ": valid input failed to decode"));
                assert_eq!(written, payload.len(), concat!($name, ": decoded length mismatch"));
                assert_eq!(&out_dec[..written], payload, concat!($name, ": decode mismatch"));
            }

            if !arbitrary.is_empty() {
                let mut out_dec = vec![0u8; arbitrary_dec_len];
                // Arbitrary bytes: any result is acceptable, a panic or an
                // out-of-bounds write is not.
                let _ = unsafe { engine.$decode(arbitrary, &mut out_dec) };
            }
        }};
    }

    // --- Scalar (always available) ---
    // Safe, not unsafe -- the scalar kernel forbids `unsafe` -- but exercised
    // through the same shape so the kernels stay comparable.
    if enc_len > 0 {
        let mut out_enc = vec![0u8; enc_len];
        engine.encode_scalar(payload, &mut out_enc);
        assert_eq!(&out_enc[..], valid_encoded, "scalar: encode mismatch");
    }

    if !valid_encoded.is_empty() {
        let mut out_dec = vec![0u8; dec_len];
        let written = engine
            .decode_scalar(valid_encoded, &mut out_dec)
            .expect("scalar: valid input failed to decode");
        assert_eq!(written, payload.len(), "scalar: decoded length mismatch");
        assert_eq!(&out_dec[..written], payload, "scalar: decode mismatch");
    }

    if !arbitrary.is_empty() {
        let mut out_dec = vec![0u8; arbitrary_dec_len];
        let _ = engine.decode_scalar(arbitrary, &mut out_dec);
    }

    // --- AVX2 (x86/x86_64 only) ---
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        exercise_kernel!("avx2", encode_avx2, decode_avx2);
    }

    // --- AVX-512-VBMI (x86/x86_64 only) ---
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if has_avx512_vbmi() {
        exercise_kernel!("avx512-vbmi", encode_avx512_vbmi, decode_avx512_vbmi);
    }
});
