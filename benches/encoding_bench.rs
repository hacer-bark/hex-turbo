//! Throughput benchmarks comparing `hex-turbo` against the `hex`, `faster-hex`
//! and `hex-simd` crates.
//!
//! Every engine writes into a caller-owned buffer that is allocated once,
//! outside the timed loop, so what is measured is the codec and nothing else.
//! Every engine also produces lowercase, so all four are doing identical work.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    missing_docs,
    clippy::too_many_lines
)]

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use rand::RngExt;
use std::env;
use std::hint::black_box;
use std::time::Duration;

// 1. The hex-turbo
use hex_turbo::LOWER_CASE as TURBO;

// 2. The standard 'hex' crate
use hex::{
    decode_to_slice as decode_std, encode as encode_std, encode_to_slice as encode_std_into,
};

// 3. The 'faster-hex' crate
use faster_hex::{hex_decode as decode_fast, hex_encode as encode_fast};

// 4. The 'hex-simd' crate
use hex_simd::{AsOut, AsciiCase, decode as decode_simd, encode as encode_simd};

fn generate_random_data(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size];
    rand::rng().fill(&mut data[..]);
    data
}

/// Helper to check if a specific engine should be benchmarked based on ENV vars.
/// Usage: `BENCH_TARGET=turbo cargo bench` or `BENCH_TARGET=all cargo bench`
fn should_run(target_name: &str) -> bool {
    let var = env::var("BENCH_TARGET").unwrap_or_else(|_| "turbo".to_string());
    let targets: Vec<String> = var.split(',').map(|s| s.trim().to_lowercase()).collect();
    if targets.contains(&"all".to_string()) {
        return true;
    }
    targets.contains(&target_name.to_lowercase())
}

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hex_Performances");

    // Logarithmic scaling to view 32 B and 10 MB on the same axis.
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(5));
    group.noise_threshold(0.05);
    group.sample_size(50);

    let sizes = [
        32,               // 32 B
        512,              // 512 B
        4 * 1024,         // 4 KB
        64 * 1024,        // 64 KB
        512 * 1024,       // 512 KB
        1024 * 1024,      // 1 MB
        10 * 1024 * 1024, // 10 MB
    ];

    for size in &sizes {
        let input_data = generate_random_data(*size);
        let encoded_str = encode_std(&input_data);

        // ======================================================================
        // ENCODE
        // ======================================================================
        group.throughput(Throughput::Bytes(*size as u64));

        let mut encode_buffer = vec![0u8; *size * 2];

        // 1. Hex Turbo
        if should_run("turbo") {
            group.bench_with_input(
                BenchmarkId::new("Encode/Turbo", size),
                &input_data,
                |b, d| {
                    b.iter(|| {
                        TURBO
                            .encode_slice(black_box(d), black_box(&mut encode_buffer))
                            .unwrap()
                    });
                },
            );
        }

        // 2. Hex Standard (hex crate)
        if should_run("std") || should_run("hex") {
            group.bench_with_input(BenchmarkId::new("Encode/Std", size), &input_data, |b, d| {
                b.iter(|| {
                    encode_std_into(black_box(d), black_box(&mut encode_buffer)).unwrap();
                });
            });
        }

        // 3. Faster-hex
        if should_run("fast") {
            group.bench_with_input(
                BenchmarkId::new("Encode/Fast", size),
                &input_data,
                |b, d| {
                    b.iter(|| {
                        encode_fast(black_box(d), black_box(&mut encode_buffer)).unwrap();
                    });
                },
            );
        }

        // 4. Hex-SIMD
        if should_run("simd") {
            group.bench_with_input(
                BenchmarkId::new("Encode/Simd", size),
                &input_data,
                |b, d| {
                    b.iter(|| {
                        black_box(encode_simd(
                            black_box(d),
                            black_box(&mut encode_buffer).as_out(),
                            AsciiCase::Lower,
                        ));
                    });
                },
            );
        }

        // ======================================================================
        // DECODE
        // ======================================================================
        group.throughput(Throughput::Bytes(encoded_str.len() as u64));

        let mut decode_buffer = vec![0u8; *size];

        // 1. Hex Turbo
        if should_run("turbo") {
            group.bench_with_input(
                BenchmarkId::new("Decode/Turbo", size),
                &encoded_str,
                |b, s| {
                    b.iter(|| {
                        TURBO
                            .decode_slice(black_box(s.as_bytes()), black_box(&mut decode_buffer))
                            .unwrap()
                    });
                },
            );
        }

        // 2. Hex Standard (hex crate)
        if should_run("std") || should_run("hex") {
            group.bench_with_input(
                BenchmarkId::new("Decode/Std", size),
                &encoded_str,
                |b, s| {
                    b.iter(|| {
                        decode_std(black_box(s), black_box(&mut decode_buffer)).unwrap();
                    });
                },
            );
        }

        // 3. Faster-hex
        if should_run("fast") {
            group.bench_with_input(
                BenchmarkId::new("Decode/Fast", size),
                &encoded_str,
                |b, s| {
                    b.iter(|| {
                        decode_fast(black_box(s.as_bytes()), black_box(&mut decode_buffer))
                            .unwrap();
                    });
                },
            );
        }

        // 4. Hex-SIMD
        if should_run("simd") {
            group.bench_with_input(
                BenchmarkId::new("Decode/Simd", size),
                &encoded_str,
                |b, s| {
                    b.iter(|| {
                        black_box(
                            decode_simd(
                                black_box(s.as_bytes()),
                                black_box(&mut decode_buffer).as_out(),
                            )
                            .unwrap(),
                        );
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_comparison);
criterion_main!(benches);
