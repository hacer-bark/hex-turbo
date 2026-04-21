use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use rand::RngExt;
use std::env;
use std::hint::black_box;
use std::time::Duration;

// 1. The hex-turbo
use hex_turbo::{LOWER_CASE as TURBO_LOWER, UPPER_CASE as TURBO_UPPER};

// 2. Competitor 1: The standard 'hex' crate
use hex::{decode as decode_std, encode as encode_std};

// 3. Competitor 2: The 'faster-hex' crate
use faster_hex::{hex_decode as decode_fast, hex_encode as encode_fast};

// 4. Competitor 3: The 'hex-simd' crate
use hex_simd::{AsciiCase, decode_append, encode_append};

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

    // Logarithmic scaling is essential for viewing 32B vs 10MB
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(3));
    group.noise_threshold(0.05);
    group.sample_size(50);

    let sizes = [
        32,
        64,
        256,
        512,
        1024,
        4 * 1024,
        16 * 1024,
        64 * 1024,
        128 * 1024,
    ];

    for size in sizes.iter() {
        let input_data = generate_random_data(*size);

        // ======================================================================
        // ENCODE
        // ======================================================================
        group.throughput(Throughput::Bytes(*size as u64));

        // 1a. Hex Turbo (Upper Case - Allocating)
        if should_run("turbo") || should_run("upper") {
            group.bench_with_input(
                BenchmarkId::new("Encode/Turbo", size),
                &input_data,
                |b, d| b.iter(|| TURBO_UPPER.encode(black_box(d))),
            );
        }

        // 1b. Hex Turbo (Upper Case - Buff / No-Alloc)
        if should_run("turbo-buff") || should_run("upper-buff") {
            let encoded_len = TURBO_UPPER.encoded_len(*size);
            let mut output_buffer = vec![0u8; encoded_len];
            group.bench_with_input(
                BenchmarkId::new("Encode/TurboBuff", size),
                &input_data,
                |b, d| {
                    b.iter(|| TURBO_UPPER.encode_into(black_box(d), black_box(&mut output_buffer)))
                },
            );
        }

        // 2. Hex Standard (hex crate)
        if should_run("std") || should_run("hex") {
            group.bench_with_input(BenchmarkId::new("Encode/Std", size), &input_data, |b, d| {
                b.iter(|| encode_std(black_box(d)))
            });
        }

        // 3. Faster-hex (buffer-only)
        if should_run("fast") {
            let mut output_buffer = vec![0u8; *size * 2];
            group.bench_with_input(
                BenchmarkId::new("Encode/FastBuff", size),
                &input_data,
                |b, d| {
                    b.iter(|| {
                        encode_fast(black_box(d), black_box(&mut output_buffer)).unwrap();
                    })
                },
            );
        }

        // 4. Hex-SIMD (zero-allocation via append + truncate)
        if should_run("simd") {
            let mut output_buffer = vec![0u8; 0];
            output_buffer.reserve(*size * 2);
            group.bench_with_input(
                BenchmarkId::new("Encode/SimdBuff", size),
                &input_data,
                |b, d| {
                    b.iter(|| {
                        output_buffer.truncate(0);
                        encode_append(
                            black_box(d),
                            black_box(&mut output_buffer),
                            black_box(AsciiCase::Lower),
                        );
                    })
                },
            );
        }

        // ======================================================================
        // DECODE
        // ======================================================================
        let encoded_str = encode_std(&input_data);
        group.throughput(Throughput::Bytes(encoded_str.len() as u64));

        // 1a. Hex Turbo Decode (Allocating)
        if should_run("turbo") {
            group.bench_with_input(
                BenchmarkId::new("Decode/Turbo", size),
                &encoded_str,
                |b, s| b.iter(|| TURBO_LOWER.decode(black_box(s)).unwrap()),
            );
        }

        // 1b. Hex Turbo Decode (Buff / No-Alloc)
        if should_run("turbo-buff") {
            let decoded_len = TURBO_LOWER.decoded_len(encoded_str.len());
            let mut output_buffer = vec![0u8; decoded_len];
            group.bench_with_input(
                BenchmarkId::new("Decode/TurboBuff", size),
                &encoded_str,
                |b, s| {
                    b.iter(|| {
                        TURBO_LOWER
                            .decode_into(black_box(s.as_bytes()), black_box(&mut output_buffer))
                            .unwrap()
                    })
                },
            );
        }

        // 2. Hex Standard Decode (hex crate)
        if should_run("std") {
            group.bench_with_input(
                BenchmarkId::new("Decode/Std", size),
                &encoded_str,
                |b, s| b.iter(|| decode_std(black_box(s)).unwrap()),
            );
        }

        // 3. Faster-hex Decode (buffer-only)
        if should_run("fast") {
            let decoded_len = encoded_str.len() / 2;
            let mut output_buffer = vec![0u8; decoded_len];
            group.bench_with_input(
                BenchmarkId::new("Decode/FastBuff", size),
                &encoded_str,
                |b, s| {
                    b.iter(|| {
                        decode_fast(black_box(s.as_bytes()), black_box(&mut output_buffer))
                            .unwrap();
                    })
                },
            );
        }

        // 4. Hex-SIMD (zero-allocation via append + truncate)
        if should_run("simd") {
            let mut output_buffer = vec![0u8; 0];
            output_buffer.reserve(*size);
            group.bench_with_input(
                BenchmarkId::new("Decode/SimdBuff", size),
                &encoded_str,
                |b, s| {
                    b.iter(|| {
                        output_buffer.truncate(0);
                        decode_append(black_box(s), black_box(&mut output_buffer)).unwrap();
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_comparison);
criterion_main!(benches);
