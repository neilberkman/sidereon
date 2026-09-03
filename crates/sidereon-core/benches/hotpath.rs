use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra::DMatrix;
use sidereon_core::astro::math::portable;
use sidereon_core::constants::F_L1_HZ;
use sidereon_core::ephemeris::{ObservableEphemerisSource, PreciseEphemerisInterpolant, Sp3};
use sidereon_core::observables::{
    emission_media_batch_at_j2000_s, emission_media_batch_at_j2000_s_into,
    emission_media_batch_at_j2000_s_with_receiver_context_into, EmissionMediaBatch,
    EmissionMediaBatchOptions, EmissionMediaReceiverContext, EmissionMediaStatus,
    ObservableIonosphereCorrection, ObservableMediaOptions,
};
use sidereon_core::positioning::{spp_inputs_from_rinex_obs, Corrections, RinexSppOptions};
use sidereon_core::rinex::observations::ObservationFile;
use sidereon_core::{
    atmosphere::{Ionex, IonexCoveragePolicy},
    geodetic_to_itrf, Wgs84Geodetic,
};

const SP3_BYTES: &[u8] =
    include_bytes!("../tests/fixtures/sp3/IGS0OPSFIN_20261330000_03H_15M_ORB.SP3");
const IONEX_BYTES: &[u8] = include_bytes!("../tests/fixtures/ionex/esa_2024176_first_map_2row.inx");
const OBS_TEXT: &str =
    include_str!("../tests/fixtures/obs/ZIM200CHE_R_20261330000_01H_30S_MO_120epoch.rnx");

struct CountingAlloc;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, old, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[derive(Clone)]
struct BundleRequest {
    satellites: Vec<sidereon_core::GnssSatelliteId>,
    emission_epochs: Vec<f64>,
}

#[derive(Clone)]
struct SolveRequest {
    inputs: sidereon_core::positioning::SolveInputs,
}

struct HotpathFixture {
    precise: PreciseEphemerisInterpolant,
    ionex: Ionex,
    receiver_ecef_m: [f64; 3],
    receiver_context: EmissionMediaReceiverContext,
    bundle: Vec<(usize, Vec<BundleRequest>)>,
    solve: Vec<(usize, Vec<SolveRequest>)>,
}

fn receiver_ecef_m() -> [f64; 3] {
    geodetic_to_itrf(
        Wgs84Geodetic::new(0.0_f64.to_radians(), 0.0_f64.to_radians(), 0.0)
            .expect("valid receiver"),
    )
    .expect("receiver to ECEF")
    .as_array()
}

fn media_options(ionex: &Ionex) -> ObservableMediaOptions<'_> {
    let mut options = ObservableMediaOptions::default();
    options.troposphere = Some(Default::default());
    options.ionosphere = Some(ObservableIonosphereCorrection::IonexWithPolicy(
        ionex,
        IonexCoveragePolicy::Hold,
    ));
    options
}

fn bundle_options(ionex: &Ionex) -> EmissionMediaBatchOptions<'_> {
    let mut options = EmissionMediaBatchOptions::default();
    options.carrier_hz = F_L1_HZ;
    options.media = media_options(ionex);
    options.min_elevation_rad = None;
    options
}

fn next_index(index: &mut usize, len: usize) -> usize {
    let current = *index;
    *index += 1;
    if *index == len {
        *index = 0;
    }
    current
}

fn measured_allocs<F, R>(mut f: F) -> (R, u64, u64)
where
    F: FnMut() -> R,
{
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    let result = f();
    let calls = ALLOC_CALLS.load(Ordering::Relaxed);
    let bytes = ALLOC_BYTES.load(Ordering::Relaxed);
    (result, calls, bytes)
}

fn build_bundle_requests(
    source: &dyn ObservableEphemerisSource,
    sp3: &Sp3,
    receiver_ecef_m: [f64; 3],
    ionex: &Ionex,
    count: usize,
) -> Vec<BundleRequest> {
    let satellites = sp3.satellites();
    let epochs = sp3.epochs_j2000_seconds();
    let mut requests = Vec::new();
    let mut k = 0usize;
    while requests.len() < 64 && k < 512 {
        let epoch_index = 2 + (k * 7) % (epochs.len() - 4);
        let epoch = if k.is_multiple_of(2) {
            epochs[epoch_index]
        } else {
            0.5 * (epochs[epoch_index] + epochs[epoch_index + 1])
        };
        let rotation = (k * 11) % satellites.len();
        let mut sats = Vec::with_capacity(count);
        for offset in 0..satellites.len() {
            let sat = satellites[(rotation + offset) % satellites.len()];
            let probe_epochs = [epoch];
            let probe = emission_media_batch_at_j2000_s(
                source,
                &[sat],
                &probe_epochs,
                receiver_ecef_m,
                bundle_options(ionex),
            )
            .expect("single-satellite bundle probe");
            if probe.element_status(0) == Some(EmissionMediaStatus::Valid) {
                sats.push(sat);
                if sats.len() == count {
                    break;
                }
            }
        }
        if sats.len() < count {
            k += 1;
            continue;
        }
        let emission_epochs = vec![epoch; sats.len()];
        let probe = emission_media_batch_at_j2000_s(
            source,
            &sats,
            &emission_epochs,
            receiver_ecef_m,
            bundle_options(ionex),
        )
        .expect("bundle probe");
        assert_eq!(probe.len(), count);
        requests.push(BundleRequest {
            satellites: sats,
            emission_epochs,
        });
        k += 1;
    }
    assert_eq!(
        requests.len(),
        64,
        "fixture must build valid {count}-satellite bundles"
    );
    requests
}

fn build_solve_requests(sp3: &Sp3, count: usize) -> Vec<SolveRequest> {
    let obs = ObservationFile::parse(OBS_TEXT).expect("parse hotpath OBS fixture");
    let options = RinexSppOptions::default_for(&obs)
        .expect("default signal policy")
        .with_corrections(Corrections::NONE);
    let assembled = spp_inputs_from_rinex_obs(&obs, sp3, &options).expect("assemble SPP inputs");
    let mut requests = Vec::new();
    for epoch in assembled {
        if epoch.inputs.observations.len() < count {
            continue;
        }
        let mut inputs = epoch.inputs.clone();
        inputs.observations.truncate(count);
        if sidereon_core::positioning::solve(sp3, &inputs, false).is_ok() {
            requests.push(SolveRequest { inputs });
        }
        if requests.len() == 64 {
            break;
        }
    }
    assert!(
        !requests.is_empty(),
        "fixture must produce a solvable {count}-satellite epoch"
    );
    requests
}

fn fixture() -> HotpathFixture {
    let sp3 = Sp3::parse(SP3_BYTES).expect("parse hotpath SP3 fixture");
    let precise = PreciseEphemerisInterpolant::from_sp3(&sp3);
    let ionex = Ionex::parse(IONEX_BYTES).expect("parse hotpath IONEX fixture");
    let receiver_ecef_m = receiver_ecef_m();
    let receiver_context =
        EmissionMediaReceiverContext::new(receiver_ecef_m).expect("receiver context");
    let bundle = [4, 8, 12]
        .into_iter()
        .map(|count| {
            (
                count,
                build_bundle_requests(&precise, &sp3, receiver_ecef_m, &ionex, count),
            )
        })
        .collect();
    let solve = [4, 8, 12]
        .into_iter()
        .map(|count| (count, build_solve_requests(&sp3, count)))
        .collect();
    HotpathFixture {
        precise,
        ionex,
        receiver_ecef_m,
        receiver_context,
        bundle,
        solve,
    }
}

fn print_allocation_snapshot(fx: &HotpathFixture) {
    for (count, requests) in &fx.bundle {
        let request = &requests[0];
        let (result, calls, bytes) = measured_allocs(|| {
            emission_media_batch_at_j2000_s(
                &fx.precise,
                &request.satellites,
                &request.emission_epochs,
                fx.receiver_ecef_m,
                bundle_options(&fx.ionex),
            )
            .expect("bundle alloc probe")
        });
        black_box(result);
        eprintln!("alloc hotpath_bundle/{count}_sat calls={calls} bytes={bytes}");

        let mut output = EmissionMediaBatch::with_capacity(request.satellites.len());
        emission_media_batch_at_j2000_s_into(
            &fx.precise,
            &request.satellites,
            &request.emission_epochs,
            fx.receiver_ecef_m,
            bundle_options(&fx.ionex),
            &mut output,
        )
        .expect("bundle alloc probe warm-up");
        let ((), calls, bytes) = measured_allocs(|| {
            emission_media_batch_at_j2000_s_into(
                &fx.precise,
                &request.satellites,
                &request.emission_epochs,
                fx.receiver_ecef_m,
                bundle_options(&fx.ionex),
                &mut output,
            )
            .expect("in-place bundle alloc probe")
        });
        black_box(output);
        eprintln!("alloc hotpath_bundle_into/{count}_sat calls={calls} bytes={bytes}");

        let mut output = EmissionMediaBatch::with_capacity(request.satellites.len());
        emission_media_batch_at_j2000_s_with_receiver_context_into(
            &fx.precise,
            &request.satellites,
            &request.emission_epochs,
            &fx.receiver_context,
            bundle_options(&fx.ionex),
            &mut output,
        )
        .expect("cached receiver bundle alloc probe warm-up");
        let ((), calls, bytes) = measured_allocs(|| {
            emission_media_batch_at_j2000_s_with_receiver_context_into(
                &fx.precise,
                &request.satellites,
                &request.emission_epochs,
                &fx.receiver_context,
                bundle_options(&fx.ionex),
                &mut output,
            )
            .expect("cached receiver bundle alloc probe")
        });
        black_box(output);
        eprintln!("alloc hotpath_bundle_cached/{count}_sat calls={calls} bytes={bytes}");
    }
    for (count, requests) in &fx.solve {
        let request = &requests[0];
        let (result, calls, bytes) = measured_allocs(|| {
            sidereon_core::positioning::solve(&fx.precise, &request.inputs, false)
                .expect("solve alloc probe")
        });
        black_box(result);
        eprintln!("alloc hotpath_solve/{count}_sat calls={calls} bytes={bytes}");
    }
}

fn bench_hotpath(c: &mut Criterion) {
    let fx = fixture();
    print_allocation_snapshot(&fx);

    let mut group = c.benchmark_group("hotpath_bundle");
    for (count, requests) in &fx.bundle {
        group.throughput(Throughput::Elements(1));
        let mut index = 0usize;
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{count}_sat")),
            count,
            |b, _| {
                b.iter(|| {
                    let request = &requests[next_index(&mut index, requests.len())];
                    let result = emission_media_batch_at_j2000_s(
                        black_box(&fx.precise),
                        black_box(&request.satellites),
                        black_box(&request.emission_epochs),
                        black_box(fx.receiver_ecef_m),
                        black_box(bundle_options(&fx.ionex)),
                    )
                    .expect("bundle benchmark");
                    black_box(result);
                })
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("hotpath_bundle_into");
    for (count, requests) in &fx.bundle {
        group.throughput(Throughput::Elements(1));
        let mut index = 0usize;
        let mut output = EmissionMediaBatch::with_capacity(*count);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{count}_sat")),
            count,
            |b, _| {
                b.iter(|| {
                    let request = &requests[next_index(&mut index, requests.len())];
                    emission_media_batch_at_j2000_s_into(
                        black_box(&fx.precise),
                        black_box(&request.satellites),
                        black_box(&request.emission_epochs),
                        black_box(fx.receiver_ecef_m),
                        black_box(bundle_options(&fx.ionex)),
                        black_box(&mut output),
                    )
                    .expect("in-place bundle benchmark");
                    black_box(&output);
                })
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("hotpath_bundle_cached");
    for (count, requests) in &fx.bundle {
        group.throughput(Throughput::Elements(1));
        let mut index = 0usize;
        let mut output = EmissionMediaBatch::with_capacity(*count);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{count}_sat")),
            count,
            |b, _| {
                b.iter(|| {
                    let request = &requests[next_index(&mut index, requests.len())];
                    sidereon_core::observables::emission_media_batch_at_j2000_s_with_receiver_context_into(
                        black_box(&fx.precise),
                        black_box(&request.satellites),
                        black_box(&request.emission_epochs),
                        black_box(&fx.receiver_context),
                        black_box(bundle_options(&fx.ionex)),
                        black_box(&mut output),
                    )
                    .expect("cached receiver bundle benchmark");
                    black_box(&output);
                })
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("hotpath_solve");
    for (count, requests) in &fx.solve {
        group.throughput(Throughput::Elements(1));
        let mut index = 0usize;
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{count}_sat")),
            count,
            |b, _| {
                b.iter(|| {
                    let request = &requests[next_index(&mut index, requests.len())];
                    let result = sidereon_core::positioning::solve(
                        black_box(&fx.precise),
                        black_box(&request.inputs),
                        black_box(false),
                    )
                    .expect("solve benchmark");
                    black_box(result);
                })
            },
        );
    }
    group.finish();
}

fn next_linalg_value(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(0xd1342543de82ef95)
        .wrapping_add(0xa4093822299f31d0);
    let fraction = f64::from_bits(0x3fe0000000000000 | (*state >> 12)) - 0.5;
    if *state & 1 == 0 {
        fraction
    } else {
        -fraction
    }
}

fn linalg_matrix(rows: usize, cols: usize, seed: u64) -> DMatrix<f64> {
    let mut state = seed;
    DMatrix::from_fn(rows, cols, |_, _| next_linalg_value(&mut state))
}

fn normal_matrix(order: usize) -> DMatrix<f64> {
    let factor = linalg_matrix(order, order, order as u64);
    let mut normal = &factor.transpose() * &factor;
    for diagonal in 0..order {
        normal[(diagonal, diagonal)] += order as f64 + 1.0;
    }
    normal
}

fn bench_linalg(c: &mut Criterion) {
    let mut products = c.benchmark_group("hotpath_linalg_product");
    for order in [50, 200, 500] {
        let normal = normal_matrix(order);
        products.throughput(Throughput::Elements(1));
        products.bench_with_input(
            BenchmarkId::new("plain_normal", order),
            &normal,
            |b, matrix| b.iter(|| black_box(black_box(matrix) * black_box(matrix))),
        );
        products.bench_with_input(
            BenchmarkId::new("portable_normal", order),
            &normal,
            |b, matrix| {
                b.iter(|| black_box(portable::product(black_box(matrix), black_box(matrix))))
            },
        );
    }

    let jacobian = linalg_matrix(2_000, 200, 0x517cc1b727220a95);
    let transposed = jacobian.transpose();
    products.throughput(Throughput::Elements(1));
    products.bench_function("plain_jacobian_2000x200", |b| {
        b.iter(|| black_box(black_box(&transposed) * black_box(&jacobian)))
    });
    products.bench_function("portable_jacobian_2000x200", |b| {
        b.iter(|| {
            black_box(portable::product(
                black_box(&transposed),
                black_box(&jacobian),
            ))
        })
    });
    products.finish();

    let mut cholesky = c.benchmark_group("hotpath_linalg_cholesky");
    for order in [50, 200, 500] {
        let normal = normal_matrix(order);
        cholesky.throughput(Throughput::Elements(1));
        cholesky.bench_with_input(
            BenchmarkId::new("plain_normal", order),
            &normal,
            |b, matrix| {
                b.iter(|| black_box(black_box(matrix).clone().cholesky().expect("SPD matrix")))
            },
        );
        cholesky.bench_with_input(
            BenchmarkId::new("portable_normal", order),
            &normal,
            |b, matrix| b.iter(|| black_box(portable::cholesky_lower_dynamic(black_box(matrix)))),
        );
    }
    cholesky.finish();
}

criterion_group!(benches, bench_hotpath, bench_linalg);
criterion_main!(benches);
