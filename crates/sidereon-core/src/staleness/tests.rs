//! Selection-layer tests for product-staleness graceful degradation.
//!
//! Two correctness bars are covered:
//!
//! - **Present path is byte-identical**: when a product covers the requested
//!   epoch the selection delegates to the existing IONEX slant / SP3
//!   interpolation entry points on the caller's own product, so the result bits
//!   match calling those entry points directly. Asserted by `to_bits()` equality.
//! - **Missing path is correct**: with the exact-day product withheld from a set,
//!   the IONEX path returns the prior day's grid advanced by whole days (the grid
//!   values unchanged, the epoch axis shifted, the slant delay equal to the prior
//!   day at the same time-of-day) and the SP3 path returns the nearest-prior
//!   product, both with the correct staleness metadata.

use super::*;

use crate::frame::Wgs84Geodetic;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::ionex::ionex_slant_delay;

const L1_HZ: f64 = 1_575_420_000.0;

fn ionex_record(data: &str, label: &str) -> String {
    format!("{data:<60}{label}\n")
}

/// A synthetic single-day IONEX product with two maps (00:00 and 06:00) on a
/// 3x3 lat/lon grid. `base_tec` shifts every TEC value so distinct days carry
/// distinct grids.
fn make_ionex(year: i64, month: i64, day: i64, base_tec: i64) -> Ionex {
    let mut text = String::new();
    text.push_str(&ionex_record("1.0", "IONEX VERSION / TYPE"));
    text.push_str(&ionex_record("2", "# OF MAPS IN FILE"));
    text.push_str(&ionex_record("2.0 -2.0 -2.0", "LAT1 / LAT2 / DLAT"));
    text.push_str(&ionex_record("-2.0 2.0 2.0", "LON1 / LON2 / DLON"));
    text.push_str(&ionex_record("450.0 450.0 0.0", "HGT1 / HGT2 / DHGT"));
    text.push_str(&ionex_record("6371.0", "BASE RADIUS"));
    text.push_str(&ionex_record("0", "EXPONENT"));
    text.push_str(&ionex_record("", "END OF HEADER"));
    for (map_idx, hour) in [(1i64, 0i64), (2, 6)] {
        text.push_str(&ionex_record(&map_idx.to_string(), "START OF TEC MAP"));
        text.push_str(&ionex_record(
            &format!("{year} {month} {day} {hour} 0 0"),
            "EPOCH OF CURRENT MAP",
        ));
        for lat_idx in 0..3i64 {
            let lat = 2.0 - (lat_idx as f64) * 2.0;
            text.push_str(&ionex_record(
                &format!("{lat:.1} -2.0 2.0 2.0 450.0"),
                "LAT/LON1/LON2/DLON/H",
            ));
            let a = base_tec + lat_idx * 10 + map_idx * 100;
            text.push_str(&format!("{} {} {}\n", a, a + 1, a + 2));
        }
        text.push_str(&ionex_record(&map_idx.to_string(), "END OF TEC MAP"));
    }
    Ionex::parse_str(&text).expect("valid synthetic IONEX")
}

/// A synthetic single-day SP3-c product with two epochs (00:00 and 00:15) on the
/// given calendar date. The header (fixed-column) is constant; only the
/// whitespace-delimited epoch lines move the product in time.
fn make_sp3(year: i64, month: i64, day: i64) -> Sp3 {
    let e0 = format!("*  {year} {month} {day}  0  0  0.00000000");
    let e1 = format!("*  {year} {month} {day}  0 15  0.00000000");
    let text = format!(
        "\
#cP2020  6 24  0  0  0.00000000       2 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    2   G01G02  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* TEST SP3 FIXTURE
{e0}
PG01  15000.000000 -20000.000000   5000.000000    123.456789
PG02  -1234.567890   2345.678901  -3456.789012    100.000000
{e1}
PG01  15100.000000 -20100.000000   5100.000000   -987.654321
PG02  -1334.567890   2445.678901  -3556.789012    100.000000
EOF
"
    );
    Sp3::parse(text.as_bytes()).expect("valid synthetic SP3")
}

/// A synthetic IONEX product with a map at each `(day, hour)` on the given
/// year/month, so a product can span several days for tie-break tests. `base_tec`
/// shifts every TEC value so distinct products carry distinct grids.
fn make_ionex_days(year: i64, month: i64, day_hours: &[(i64, i64)], base_tec: i64) -> Ionex {
    let mut text = String::new();
    text.push_str(&ionex_record("1.0", "IONEX VERSION / TYPE"));
    text.push_str(&ionex_record(
        &day_hours.len().to_string(),
        "# OF MAPS IN FILE",
    ));
    text.push_str(&ionex_record("2.0 -2.0 -2.0", "LAT1 / LAT2 / DLAT"));
    text.push_str(&ionex_record("-2.0 2.0 2.0", "LON1 / LON2 / DLON"));
    text.push_str(&ionex_record("450.0 450.0 0.0", "HGT1 / HGT2 / DHGT"));
    text.push_str(&ionex_record("6371.0", "BASE RADIUS"));
    text.push_str(&ionex_record("0", "EXPONENT"));
    text.push_str(&ionex_record("", "END OF HEADER"));
    for (i, (day, hour)) in day_hours.iter().enumerate() {
        let map_idx = (i + 1) as i64;
        text.push_str(&ionex_record(&map_idx.to_string(), "START OF TEC MAP"));
        text.push_str(&ionex_record(
            &format!("{year} {month} {day} {hour} 0 0"),
            "EPOCH OF CURRENT MAP",
        ));
        for lat_idx in 0..3i64 {
            let lat = 2.0 - (lat_idx as f64) * 2.0;
            text.push_str(&ionex_record(
                &format!("{lat:.1} -2.0 2.0 2.0 450.0"),
                "LAT/LON1/LON2/DLON/H",
            ));
            let a = base_tec + lat_idx * 10 + map_idx * 100;
            text.push_str(&format!("{} {} {}\n", a, a + 1, a + 2));
        }
        text.push_str(&ionex_record(&map_idx.to_string(), "END OF TEC MAP"));
    }
    Ionex::parse_str(&text).expect("valid synthetic IONEX")
}

/// A synthetic SP3-c product with one position record per `(day, hour, minute)`
/// epoch on year 2024 month 3, so a product can straddle days for tie-break and
/// range tests. The fixed-column header is constant; only the epoch lines move.
fn make_sp3_epochs(day_h_m: &[(i64, i64, i64)]) -> Sp3 {
    let mut body = String::new();
    for (day, hour, minute) in day_h_m {
        body.push_str(&format!("*  2024 3 {day} {hour} {minute}  0.00000000\n"));
        body.push_str("PG01  15000.000000 -20000.000000   5000.000000    123.456789\n");
        body.push_str("PG02  -1234.567890   2345.678901  -3456.789012    100.000000\n");
    }
    let text = format!(
        "\
#cP2020  6 24  0  0  0.00000000       2 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    2   G01G02  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* TEST SP3 FIXTURE
{body}EOF
"
    );
    Sp3::parse(text.as_bytes()).expect("valid synthetic SP3")
}

fn receiver() -> Wgs84Geodetic {
    Wgs84Geodetic::new(0.01, 0.01, 0.0).expect("valid receiver")
}

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id")
}

// ---------------------------------------------------------------------------
// IONEX: present path is byte-identical.
// ---------------------------------------------------------------------------

#[test]
fn ionex_present_path_is_byte_identical() {
    let day = make_ionex(2024, 3, 10, 50);
    let span = day.map_epochs_s();
    let requested = span[0] + 3_600; // 01:00, inside the 00:00..06:00 span

    let direct = ionex_slant_delay(&day, receiver(), 0.5, 0.3, requested, L1_HZ)
        .expect("direct slant delay");

    let set = [day.clone()];
    let selection =
        select_ionex(&set, requested, StalenessPolicy::default()).expect("exact selection");
    let via_layer = selection
        .slant_delay(receiver(), 0.5, 0.3, requested, L1_HZ)
        .expect("layered slant delay");

    assert_eq!(selection.metadata().kind, DegradationKind::Exact);
    assert_eq!(selection.metadata().staleness_s, 0.0);
    assert_eq!(
        via_layer.to_bits(),
        direct.to_bits(),
        "present IONEX slant delay must be bit-identical to the direct call"
    );
}

#[test]
fn ionex_present_path_byte_identical_when_chosen_from_a_set() {
    // A set with a future product present too; the covering product must be
    // picked and the result still byte-identical to the direct call on it.
    let day = make_ionex(2024, 3, 10, 50);
    let later = make_ionex(2024, 3, 12, 70);
    let span = day.map_epochs_s();
    let requested = span[0] + 7_200; // 02:00, inside `day`

    let direct = ionex_slant_delay(&day, receiver(), 0.4, 0.2, requested, L1_HZ).expect("direct");

    let set = [later, day.clone()];
    let selection = select_ionex(&set, requested, StalenessPolicy::default()).expect("selection");
    assert_eq!(selection.metadata().kind, DegradationKind::Exact);
    let via_layer = selection
        .slant_delay(receiver(), 0.4, 0.2, requested, L1_HZ)
        .expect("layer");
    assert_eq!(via_layer.to_bits(), direct.to_bits());
}

// ---------------------------------------------------------------------------
// IONEX: missing path is a correct whole-day diurnal shift.
// ---------------------------------------------------------------------------

#[test]
fn ionex_missing_day_uses_diurnal_shift_of_prior_day() {
    // Days D and D+2 present; D+1 withheld. Request 03:00 on D+1.
    let day0 = make_ionex(2024, 3, 10, 50);
    let day2 = make_ionex(2024, 3, 12, 70);
    let d0_span = day0.map_epochs_s();
    let one_day = 86_400i64;
    let requested = d0_span[0] + one_day + 3 * 3_600; // D+1 03:00

    let set = [day2, day0.clone()];
    let selection =
        select_ionex(&set, requested, StalenessPolicy::default()).expect("diurnal selection");
    let meta = selection.metadata();

    assert_eq!(meta.kind, DegradationKind::DiurnalShift);
    assert_eq!(meta.staleness_days, 1.0);
    assert_eq!(meta.staleness_s, one_day as f64);
    assert_eq!(meta.requested_epoch_j2000_s, requested as f64);
    assert_eq!(meta.source_epoch_j2000_s, (requested - one_day) as f64);

    // (a) The shifted grid equals the prior day's grid shifted by whole days:
    // identical TEC values, epoch axis advanced by one day.
    let shifted = selection.ionex();
    assert_eq!(
        shifted.tec_maps(),
        day0.tec_maps(),
        "diurnal shift must not change the grid values"
    );
    for (s, base) in shifted
        .map_epochs_s()
        .iter()
        .zip(day0.map_epochs_s().iter())
    {
        assert_eq!(*s, *base + one_day);
    }

    // The slant delay at the requested epoch equals the prior day's delay at the
    // same time-of-day, bit-for-bit (diurnal persistence is near-lossless here).
    let via_layer = selection
        .slant_delay(receiver(), 0.5, 0.3, requested, L1_HZ)
        .expect("shifted slant delay");
    let prior_same_tod = ionex_slant_delay(&day0, receiver(), 0.5, 0.3, requested - one_day, L1_HZ)
        .expect("prior-day slant delay");
    assert_eq!(via_layer.to_bits(), prior_same_tod.to_bits());
}

#[test]
fn ionex_beyond_cap_is_a_typed_error() {
    let day0 = make_ionex(2024, 3, 10, 50);
    let span = day0.map_epochs_s();
    let requested = span[1] + 10 * 86_400; // 10 days past the only product

    let set = [day0];
    let err = select_ionex(&set, requested, StalenessPolicy::default())
        .expect_err("must exceed the 3-day cap");
    match err {
        SelectionError::BeyondStalenessCap {
            staleness_s,
            max_staleness_s,
            ..
        } => {
            assert!(staleness_s > max_staleness_s);
            assert_eq!(max_staleness_s, 3.0 * 86_400.0);
        }
        other => panic!("expected BeyondStalenessCap, got {other:?}"),
    }
}

#[test]
fn ionex_no_prior_product_is_a_typed_error() {
    let day0 = make_ionex(2024, 3, 10, 50);
    let span = day0.map_epochs_s();
    let requested = span[0] - 86_400; // before the only product

    let set = [day0];
    let err = select_ionex(&set, requested, StalenessPolicy::default()).expect_err("nothing prior");
    assert!(matches!(err, SelectionError::NoPriorProduct { .. }));
}

#[test]
fn ionex_empty_set_is_a_typed_error() {
    let err = select_ionex(&[], 0, StalenessPolicy::default()).expect_err("empty set");
    assert_eq!(err, SelectionError::EmptyProductSet);
}

#[test]
fn ionex_range_diurnal_shift_covers_the_whole_window() {
    let day0 = make_ionex(2024, 3, 10, 50);
    let day2 = make_ionex(2024, 3, 12, 70);
    let d0_span = day0.map_epochs_s();
    let one_day = 86_400i64;
    let start = d0_span[0] + one_day + 3_600; // D+1 01:00
    let end = d0_span[0] + one_day + 5 * 3_600; // D+1 05:00

    let set = [day2, day0.clone()];
    let selection = select_ionex_over_range(&set, start, end, StalenessPolicy::default())
        .expect("range selection");
    assert_eq!(selection.metadata().kind, DegradationKind::DiurnalShift);
    assert_eq!(selection.metadata().staleness_days, 1.0);

    // Both the start and end of the range fall inside the shifted span.
    let shifted_span = selection.ionex().map_epochs_s();
    assert!(shifted_span[0] <= start && end <= shifted_span[shifted_span.len() - 1]);
}

// ---------------------------------------------------------------------------
// SP3: present path is byte-identical; missing path is nearest-prior.
// ---------------------------------------------------------------------------

#[test]
fn sp3_present_path_is_byte_identical() {
    let day = make_sp3(2024, 3, 10);
    let span = day.epochs_j2000_seconds();
    let requested = span[0] + 300.0; // inside [00:00, 00:15]

    let direct = day
        .position_at_j2000_seconds(gps(1), requested)
        .expect("direct interp");

    let set = [day.clone()];
    let selection =
        select_sp3(&set, requested, StalenessPolicy::default()).expect("exact selection");
    assert_eq!(selection.metadata().kind, DegradationKind::Exact);
    assert_eq!(selection.metadata().staleness_s, 0.0);

    let via_layer = selection
        .position_at_j2000_seconds(gps(1), requested)
        .expect("layered interp");
    assert_eq!(
        via_layer.position.as_array().map(f64::to_bits),
        direct.position.as_array().map(f64::to_bits),
        "present SP3 position must be bit-identical to the direct call"
    );
    assert_eq!(
        via_layer.clock_s.map(f64::to_bits),
        direct.clock_s.map(f64::to_bits)
    );

    // The EphemerisSource impl delegates identically.
    let eph = selection.position_clock_at_j2000_s(gps(1), requested);
    let direct_eph = day.position_clock_at_j2000_s(gps(1), requested);
    assert_eq!(
        eph.map(|(p, c)| (p.map(f64::to_bits), c.to_bits())),
        direct_eph.map(|(p, c)| (p.map(f64::to_bits), c.to_bits()))
    );
}

#[test]
fn sp3_missing_day_selects_nearest_prior_with_correct_staleness() {
    // Days D and D+2 present; D+1 withheld. Request 00:07 on D+1.
    let day0 = make_sp3(2024, 3, 10);
    let day2 = make_sp3(2024, 3, 12);
    let d0_span = day0.epochs_j2000_seconds();
    let d0_last = d0_span[d0_span.len() - 1]; // D 00:15
    let requested = d0_span[0] + 86_400.0 + 7.0 * 60.0; // D+1 00:07

    let set = [day2, day0.clone()];
    let selection =
        select_sp3(&set, requested, StalenessPolicy::default()).expect("nearest-prior selection");
    let meta = selection.metadata();

    assert_eq!(meta.kind, DegradationKind::NearestPrior);
    assert_eq!(meta.source_epoch_j2000_s, d0_last);
    assert_eq!(meta.requested_epoch_j2000_s, requested);
    assert_eq!(meta.staleness_s, requested - d0_last);
    assert_eq!(meta.staleness_days, (requested - d0_last) / 86_400.0);

    // The selected product is the prior day (D), not the later one.
    assert_eq!(selection.sp3().epochs_j2000_seconds(), d0_span);
}

#[test]
fn sp3_beyond_cap_is_a_typed_error() {
    let day0 = make_sp3(2024, 3, 10);
    let span = day0.epochs_j2000_seconds();
    let requested = span[span.len() - 1] + 5.0 * 86_400.0; // 5 days past

    let set = [day0];
    let err =
        select_sp3(&set, requested, StalenessPolicy::default()).expect_err("exceeds 3-day cap");
    assert!(matches!(err, SelectionError::BeyondStalenessCap { .. }));

    // A wider cap admits the same request.
    let day0b = make_sp3(2024, 3, 10);
    let set2 = [day0b];
    let ok = select_sp3(&set2, requested, StalenessPolicy::days(7.0)).expect("within 7-day cap");
    assert_eq!(ok.metadata().kind, DegradationKind::NearestPrior);
}

#[test]
fn sp3_empty_set_is_a_typed_error() {
    let err = select_sp3(&[], 0.0, StalenessPolicy::default()).expect_err("empty set");
    assert_eq!(err, SelectionError::EmptyProductSet);
}

#[test]
fn sp3_invalid_range_is_a_typed_error() {
    let day0 = make_sp3(2024, 3, 10);
    let set = [day0];
    let err = select_sp3_over_range(&set, 100.0, 50.0, StalenessPolicy::default())
        .expect_err("end before start");
    assert!(matches!(err, SelectionError::InvalidRange { .. }));
}

// ---------------------------------------------------------------------------
// Regression: range-selection, shifted-coverage, and cap-validation holes.
// ---------------------------------------------------------------------------

#[test]
fn sp3_range_selects_product_covering_start_but_ending_before_end() {
    // A single product covers the range start but ends before the range end. It
    // is the nearest-prior source for the worst-case end and must be selected
    // (not rejected as if no prior product existed), with staleness `end - hi`.
    let day = make_sp3(2024, 3, 10);
    let span = day.epochs_j2000_seconds();
    let last = span[span.len() - 1]; // D 00:15
    let start = span[0] + 300.0; // D 00:05, inside the product
    let end = last + 900.0; // D 00:30, past the product's last epoch

    let set = [day.clone()];
    let selection = select_sp3_over_range(&set, start, end, StalenessPolicy::default())
        .expect("nearest-prior over range end");
    let meta = selection.metadata();

    assert_eq!(meta.kind, DegradationKind::NearestPrior);
    assert_eq!(meta.source_epoch_j2000_s, last);
    assert_eq!(meta.requested_epoch_j2000_s, end);
    assert_eq!(meta.staleness_s, end - last);
    assert_eq!(selection.sp3().epochs_j2000_seconds(), span);
}

#[test]
fn sp3_skips_a_prior_that_starts_after_the_range_start() {
    // `later` has the larger last epoch but begins after the range start, so it
    // cannot serve the start and must not be chosen on last-epoch alone. The
    // genuine prior that covers the start (`prior`) is the nearest-prior source.
    let prior = make_sp3_epochs(&[(10, 0, 0), (10, 0, 15)]); // [D10 00:00, D10 00:15]
    let later = make_sp3_epochs(&[(11, 0, 0), (11, 0, 10)]); // [D11 00:00, D11 00:10]
    let prior_span = prior.epochs_j2000_seconds();
    let later_span = later.epochs_j2000_seconds();
    let start = prior_span[0] + 300.0; // D10 00:05, inside `prior`, before `later`
    let end = later_span[later_span.len() - 1] + 1_200.0; // D11 00:30

    for set in [
        [prior.clone(), later.clone()],
        [later.clone(), prior.clone()],
    ] {
        let selection = select_sp3_over_range(&set, start, end, StalenessPolicy::default())
            .expect("the start-covering prior is selected");
        assert_eq!(selection.metadata().kind, DegradationKind::NearestPrior);
        assert_eq!(
            selection.sp3().epochs_j2000_seconds(),
            prior_span,
            "must skip the prior that begins after the range start"
        );
        assert_eq!(
            selection.metadata().source_epoch_j2000_s,
            prior_span[prior_span.len() - 1]
        );
    }
}

#[test]
fn ionex_partial_freshest_prior_does_not_mask_an_older_covering_prior() {
    // The freshest prior is narrow (22:00..23:00); shifted onto the request it
    // does not cover the range. An older, wide prior (00:00..20:00) does cover
    // once shifted. The freshest must not block the older covering candidate.
    let fresh_partial = make_ionex_days(2024, 3, &[(10, 22), (10, 23)], 50);
    let old_wide = make_ionex_days(2024, 3, &[(10, 0), (10, 20)], 70);
    let wide_span = old_wide.map_epochs_s();
    let one_day = 86_400i64;
    let start = wide_span[0] + one_day + 3_600; // D11 01:00
    let end = wide_span[0] + one_day + 19 * 3_600; // D11 19:00

    // The narrow prior alone cannot cover the range once shifted.
    let only_partial = [fresh_partial.clone()];
    let err = select_ionex_over_range(&only_partial, start, end, StalenessPolicy::default())
        .expect_err("narrow prior cannot cover");
    assert!(matches!(err, SelectionError::InvalidProduct(_)));

    // With the wide prior also present, it is selected regardless of slice order.
    for set in [
        [fresh_partial.clone(), old_wide.clone()],
        [old_wide.clone(), fresh_partial.clone()],
    ] {
        let selection = select_ionex_over_range(&set, start, end, StalenessPolicy::default())
            .expect("older wide prior covers after the shift");
        assert_eq!(selection.metadata().kind, DegradationKind::DiurnalShift);
        assert_eq!(selection.metadata().staleness_days, 1.0);
        assert_eq!(
            selection.ionex().tec_maps(),
            old_wide.tec_maps(),
            "must fall through to the older, wider covering prior"
        );
    }
}

#[test]
fn ionex_shifted_product_that_fails_to_cover_is_a_typed_error() {
    // Prior product spans only [00:00, 06:00]. A request whose end time-of-day is
    // past 06:00 forces a ceil whole-day shift that overshoots: the shifted grid
    // does not actually cover the requested range, so the request must be
    // declined rather than returning a non-covering grid.
    let prior = make_ionex(2024, 3, 10, 50); // maps at 00:00 and 06:00
    let span = prior.map_epochs_s();
    let one_day = 86_400i64;
    let start = span[0] + one_day + 2 * 3_600; // D+1 02:00
    let end = span[0] + one_day + 8 * 3_600; // D+1 08:00, past 06:00

    let set = [prior];
    let err = select_ionex_over_range(&set, start, end, StalenessPolicy::default())
        .expect_err("shifted grid does not cover the range");
    match err {
        SelectionError::InvalidProduct(msg) => {
            assert!(
                msg.contains("covers requested range"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected InvalidProduct, got {other:?}"),
    }
}

#[test]
fn ionex_non_finite_or_negative_cap_is_a_typed_error() {
    // A request that would degrade (10 days past the only product): a NaN, inf,
    // or negative cap must be rejected up front, never silently admitting the
    // beyond-cap data because `staleness_s > NaN` is false.
    let day0 = make_ionex(2024, 3, 10, 50);
    let span = day0.map_epochs_s();
    let requested = span[1] + 10 * 86_400;
    let set = [day0];

    for cap in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let err = select_ionex(&set, requested, StalenessPolicy::seconds(cap))
            .expect_err("non-finite/negative cap must be rejected");
        match err {
            SelectionError::InvalidPolicy { max_staleness_s } => {
                assert!(max_staleness_s.is_nan() || max_staleness_s == cap);
            }
            other => panic!("expected InvalidPolicy, got {other:?}"),
        }
    }
}

#[test]
fn sp3_non_finite_or_negative_cap_is_a_typed_error() {
    let day0 = make_sp3(2024, 3, 10);
    let span = day0.epochs_j2000_seconds();
    let requested = span[span.len() - 1] + 5.0 * 86_400.0;
    let set = [day0];

    for cap in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let err = select_sp3(&set, requested, StalenessPolicy::seconds(cap))
            .expect_err("non-finite/negative cap must be rejected");
        assert!(matches!(err, SelectionError::InvalidPolicy { .. }));
    }
}

#[test]
fn ionex_extreme_epoch_overflow_is_a_typed_error() {
    // An extreme requested epoch must decline with a typed overflow error, never
    // wrap the i64 J2000-second axis in release. The prior product predates the
    // J2000 epoch (negative `hi`), so the gap `end - hi` to `i64::MAX` overflows
    // outright.
    let prior = make_ionex_days(1999, 12, &[(30, 0), (30, 12)], 50);
    let span = prior.map_epochs_s();
    assert!(
        span[span.len() - 1] < 0,
        "prior must have a negative last epoch"
    );
    let set = [prior];
    let err = select_ionex_over_range(&set, i64::MAX, i64::MAX, StalenessPolicy::default())
        .expect_err("epoch arithmetic overflows");
    assert!(matches!(err, SelectionError::Overflow { .. }));
}

#[test]
fn ionex_exact_tie_break_is_deterministic() {
    // Two products both cover the requested range with different starts. The
    // freshest (latest start) wins, regardless of slice order.
    let earlier_start = make_ionex_days(2024, 3, &[(10, 0), (11, 6)], 50); // lo = D10 00:00
    let later_start = make_ionex_days(2024, 3, &[(11, 0), (11, 6)], 70); // lo = D11 00:00
    let span = later_start.map_epochs_s();
    let start = span[0] + 3_600; // D11 01:00
    let end = span[0] + 5 * 3_600; // D11 05:00

    for set in [
        [earlier_start.clone(), later_start.clone()],
        [later_start.clone(), earlier_start.clone()],
    ] {
        let selection = select_ionex_over_range(&set, start, end, StalenessPolicy::default())
            .expect("both products cover the range");
        assert_eq!(selection.metadata().kind, DegradationKind::Exact);
        assert_eq!(
            selection.ionex().tec_maps(),
            later_start.tec_maps(),
            "latest-start product must win regardless of slice order"
        );
    }

    // Same start, different last epoch: the tightest span (smallest hi) wins.
    let wide = make_ionex_days(2024, 3, &[(11, 0), (11, 12)], 80); // hi = D11 12:00
    let tight = make_ionex_days(2024, 3, &[(11, 0), (11, 6)], 90); // hi = D11 06:00
    for set in [[wide.clone(), tight.clone()], [tight.clone(), wide.clone()]] {
        let selection = select_ionex_over_range(&set, start, end, StalenessPolicy::default())
            .expect("both cover the range");
        assert_eq!(selection.ionex().tec_maps(), tight.tec_maps());
    }
}

#[test]
fn sp3_exact_tie_break_is_deterministic() {
    // Two products both cover the request; the freshest (latest start) wins,
    // independent of slice order.
    let earlier_start = make_sp3_epochs(&[(10, 0, 0), (11, 0, 15)]); // lo = D10 00:00
    let later_start = make_sp3_epochs(&[(11, 0, 0), (11, 0, 15)]); // lo = D11 00:00
    let later_span = later_start.epochs_j2000_seconds();
    let start = later_span[0] + 300.0; // D11 00:05
    let end = later_span[0] + 600.0; // D11 00:10

    for set in [
        [earlier_start.clone(), later_start.clone()],
        [later_start.clone(), earlier_start.clone()],
    ] {
        let selection = select_sp3_over_range(&set, start, end, StalenessPolicy::default())
            .expect("both products cover the range");
        assert_eq!(selection.metadata().kind, DegradationKind::Exact);
        assert_eq!(
            selection.sp3().epochs_j2000_seconds(),
            later_span,
            "latest-start product must win regardless of slice order"
        );
    }
}

/// A synthetic single-map IONEX product at one calendar instant (seconds
/// resolution), for placing a map epoch at a precise J2000 second.
fn make_ionex_at(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: i64,
    base_tec: i64,
) -> Ionex {
    let mut text = String::new();
    text.push_str(&ionex_record("1.0", "IONEX VERSION / TYPE"));
    text.push_str(&ionex_record("1", "# OF MAPS IN FILE"));
    text.push_str(&ionex_record("2.0 -2.0 -2.0", "LAT1 / LAT2 / DLAT"));
    text.push_str(&ionex_record("-2.0 2.0 2.0", "LON1 / LON2 / DLON"));
    text.push_str(&ionex_record("450.0 450.0 0.0", "HGT1 / HGT2 / DHGT"));
    text.push_str(&ionex_record("6371.0", "BASE RADIUS"));
    text.push_str(&ionex_record("0", "EXPONENT"));
    text.push_str(&ionex_record("", "END OF HEADER"));
    text.push_str(&ionex_record("1", "START OF TEC MAP"));
    text.push_str(&ionex_record(
        &format!("{year} {month} {day} {hour} {minute} {second}"),
        "EPOCH OF CURRENT MAP",
    ));
    for lat_idx in 0..3i64 {
        let lat = 2.0 - (lat_idx as f64) * 2.0;
        text.push_str(&ionex_record(
            &format!("{lat:.1} -2.0 2.0 2.0 450.0"),
            "LAT/LON1/LON2/DLON/H",
        ));
        let a = base_tec + lat_idx * 10 + 100;
        text.push_str(&format!("{} {} {}\n", a, a + 1, a + 2));
    }
    text.push_str(&ionex_record("1", "END OF TEC MAP"));
    Ionex::parse_str(&text).expect("valid synthetic IONEX")
}

#[test]
fn ionex_shift_overflow_of_freshest_prior_does_not_mask_an_older_covering_prior() {
    // The IONEX prior loop must not abort when a candidate's whole-day shift
    // overflows the i64 axis during construction: an older candidate can still
    // shift exactly onto the request and cover.
    //
    // With `end = i64::MAX` (`MAX % 86400 == 55807`), a prior whose last epoch is
    // at residue 55807 shifts to exactly `MAX` (no overflow, covers), while the
    // one-second-fresher prior at residue 55808 shifts to `MAX + 1` (overflow).
    // The freshest is tried first; its overflow must be skipped, not fatal.
    let day = 86_400i64;
    let covering = make_ionex_at(2000, 1, 2, 3, 30, 7, 50); // J2000 55807, MAX-aligned
    let fresher_overflow = make_ionex_at(2000, 1, 2, 3, 30, 8, 70); // J2000 55808
    assert_eq!(covering.map_epochs_s()[0].rem_euclid(day), 55_807);
    assert_eq!(fresher_overflow.map_epochs_s()[0].rem_euclid(day), 55_808);

    // The shifted staleness is astronomically large, so a cap that admits it is
    // required to reach the shift at all (otherwise it is cap-rejected first).
    let policy = StalenessPolicy::seconds(f64::MAX);
    for set in [
        [fresher_overflow.clone(), covering.clone()],
        [covering.clone(), fresher_overflow.clone()],
    ] {
        let selection = select_ionex_over_range(&set, i64::MAX, i64::MAX, policy)
            .expect("older prior shifts exactly onto the request");
        assert_eq!(selection.metadata().kind, DegradationKind::DiurnalShift);
        assert_eq!(
            selection.ionex().tec_maps(),
            covering.tec_maps(),
            "must skip the freshest prior whose shift overflows and use the covering one"
        );
        assert_eq!(selection.metadata().source_epoch_j2000_s, 55_807.0);
    }
}
