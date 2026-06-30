#![no_main]

mod compute_common;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::time::{
    eop, scales, Duration, GnssWeekTow, Instant, JulianDateSplit, Time, TimeScales, ValidityMode,
};

#[derive(Debug, Arbitrary)]
struct Input {
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    week: u32,
    rollovers: u32,
    nanos: i128,
    scale: u8,
    mode: bool,
    seconds: [f64; 8],
}

fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };

    let scale = time_scale(input.scale);
    assert_ok_or_err("Time::new", Time::new(input.seconds[0]));
    if let Ok(time) = Time::new(input.seconds[0]) {
        assert_success("Time::new::tdb", time.tdb());
    }

    let jd = JulianDateSplit::new(input.seconds[1], input.seconds[2]);
    assert_ok_finite_or_err("JulianDateSplit::new", jd);
    if let Ok(jd) = jd {
        assert_success("JulianDateSplit::to_jd", jd.to_jd());
        assert_success(
            "Instant::from_julian_date",
            Instant::from_julian_date(scale, jd),
        );
    }
    assert_success(
        "Instant::from_nanos",
        Instant::from_nanos(scale, input.nanos),
    );
    assert_ok_finite_or_err(
        "Duration::from_seconds",
        Duration::from_seconds(input.seconds[3]),
    );
    assert_success("Duration::from_nanos", Duration::from_nanos(input.nanos));

    let wt = GnssWeekTow::new(scale, input.week, input.seconds[4]);
    assert_ok_finite_or_err("GnssWeekTow::new", wt);
    if let Ok(wt) = wt {
        assert_ok_finite_or_err("GnssWeekTow::normalized", wt.normalized());
        assert_ok_or_err(
            "GnssWeekTow::unrolled_week",
            wt.unrolled_week(input.rollovers),
        );
    }

    assert_ok_finite_or_err(
        "TimeScales::from_utc",
        TimeScales::from_utc(
            input.year,
            input.month,
            input.day,
            input.hour,
            input.minute,
            input.seconds[5],
        ),
    );
    assert_ok_finite_or_err(
        "TimeScales::from_utc_validated",
        TimeScales::from_utc_validated(
            input.year,
            input.month,
            input.day,
            input.hour,
            input.minute,
            input.seconds[5],
            if input.mode {
                ValidityMode::Strict
            } else {
                ValidityMode::Permissive
            },
        ),
    );
    let _ = scales::julian_day_number(input.year, input.month, input.day);
    assert_success(
        "scales::find_leap_seconds",
        scales::find_leap_seconds(input.seconds[6]),
    );
    let table = scales::leap_second_table();
    let _ = (table.first_mjd, table.last_mjd, table.entries);
    let prov = scales::ut1_coverage();
    let _ = eop::check_ut1_coverage(
        &prov,
        input.seconds[7],
        if input.mode {
            ValidityMode::Strict
        } else {
            ValidityMode::Permissive
        },
    );
});
