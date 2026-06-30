#![cfg(sidereon_repo_tests)]
//! RTKLIB cross-implementation oracle gates.
//!
//! Fixture provenance (all under `tests/fixtures/rtk/`). RTKLIB `rnx2rtkp` output
//! is a cross-implementation reference, NOT a bit-match target.
//!
//! ## Wettzell WTZR/WTZZ static short-baseline (2020-06-25, 120 epochs)
//!
//! `wtzr_wtzz_rtklib_oracle.json`, `wtzr_wtzz_rtklib_precise_oracle.json`,
//! `wtzr_wtzz_rtklib_precise_epoch2_ambiguity.json`,
//! `wtzr_wtzz_multignss_static_rtklib_oracle.json`: RTKLIB `rnx2rtkp` 2.4.2 (local
//! checkout commit 71db0ff) over the vendored WTZR/WTZZ
//! `test/fixtures/obs/WTZ*00DEU_R_20201770000_01D_30S_MO_120epoch.rnx` arc. Truth is
//! the antenna-reference-point baseline from the marker ECEF + RINEX antenna-height
//! fields. Broadcast nav `nav/ESBC00DNK_R_20201770000_01D_MN.rnx`; Track B multi-GNSS
//! uses BKG `nav/BRDC00WRD_R_20201770000_01D_GREC.rnx` (so GLONASS is present;
//! GLONASS AR off, `pos2-gloarmode=off`). Primary cmd `rnx2rtkp -p 3 -f 1 -h -m 10`
//! (kinematic/RTK, L1, fix-and-hold, 10 deg mask). Precise cmd adds
//! `-k precise_rinexhead.conf -a` with `pos1-sateph=precise`, CODE orbit
//! `COD0MGXFIN_20201770000_01D_05M_ORB.SP3` + CNES/CLS clock
//! `GRG0MGXFIN_20201770000_01D_30S_CLK.CLK`; max per-epoch baseline delta vs
//! broadcast ~1.6 mm. The epoch-2 ambiguity fixture is an instrumented
//! `resamb_LAMBDA` DD trace. Multi-GNSS oracle fixes 120/120, ends 1.834629036 mm
//! truth error. Note: passing an SP3 to rnx2rtkp does NOT alone select precise
//! ephemeris (default `pos1-sateph=brdc`); RTKLIB 2.4.2 also fails uppercase `.SP3`
//! paths, so the precise fixture staged CODE SP3 as `cod.sp3` and GRG clock as
//! `grg.clk`.
//!
//! ## GSDC moving-rover arcs (RTKLIB Explorer demo5)
//!
//! `gsdc_*_pixel5_p222_demo5_rtklib_oracle.json`: `rnx2rtkp RTKLIB EX 2.5.0` (demo5
//! commit 57d39e7, LDLIBS="-lm") over four GSDC 2021 GooglePixel5 drives from the
//! Kaggle smartphone-decimeter-2022 corpus (zip sha256
//! eee3bc1c5d7414e97dca5929e0960b6be2f93d0702dac53fe2602b55ae4e4ca3). Drives:
//! SVL1 2021-08-24 (suburban, vendored baseline), SJC1 2021-08-04, MTV1 2021-12-15,
//! MTV1 2021-12-28; selected by metadata only, solver output is not a selection
//! criterion. Rover obs `supplemental/gnss_rinex.21o` (1 Hz phone), truth
//! `ground_truth.csv` -> WGS84 ECEF aligned to GPST with GPS-UTC 18 s. Base is NOAA
//! CORS P222 (nearest operational P-class station, ARP ECEF
//! -2689639.5060,-4290438.6360,3865050.9560 m), obs from
//! `https://geodesy.noaa.gov/corsdata/rinex/2021/<doy>/p222/...`. Nav is BKG IGS
//! combined broadcast `BRDC00WRD_R_<doy>_01D_MN.rnx` from
//! `https://igs.bkg.bund.de/.../BRDC/...` (so phone Galileo+BeiDou are usable;
//! NOAA GPS/GLONASS-only nav fixed 0-1 epochs and was rejected). Config:
//! `pos1-posmode=kinematic`, `pos1-soltype=combined`, `pos1-frequency=l1`,
//! `pos1-navsys=45` (G+R+E+C), `misc-timeinterp=on`, `ant1-postype=single`
//! (phone approx pos all zeros), `ant2-postype=rinexhead`, `pos2-gloarmode=off`,
//! `pos2-arthres*=3.0` (RTKLIB default ratio gate; a prior `=1.0` artifact was
//! rejected for effectively disabling ratio validation). These are calibrated
//! meter-class phone trajectory references, not cm-grade; fixed status is not a
//! confidence target (it does not consistently beat float on 3D median). Per-arc
//! checksums (extracted rover gnss_rinex.21o / ground_truth.csv / committed oracle
//! JSON):
//!   * SVL1 2021-08-24: fec1eb49…06317 / 10edb68f…537be / c0112f7c…1a726e
//!   * SJC1 2021-08-04: 0c7c5d62…b9075 / 5403afdf…24c11c1 / 67d4423f…ea88ca
//!   * MTV1 2021-12-15: 60a497c6…27a854 / dea81f2a…4247a / 01a98b22…e26528
//!   * MTV1 2021-12-28: 398e497a…cc8638 / 149069d0…292f153 / d649a423…feba4dc65
//!
//! The three added oracles use `--truth-time-tolerance-ms 2` for truth lookup only
//! (matching demo5 rounded-ms output to GSDC truth rows), not the RTKLIB solve.
//! GSDC Kaggle data is not redistributable: only derived oracle JSON + config +
//! generator + this provenance are vendored; raw GSDC, decoded CORS, and full nav
//! products remain recipe-only under /tmp/gsdc-work.
//!
//! ## C+D Phase 1 static EPN short baseline (2026-04-30, DOY 120)
//!
//! `pasa_scoa_2026_120_l1_static_fixhold_rtklib_oracle.json` and
//! `pasa_scoa_2026_120_l1l2_static_rtklib_oracle.json`: RTKLIB v2.4.2-p13 (commit
//! 71db0ffa0d9735697c6adfd06fdf766d0e5ce807, built with -O3 -std=gnu89 -DTRACE
//! -DENAGLO -DENAQZS -DENAGAL -DNFREQ=3) over PASA00ESP rover vs SCOA00FRA base
//! (Pasaia ES / Ciboure FR, 21.836327792 km, different receiver antennas), 10:00-12:00
//! GPST. Selected from EPN candidates for 15-40 km baseline, same-day public data,
//! ANTEX-listed active antennas, zero marker-to-ARP eccentricity. Source obs BKG
//! EUREF CRINEX `https://igs.bkg.bund.de/root_ftp/EUREF/obs/2026/120/{PASA00ESP,SCOA00FRA}_R_20261200000_01D_30S_MO.crx.gz`
//! (sha256 f749babd…5ebc48 / ef01c8f4…b5f9ca); nav BKG IGS BRDC
//! `BRDC00WRD_R_20261200000_01D_MN.rnx.gz` (sha256 1325a273…25b7f7); precise IGS
//! final IGS0OPSFIN SP3+CLK for week 2416 (GPS-only, so the precise solves set
//! `pos1-navsys=1`; SP3 gz sha256 c06164f3…22a802, CLK gz sha256 8483e969…ccdbe6);
//! IGS20 ANTEX (sha256 70e963f6…ce9699), EPN C2385 SSC, station logs. Truth from
//! the EPN C2385 ITRF2020 solution propagated from epoch 2020-01-01 to 2026-04-30
//! 11:00 GPST: SCOA base ARP 4639940.429559,-136224.811560,4359552.502780 m, PASA
//! rover ARP 4644908.987020,-156644.937061,4353623.158575 m. L1 static fix-and-hold
//! (`pos2-armode=fix-and-hold`, `pos2-arthres=3.0`, `pos1-tidecorr=on`,
//! `pos1-ionoopt=brdc`, `pos1-tropopt=saas`, receiver-PCV on) fixes 171/240, final
//! truth error 0.0524 m; L1/L2 static continuous fixes 80/240, final 0.0581 m.
//! RTKLIB's precise path loads staged lowercase `igs_fin.sp3`/`igs_fin.clk`.
//! Committed trimmed inputs (obs/nav/sp3/clk/antex/truth) carry their own sha256 in
//! the generator `test/fixtures/rtk/generators/cd_phase1_pasa_scoa_2026_120.py`.

use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy)]
struct GsdcArc {
    fixture: &'static str,
    config: &'static str,
    source_pos: &'static str,
    label: &'static str,
    description: &'static str,
    base_distance_km: f64,
    truth_time_tolerance_ms: Option<i64>,
    epochs: usize,
    fixed_epochs: usize,
    q_counts: &'static [(&'static str, i64)],
    first_fixed_index: usize,
    first_fixed_time: &'static str,
    first_fixed_truth_time_utc: &'static str,
    final_status: &'static str,
    fix_rate: f64,
    error_3d_median: f64,
    error_3d_p95: f64,
    horizontal_p95: f64,
    first_time: &'static str,
    first_truth_time_utc: &'static str,
    last_time: &'static str,
    fixed_3d_median: f64,
    fixed_3d_p95: f64,
    float_3d_median: f64,
    float_3d_p95: f64,
    fixed_beats_float: bool,
}

#[derive(Debug, Clone, Copy)]
struct CdArc {
    fixture: &'static str,
    config: &'static str,
    source_pos: &'static str,
    label: &'static str,
    description: &'static str,
    epochs: usize,
    fixed_epochs: usize,
    q_counts: &'static [(&'static str, i64)],
    fix_rate: f64,
    first_fixed_index: usize,
    first_fixed_time: &'static str,
    final_status: &'static str,
    final_ratio: f64,
    final_truth_error_m: f64,
    mean_truth_error_m: f64,
    max_truth_error_m: f64,
    first_time: &'static str,
    last_time: &'static str,
    satellites_min: i64,
    satellites_max: i64,
}

const GSDC_ORACLES: &[GsdcArc] = &[
    GsdcArc {
        fixture: "gsdc_2021_08_04_sjc1_pixel5_p222_demo5_rtklib_oracle.json",
        config: "track_a_gsdc_2021_08_04_sjc1_p222_grec_l1.conf",
        source_pos: "track_a_gsdc_2021_08_04_sjc1_p222_grec_l1.pos",
        label: "gsdc_2021_08_04_sjc1_pixel5_p222_grec_l1_demo5",
        description: "RTKLIB demo5 moving-rover oracle for GSDC 2022 train/2021-08-04-US-SJC-1/GooglePixel5 against NOAA CORS P222 (G/R/E/C L1, combined, fix-and-hold, AR ratio gate 3.0). The 10 fixed epochs do not beat float on 3D median error, so fixed status is not a confidence target; the oracle is a calibrated trajectory accuracy reference.",
        base_distance_km: 27.403,
        truth_time_tolerance_ms: Some(2),
        epochs: 1554,
        fixed_epochs: 10,
        q_counts: &[("1", 10), ("2", 1538), ("4", 6)],
        first_fixed_index: 85,
        first_fixed_time: "2021-08-04T20:42:08.449",
        first_fixed_truth_time_utc: "2021-08-04T20:41:50.449",
        final_status: "float",
        fix_rate: 0.006435006435,
        error_3d_median: 4.5221765,
        error_3d_p95: 12.296687,
        horizontal_p95: 6.688258,
        first_time: "2021-08-04T20:40:43.449",
        first_truth_time_utc: "2021-08-04T20:40:25.449",
        last_time: "2021-08-04T21:06:36.450",
        fixed_3d_median: 5.044745,
        fixed_3d_p95: 7.332008,
        float_3d_median: 4.516922,
        float_3d_p95: 12.296687,
        fixed_beats_float: false,
    },
    GsdcArc {
        fixture: "gsdc_2021_08_24_svl1_pixel5_p222_demo5_rtklib_oracle.json",
        config: "track_a_gsdc_p222_grec_l1.conf",
        source_pos: "track_a_gsdc_p222_grec_l1.pos",
        label: "gsdc_svl1_pixel5_p222_grec_l1_demo5",
        description: "RTKLIB demo5 moving-rover oracle for GSDC 2022 train/2021-08-24-US-SVL-1/GooglePixel5 against NOAA CORS P222 (G/R/E/C L1, combined, fix-and-hold, AR ratio gate 3.0). Validated fixes on this phone arc are meter-class, not cm-class; the oracle is a calibrated trajectory accuracy reference, not a fix-rate target.",
        base_distance_km: 18.936,
        truth_time_tolerance_ms: None,
        epochs: 3136,
        fixed_epochs: 10,
        q_counts: &[("1", 10), ("2", 3104), ("4", 22)],
        first_fixed_index: 14,
        first_fixed_time: "2021-08-24T20:33:14.437",
        first_fixed_truth_time_utc: "2021-08-24T20:32:56.437",
        final_status: "float",
        fix_rate: 0.00318877551,
        error_3d_median: 3.9769565,
        error_3d_p95: 8.775371,
        horizontal_p95: 6.034325,
        first_time: "2021-08-24T20:33:00.437",
        first_truth_time_utc: "2021-08-24T20:32:42.437",
        last_time: "2021-08-24T21:25:16.437",
        fixed_3d_median: 3.476352,
        fixed_3d_p95: 4.562249,
        float_3d_median: 3.964981,
        float_3d_p95: 8.579962,
        fixed_beats_float: true,
    },
    GsdcArc {
        fixture: "gsdc_2021_12_15_mtv1_pixel5_p222_demo5_rtklib_oracle.json",
        config: "track_a_gsdc_2021_12_15_mtv1_p222_grec_l1.conf",
        source_pos: "track_a_gsdc_2021_12_15_mtv1_p222_grec_l1.pos",
        label: "gsdc_2021_12_15_mtv1_pixel5_p222_grec_l1_demo5",
        description: "RTKLIB demo5 moving-rover oracle for GSDC 2022 train/2021-12-15-US-MTV-1/GooglePixel5 against NOAA CORS P222 (G/R/E/C L1, combined, fix-and-hold, AR ratio gate 3.0). This highway phone arc has only one fixed epoch; the split is underpowered and meter-class, so the oracle is a calibrated trajectory accuracy reference, not a fix-rate target.",
        base_distance_km: 13.815,
        truth_time_tolerance_ms: Some(2),
        epochs: 1465,
        fixed_epochs: 1,
        q_counts: &[("1", 1), ("2", 1436), ("4", 28)],
        first_fixed_index: 1312,
        first_fixed_time: "2021-12-15T19:11:05.438",
        first_fixed_truth_time_utc: "2021-12-15T19:10:47.438",
        final_status: "float",
        fix_rate: 0.000682593857,
        error_3d_median: 3.652537,
        error_3d_p95: 7.909147,
        horizontal_p95: 4.668259,
        first_time: "2021-12-15T18:49:11.438",
        first_truth_time_utc: "2021-12-15T18:48:53.438",
        last_time: "2021-12-15T19:13:37.438",
        fixed_3d_median: 3.022934,
        fixed_3d_p95: 3.022934,
        float_3d_median: 3.633223,
        float_3d_p95: 7.466983,
        fixed_beats_float: true,
    },
    GsdcArc {
        fixture: "gsdc_2021_12_28_mtv1_pixel5_p222_demo5_rtklib_oracle.json",
        config: "track_a_gsdc_2021_12_28_mtv1_p222_grec_l1.conf",
        source_pos: "track_a_gsdc_2021_12_28_mtv1_p222_grec_l1.pos",
        label: "gsdc_2021_12_28_mtv1_pixel5_p222_grec_l1_demo5",
        description: "RTKLIB demo5 moving-rover oracle for GSDC 2022 train/2021-12-28-US-MTV-1/GooglePixel5 against NOAA CORS P222 (G/R/E/C L1, combined, fix-and-hold, AR ratio gate 3.0). The fixed split beats float on this repeat highway route, but remains meter-class; the oracle is a calibrated trajectory accuracy reference, not a fix-rate target.",
        base_distance_km: 13.702,
        truth_time_tolerance_ms: Some(2),
        epochs: 1610,
        fixed_epochs: 10,
        q_counts: &[("1", 10), ("2", 1567), ("4", 33)],
        first_fixed_index: 830,
        first_fixed_time: "2021-12-28T20:31:18.438",
        first_fixed_truth_time_utc: "2021-12-28T20:31:00.437",
        final_status: "float",
        fix_rate: 0.006211180124,
        error_3d_median: 3.973879,
        error_3d_p95: 9.033375,
        horizontal_p95: 6.674816,
        first_time: "2021-12-28T20:17:25.438",
        first_truth_time_utc: "2021-12-28T20:17:07.437",
        last_time: "2021-12-28T20:44:17.438",
        fixed_3d_median: 2.642458,
        fixed_3d_p95: 3.382031,
        float_3d_median: 3.971948,
        float_3d_p95: 8.698615,
        fixed_beats_float: true,
    },
];

const CD_ORACLES: &[CdArc] = &[
    CdArc {
        fixture: "pasa_scoa_2026_120_l1_static_fixhold_rtklib_oracle.json",
        config: "cd_pasa_scoa_l1_static_fixhold.conf",
        source_pos: "cd_pasa_scoa_l1_static_fixhold.pos",
        label: "cd_pasa_scoa_2026_120_l1_static_fixhold",
        description: "RTKLIB 2.4.2-p13 C+D Phase 1 precise-GPS oracle for PASA00ESP rover against SCOA00FRA base on 2026-04-30 10:00-12:00 GPST (L1 static, fix-and-hold, AR ratio gate 3.0, ANTEX receiver PCV and solid earth tides enabled).",
        epochs: 240,
        fixed_epochs: 171,
        q_counts: &[("1", 171), ("2", 69)],
        fix_rate: 0.7125,
        first_fixed_index: 2,
        first_fixed_time: "2026-04-30T10:01:00",
        final_status: "fixed",
        final_ratio: 999.9,
        final_truth_error_m: 0.052353514434,
        mean_truth_error_m: 0.107036863232,
        max_truth_error_m: 0.375208123609,
        first_time: "2026-04-30T10:00:00",
        last_time: "2026-04-30T11:59:30",
        satellites_min: 4,
        satellites_max: 5,
    },
    CdArc {
        fixture: "pasa_scoa_2026_120_l1l2_static_rtklib_oracle.json",
        config: "cd_pasa_scoa_l1l2_static.conf",
        source_pos: "cd_pasa_scoa_l1l2_static.pos",
        label: "cd_pasa_scoa_2026_120_l1l2_static",
        description: "RTKLIB 2.4.2-p13 C+D Phase 1 precise-GPS oracle for PASA00ESP rover against SCOA00FRA base on 2026-04-30 10:00-12:00 GPST (dual-frequency static, continuous AR, AR ratio gate 3.0, ANTEX receiver PCV and solid earth tides enabled).",
        epochs: 240,
        fixed_epochs: 80,
        q_counts: &[("1", 80), ("2", 160)],
        fix_rate: 0.333333333333,
        first_fixed_index: 2,
        first_fixed_time: "2026-04-30T10:01:00",
        final_status: "float",
        final_ratio: 1.5,
        final_truth_error_m: 0.058098081566,
        mean_truth_error_m: 0.208126085588,
        max_truth_error_m: 0.980812363794,
        first_time: "2026-04-30T10:00:00",
        last_time: "2026-04-30T11:59:30",
        satellites_min: 4,
        satellites_max: 5,
    },
];

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rtk")
        .join(name)
}

fn load_oracle(name: &str) -> Value {
    let path = fixture_path(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read RTKLIB oracle fixture {path:?}: {err}"));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("parse {path:?}: {err}"))
}

fn as_array<'a>(value: &'a Value, label: &str) -> &'a [Value] {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{label} is not an array"))
}

fn as_str<'a>(value: &'a Value, label: &str) -> &'a str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("{label} is not a string"))
}

fn as_i64(value: &Value, label: &str) -> i64 {
    value
        .as_i64()
        .unwrap_or_else(|| panic!("{label} is not an integer"))
}

fn as_usize(value: &Value, label: &str) -> usize {
    usize::try_from(as_i64(value, label)).unwrap_or_else(|_| panic!("{label} is negative"))
}

fn as_f64(value: &Value, label: &str) -> f64 {
    value
        .as_f64()
        .unwrap_or_else(|| panic!("{label} is not a float"))
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
    );
}

fn assert_q_counts(actual: &Value, expected: &[(&str, i64)], label: &str) {
    let object = actual
        .as_object()
        .unwrap_or_else(|| panic!("{label} is not an object"));
    assert_eq!(object.len(), expected.len(), "{label} key count");
    for (key, count) in expected {
        assert_eq!(
            object.get(*key).and_then(Value::as_i64),
            Some(*count),
            "{label}.{key}"
        );
    }
}

fn count_q(epochs: &[Value], q: i64) -> usize {
    epochs
        .iter()
        .filter(|epoch| epoch["q"].as_i64() == Some(q))
        .count()
}

fn mode_by_label<'a>(modes: &'a [Value], label: &str) -> &'a Value {
    modes
        .iter()
        .find(|mode| mode["label"].as_str() == Some(label))
        .unwrap_or_else(|| panic!("missing comparison mode {label}"))
}

fn status_error_values(epochs: &[Value], q: i64, key: &str) -> Vec<f64> {
    epochs
        .iter()
        .filter(|epoch| epoch["q"].as_i64() == Some(q))
        .map(|epoch| as_f64(&epoch[key], key))
        .collect()
}

fn median(values: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let count = ordered.len();
    let mid = count / 2;

    if count % 2 == 1 {
        ordered[mid]
    } else {
        (ordered[mid - 1] + ordered[mid]) / 2.0
    }
}

fn percentile(values: &[f64], pct: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let index = (pct * (ordered.len() - 1) as f64).trunc() as usize;
    ordered[index]
}

fn fixed_split_beats_float(fixed_3d: &[f64], float_3d: &[f64]) -> bool {
    median(fixed_3d) < median(float_3d) && percentile(fixed_3d, 0.95) < percentile(float_3d, 0.95)
}

#[test]
fn wettzell_broadcast_oracle_pins_l1_reference_target() {
    let oracle = load_oracle("wtzr_wtzz_rtklib_oracle.json");
    assert_eq!(oracle["version"], json!(1));

    let reference = &oracle["reference"];
    let epochs = as_array(&oracle["per_epoch"], "per_epoch");

    assert_eq!(
        as_str(&reference["label"], "reference.label"),
        "l1_brdc_fix_and_hold"
    );
    assert_eq!(as_usize(&reference["epochs"], "reference.epochs"), 120);
    assert_eq!(
        as_usize(&reference["fixed_epochs"], "reference.fixed_epochs"),
        119
    );
    assert_eq!(
        as_usize(
            &reference["first_fixed_index"],
            "reference.first_fixed_index"
        ),
        1
    );
    assert_eq!(
        as_str(&reference["first_fixed_time"], "reference.first_fixed_time"),
        "2020-06-25T00:00:30"
    );
    assert_eq!(
        as_str(&reference["final_status"], "reference.final_status"),
        "fixed"
    );
    assert!(
        as_f64(
            &reference["final_truth_error_m"],
            "reference.final_truth_error_m"
        ) < 0.004
    );

    assert_eq!(epochs.len(), 120);
    assert_eq!(
        as_str(&epochs[0]["fix_status"], "epoch0.fix_status"),
        "float"
    );
    assert_eq!(
        as_str(&epochs[1]["fix_status"], "epoch1.fix_status"),
        "fixed"
    );
    assert_eq!(
        count_q(epochs, 1),
        as_usize(&reference["fixed_epochs"], "fixed_epochs")
    );

    let final_epoch = epochs.last().expect("final epoch");
    assert_eq!(
        final_epoch["baseline_enu_m"],
        reference["final_baseline_enu_m"]
    );
    assert_eq!(final_epoch["ratio"], reference["final_ratio"]);

    let modes = as_array(&oracle["comparison_modes"], "comparison_modes");
    let instantaneous = mode_by_label(modes, "l1_instantaneous");
    assert_eq!(
        as_usize(
            &instantaneous["fixed_epochs"],
            "l1_instantaneous.fixed_epochs"
        ),
        93
    );
    assert_eq!(
        as_str(
            &instantaneous["first_fixed_time"],
            "l1_instantaneous.first_fixed_time"
        ),
        "2020-06-25T00:02:00"
    );
    assert_eq!(
        as_usize(
            &mode_by_label(modes, "l1_float")["fixed_epochs"],
            "l1_float.fixed_epochs"
        ),
        0
    );
    assert_eq!(
        as_usize(
            &mode_by_label(modes, "l1_l2_brdc_fix_and_hold")["fixed_epochs"],
            "l1_l2_brdc_fix_and_hold.fixed_epochs"
        ),
        120
    );
}

#[test]
fn wettzell_precise_oracle_pins_lowercase_sp3_precise_target() {
    let oracle = load_oracle("wtzr_wtzz_rtklib_precise_oracle.json");
    assert_eq!(oracle["version"], json!(1));

    let reference = &oracle["reference"];
    let epochs = as_array(&oracle["per_epoch"], "per_epoch");

    assert_eq!(
        as_str(&reference["label"], "reference.label"),
        "l1_precise_cod_sp3_grg_clk_fix_and_hold"
    );
    assert_eq!(as_usize(&reference["epochs"], "reference.epochs"), 120);
    assert_eq!(
        as_usize(&reference["fixed_epochs"], "reference.fixed_epochs"),
        119
    );
    assert_eq!(
        as_usize(
            &reference["first_fixed_index"],
            "reference.first_fixed_index"
        ),
        1
    );
    assert_eq!(
        as_str(&reference["first_fixed_time"], "reference.first_fixed_time"),
        "2020-06-25T00:00:30"
    );
    assert_eq!(
        as_str(&reference["final_status"], "reference.final_status"),
        "fixed"
    );
    assert!(
        as_f64(
            &reference["final_truth_error_m"],
            "reference.final_truth_error_m"
        ) < 0.004
    );

    assert_eq!(
        as_str(
            &oracle["precise_products"]["orbit"],
            "precise_products.orbit"
        ),
        "COD0MGXFIN_20201770000_01D_05M_ORB.SP3"
    );
    assert_eq!(
        as_str(
            &oracle["precise_products"]["clock"],
            "precise_products.clock"
        ),
        "GRG0MGXFIN_20201770000_01D_30S_CLK.CLK"
    );
    assert_eq!(
        as_str(
            &oracle["precise_products"]["staged_orbit_path"],
            "precise_products.staged_orbit_path"
        ),
        "cod.sp3"
    );

    assert_eq!(epochs.len(), 120);
    assert_eq!(
        as_str(&epochs[0]["fix_status"], "epoch0.fix_status"),
        "float"
    );
    assert_eq!(
        as_str(&epochs[1]["fix_status"], "epoch1.fix_status"),
        "fixed"
    );
    assert_eq!(
        count_q(epochs, 1),
        as_usize(&reference["fixed_epochs"], "fixed_epochs")
    );

    let final_epoch = epochs.last().expect("final epoch");
    assert_eq!(
        final_epoch["baseline_enu_m"],
        reference["final_baseline_enu_m"]
    );
    assert_eq!(final_epoch["ratio"], reference["final_ratio"]);

    let comparison = &oracle["broadcast_comparison"];
    assert_eq!(
        as_str(
            &comparison["source_fixture"],
            "broadcast_comparison.source_fixture"
        ),
        "wtzr_wtzz_rtklib_oracle.json"
    );
    assert!(comparison["same_fix_status_by_epoch"]
        .as_bool()
        .expect("broadcast_comparison.same_fix_status_by_epoch"));
    assert!(
        as_f64(
            &comparison["max_baseline_delta_m"],
            "broadcast_comparison.max_baseline_delta_m"
        ) < 0.002
    );
}

#[test]
fn gsdc_demo5_oracles_pin_moving_rover_references() {
    for arc in GSDC_ORACLES {
        let oracle = load_oracle(arc.fixture);
        assert_eq!(oracle["version"], json!(1), "{} version", arc.fixture);
        assert_eq!(
            as_str(&oracle["description"], "description"),
            arc.description,
            "{} description",
            arc.fixture
        );

        assert_eq!(
            oracle["generator"]["rtklib"],
            json!({
                "program": "rnx2rtkp",
                "version": "EX 2.5.0",
                "commit": "57d39e7",
            }),
            "{} generator",
            arc.fixture
        );

        assert_eq!(
            as_str(
                &oracle["truth"]["base_station"]["id"],
                "truth.base_station.id"
            ),
            "P222"
        );
        assert_eq!(
            as_i64(
                &oracle["truth"]["gps_utc_offset_s"],
                "truth.gps_utc_offset_s"
            ),
            18
        );
        assert_close(
            as_f64(
                &oracle["truth"]["base_station"]["distance_from_drive_start_km"],
                "truth.base_station.distance_from_drive_start_km",
            ),
            arc.base_distance_km,
            0.0,
            arc.fixture,
        );

        match arc.truth_time_tolerance_ms {
            Some(tolerance_ms) => assert_eq!(
                as_i64(
                    &oracle["truth"]["time_match_tolerance_ms"],
                    "truth.time_match_tolerance_ms"
                ),
                tolerance_ms,
                "{} truth time tolerance",
                arc.fixture
            ),
            None => assert!(
                !oracle["truth"]
                    .as_object()
                    .expect("truth object")
                    .contains_key("time_match_tolerance_ms"),
                "{} unexpected truth time tolerance",
                arc.fixture
            ),
        }

        let reference = &oracle["reference"];
        let epochs = as_array(&oracle["per_epoch"], "per_epoch");

        assert_eq!(as_str(&reference["label"], "reference.label"), arc.label);
        assert_eq!(as_str(&reference["config"], "reference.config"), arc.config);
        assert_eq!(
            as_str(&reference["source_pos"], "reference.source_pos"),
            arc.source_pos
        );
        assert_eq!(
            as_usize(&reference["epochs"], "reference.epochs"),
            arc.epochs
        );
        assert_eq!(
            as_usize(&reference["fixed_epochs"], "reference.fixed_epochs"),
            arc.fixed_epochs
        );
        assert_q_counts(&reference["q_counts"], arc.q_counts, "reference.q_counts");
        assert_eq!(
            as_usize(
                &reference["first_fixed_index"],
                "reference.first_fixed_index"
            ),
            arc.first_fixed_index
        );
        assert_eq!(
            as_str(&reference["first_fixed_time"], "reference.first_fixed_time"),
            arc.first_fixed_time
        );
        assert_eq!(
            as_str(
                &reference["first_fixed_truth_time_utc"],
                "reference.first_fixed_truth_time_utc"
            ),
            arc.first_fixed_truth_time_utc
        );
        assert_eq!(
            as_str(&reference["final_status"], "reference.final_status"),
            arc.final_status
        );

        assert_close(
            as_f64(&reference["fix_rate"], "reference.fix_rate"),
            arc.fix_rate,
            1.0e-12,
            "fix_rate",
        );
        assert_close(
            as_f64(
                &reference["error_3d"]["median_m"],
                "reference.error_3d.median_m",
            ),
            arc.error_3d_median,
            1.0e-6,
            "error_3d median",
        );
        assert_close(
            as_f64(&reference["error_3d"]["p95_m"], "reference.error_3d.p95_m"),
            arc.error_3d_p95,
            1.0e-6,
            "error_3d p95",
        );
        assert_close(
            as_f64(
                &reference["horizontal_error"]["p95_m"],
                "reference.horizontal_error.p95_m",
            ),
            arc.horizontal_p95,
            1.0e-6,
            "horizontal p95",
        );

        assert_eq!(epochs.len(), arc.epochs);
        assert_eq!(as_str(&epochs[0]["time"], "first time"), arc.first_time);
        assert_eq!(
            as_str(&epochs[0]["truth_time_utc"], "first truth time"),
            arc.first_truth_time_utc
        );
        assert_eq!(
            as_str(&epochs.last().expect("last epoch")["time"], "last time"),
            arc.last_time
        );
        assert_eq!(count_q(epochs, 1), arc.fixed_epochs);

        let fixed = &epochs[arc.first_fixed_index];
        assert_eq!(as_str(&fixed["fix_status"], "fixed.fix_status"), "fixed");
        assert!(as_i64(&fixed["satellites"], "fixed.satellites") >= 4);
        assert!(as_f64(&fixed["ratio"], "fixed.ratio") >= 3.0);

        let fixed_3d = status_error_values(epochs, 1, "error_3d_m");
        let float_3d = status_error_values(epochs, 2, "error_3d_m");

        assert_close(
            median(&fixed_3d),
            arc.fixed_3d_median,
            1.0e-6,
            "fixed median",
        );
        assert_close(
            percentile(&fixed_3d, 0.95),
            arc.fixed_3d_p95,
            1.0e-6,
            "fixed p95",
        );
        assert_close(
            median(&float_3d),
            arc.float_3d_median,
            1.0e-6,
            "float median",
        );
        assert_close(
            percentile(&float_3d, 0.95),
            arc.float_3d_p95,
            1.0e-6,
            "float p95",
        );
        assert_eq!(
            fixed_split_beats_float(&fixed_3d, &float_3d),
            arc.fixed_beats_float,
            "{} fixed split comparison",
            arc.fixture
        );
    }
}

#[test]
fn pasa_scoa_cd_oracles_pin_epn_references() {
    for arc in CD_ORACLES {
        let oracle = load_oracle(arc.fixture);
        assert_eq!(oracle["version"], json!("1"), "{} version", arc.fixture);
        assert_eq!(
            as_str(&oracle["description"], "description"),
            arc.description,
            "{} description",
            arc.fixture
        );

        assert_eq!(
            oracle["generator"]["rtklib"],
            json!({
                "program": "rnx2rtkp",
                "version": "v2.4.2-p13",
                "commit": "71db0ff",
            }),
            "{} generator",
            arc.fixture
        );

        assert_eq!(
            as_str(&oracle["truth"]["frame"], "truth.frame"),
            "ITRF2020 ECEF metres propagated to 2026-04-30T11:00:00 GPST; ENU baseline at SCOA00FRA ARP, metres"
        );
        assert_eq!(
            as_str(
                &oracle["truth"]["base_station"]["id"],
                "truth.base_station.id"
            ),
            "SCOA00FRA"
        );
        assert_eq!(
            as_str(
                &oracle["truth"]["rover_station"]["id"],
                "truth.rover_station.id"
            ),
            "PASA00ESP"
        );
        assert_close(
            as_f64(
                &oracle["truth"]["baseline_length_km"],
                "truth.baseline_length_km",
            ),
            21.836327792,
            0.0,
            "baseline length",
        );

        let truth = &oracle["truth"]["antenna_baseline_enu_m"];
        assert_close(
            as_f64(&truth["east"], "truth.antenna_baseline_enu_m.east"),
            -20_265.520_602_760_01,
            1.0e-12,
            "truth east",
        );
        assert_close(
            as_f64(&truth["north"], "truth.antenna_baseline_enu_m.north"),
            -8_132.221_127_240_546,
            1.0e-12,
            "truth north",
        );
        assert_close(
            as_f64(&truth["up"], "truth.antenna_baseline_enu_m.up"),
            -29.422_321_279_917,
            1.0e-12,
            "truth up",
        );

        assert_eq!(
            oracle["inputs"],
            json!({
                "rover_obs": "test/fixtures/obs/PASA00ESP_R_20261201000_02H_30S_MO.rnx",
                "base_obs": "test/fixtures/obs/SCOA00FRA_R_20261201000_02H_30S_MO.rnx",
                "nav": "test/fixtures/nav/BRDC00WRD_R_20261200800_06H_MN.rnx",
                "sp3": "test/fixtures/sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3",
                "clk": "test/fixtures/clk/IGS0OPSFIN_2026120095930_02H01M_30S_CLK.CLK",
                "antex": "test/fixtures/antex/igs20_pasa_scoa_gps.atx",
            }),
            "{} inputs",
            arc.fixture
        );

        let reference = &oracle["reference"];
        let epochs = as_array(&oracle["per_epoch"], "per_epoch");

        assert_eq!(as_str(&reference["label"], "reference.label"), arc.label);
        assert_eq!(as_str(&reference["config"], "reference.config"), arc.config);
        assert_eq!(
            as_str(&reference["source_pos"], "reference.source_pos"),
            arc.source_pos
        );
        assert_eq!(
            as_usize(&reference["epochs"], "reference.epochs"),
            arc.epochs
        );
        assert_eq!(
            as_usize(&reference["fixed_epochs"], "reference.fixed_epochs"),
            arc.fixed_epochs
        );
        assert_q_counts(&reference["q_counts"], arc.q_counts, "reference.q_counts");
        assert_eq!(
            as_usize(
                &reference["first_fixed_index"],
                "reference.first_fixed_index"
            ),
            arc.first_fixed_index
        );
        assert_eq!(
            as_str(&reference["first_fixed_time"], "reference.first_fixed_time"),
            arc.first_fixed_time
        );
        assert_eq!(
            as_str(&reference["final_status"], "reference.final_status"),
            arc.final_status
        );
        assert_eq!(
            as_i64(&reference["satellites_min"], "reference.satellites_min"),
            arc.satellites_min
        );
        assert_eq!(
            as_i64(&reference["satellites_max"], "reference.satellites_max"),
            arc.satellites_max
        );

        assert_close(
            as_f64(&reference["fix_rate"], "reference.fix_rate"),
            arc.fix_rate,
            1.0e-12,
            "fix rate",
        );
        assert_close(
            as_f64(&reference["final_ratio"], "reference.final_ratio"),
            arc.final_ratio,
            1.0e-12,
            "final ratio",
        );
        assert_close(
            as_f64(
                &reference["final_truth_error_m"],
                "reference.final_truth_error_m",
            ),
            arc.final_truth_error_m,
            1.0e-12,
            "final truth error",
        );
        assert_close(
            as_f64(
                &reference["mean_truth_error_m"],
                "reference.mean_truth_error_m",
            ),
            arc.mean_truth_error_m,
            1.0e-12,
            "mean truth error",
        );
        assert_close(
            as_f64(
                &reference["max_truth_error_m"],
                "reference.max_truth_error_m",
            ),
            arc.max_truth_error_m,
            1.0e-12,
            "max truth error",
        );

        assert_eq!(epochs.len(), arc.epochs);
        assert_eq!(as_str(&epochs[0]["time"], "first time"), arc.first_time);
        assert_eq!(
            as_str(&epochs.last().expect("last epoch")["time"], "last time"),
            arc.last_time
        );
        assert_eq!(count_q(epochs, 1), arc.fixed_epochs);

        let first_fixed = &epochs[arc.first_fixed_index];
        assert_eq!(
            as_str(&first_fixed["fix_status"], "first_fixed.fix_status"),
            "fixed"
        );
        assert!(as_f64(&first_fixed["ratio"], "first_fixed.ratio") >= 3.0);
    }
}
