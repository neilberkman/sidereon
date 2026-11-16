// Complete Debug projections retain shortest round-trip binary64 decimals,
// including signs of zero, every observation and all receive/transmit positions.
fn historical_rinex_projection() -> serde_json::Value {
    use serde_json::json;
    let real = RinexObs::parse(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx"
    )))
    .expect("RINEX");
    let gps = gps_values_obs(&[STEADY; 3]);
    let glonass = RinexObs::parse(&channel_code_text(&[-7, 6, -7], &[false; 3]))
        .expect("GLONASS RINEX");
    let mut cases = Vec::new();
    for (name, obs, (mut single, mut dual)) in [
        ("gps", &gps, gps_options(&[("C1C", "L1C")])),
        ("glonass", &glonass, glonass_options()),
        ("rinex_file", &real, gps_options(&[("C1C", "L1C")])),
    ] {
        single.max_epochs = Some(3);
        dual.max_epochs = Some(3);
        let single = build_rinex_rtk_arc(&FixedSource, obs, obs, &single).expect("single arc");
        let dual = build_dual_frequency_rinex_rtk_arc(&FixedSource, obs, obs, &dual)
            .expect("dual arc");
        assert!(!single.epochs.is_empty());
        assert!(!dual.epochs.is_empty());
        cases.push(json!({"name": name, "single": format!("{single:?}"), "dual": format!("{dual:?}")}));
    }
    json!(cases)
}
