use sidereon::estimation::{
    alpha_beta_apply_measurement, alpha_beta_filter_step, alpha_beta_predict,
    alpha_beta_steady_state_gains, cfar_ca_false_alarm_probability, cfar_ca_multiplier_from_pfa,
    cfar_ca_pfa_from_multiplier, cfar_ca_threshold, ewma_update, ewma_update_power_of_two,
    kalman_cv_steady_state_gains, mad_spread, nis_expected_value, nis_gate_test,
    nis_gate_threshold, nis_statistic, normalized_innovation, rts_smooth, smooth_track_rts,
    AlphaBetaGains, AlphaBetaState, TrackCoordinateFrame, TrackFilter, TrackRtsHistoryBuilder,
};

#[test]
fn scalar_estimation_and_detection_are_reachable_through_facade() {
    let gains = alpha_beta_steady_state_gains(4.0).expect("alpha-beta gains");
    assert!((gains.alpha - 0.864_145_399_682_717_8).abs() < 1.0e-12);
    assert!((gains.beta - 0.737_169_180_900_238_8).abs() < 1.0e-12);

    let state = AlphaBetaState {
        level: 5.0,
        rate: 2.0,
    };
    let predicted = alpha_beta_predict(state, 2.0).expect("alpha-beta prediction");
    assert_eq!(predicted.level, 9.0);
    assert_eq!(predicted.rate, 2.0);

    let explicit_gains = AlphaBetaGains {
        alpha: 0.6,
        beta: 0.8,
    };
    let updated = alpha_beta_apply_measurement(predicted, 8.0, 2.0, explicit_gains)
        .expect("alpha-beta update");
    assert_eq!(updated.level, 8.4);
    assert_eq!(updated.rate, 1.6);

    let step = alpha_beta_filter_step(state, 8.0, 2.0, explicit_gains).expect("filter step");
    assert_eq!(step.predicted, predicted);
    assert_eq!(step.updated, updated);
    assert_eq!(step.innovation, -1.0);

    let kalman = kalman_cv_steady_state_gains(4.0, 1.0, 1.0).expect("scalar Kalman gains");
    assert!((kalman.position_gain - gains.alpha).abs() < 1.0e-12);
    assert!((kalman.rate_gain - gains.beta).abs() < 1.0e-12);

    assert_eq!(
        normalized_innovation(2.0, 4.0).expect("normalized innovation"),
        1.0
    );
    assert_eq!(nis_statistic(2.0, 4.0).expect("NIS"), 1.0);
    assert_eq!(nis_expected_value(2).expect("NIS expectation"), 2.0);
    let threshold = nis_gate_threshold(1, 0.95).expect("NIS gate threshold");
    assert!(threshold > 3.8 && threshold < 3.9);
    assert!(
        nis_gate_test(1.0, 1.0, 1, 0.95)
            .expect("accepted NIS gate")
            .in_gate
    );
    assert!(
        !nis_gate_test(10.0, 1.0, 1, 0.95)
            .expect("rejected NIS gate")
            .in_gate
    );

    assert_eq!(ewma_update(10.0, 14.0, 0.25).expect("EWMA"), 11.0);
    assert_eq!(
        ewma_update_power_of_two(10.0, 14.0, 2).expect("power-of-two EWMA"),
        11.0
    );
    assert!(
        (mad_spread(&[1.0, 2.0, 3.0], 0.0).expect("MAD") - 1.482_602_218_505_602).abs() < 1.0e-15
    );

    let false_alarm_probability = 1.0e-3;
    let multiplier =
        cfar_ca_multiplier_from_pfa(16, false_alarm_probability).expect("CA-CFAR multiplier");
    assert!(
        (cfar_ca_pfa_from_multiplier(16, multiplier).expect("CA-CFAR PFA")
            - false_alarm_probability)
            .abs()
            < 1.0e-15
    );
    let absolute_threshold =
        cfar_ca_threshold(16, false_alarm_probability, 2.0).expect("CA-CFAR threshold");
    assert_eq!(absolute_threshold, 2.0 * multiplier);
    assert!(
        (cfar_ca_false_alarm_probability(16, absolute_threshold, 2.0)
            .expect("CA-CFAR inverse threshold")
            - false_alarm_probability)
            .abs()
            < 1.0e-15
    );
}

#[test]
fn track_filter_and_rts_smoothing_are_reachable_through_facade() {
    let mut filter = TrackFilter::from_position(
        TrackCoordinateFrame::CallerDefinedCartesian,
        0.0,
        vec![0.0],
        vec![vec![1.0]],
        1.0,
        0.1,
    )
    .expect("track filter");
    let mut history = TrackRtsHistoryBuilder::from_filter(&filter).expect("RTS history");

    let prediction = filter
        .predict_recorded(1.0, &mut history)
        .expect("recorded prediction");
    assert_eq!(prediction.predicted.position_m, vec![0.0]);
    assert_eq!(filter.state().t_s, 1.0);

    let innovation = filter
        .position_innovation(&[1.0], &[vec![0.25]])
        .expect("track innovation");
    assert_eq!(innovation.innovation, vec![1.0]);
    assert!(
        innovation
            .gate(0.95)
            .expect("track innovation gate")
            .in_gate
    );

    let update = filter
        .update_position_recorded(&[1.0], &[vec![0.25]], &mut history)
        .expect("recorded position update");
    assert!(update.updated.position_m[0] > 0.0);
    assert_eq!(filter.state(), &update.updated);

    let history = history.finish().expect("finished RTS history");
    let smoothed = rts_smooth(&history).expect("RTS smoothing");
    assert_eq!(smoothed.epochs.len(), 2);
    assert_eq!(
        smooth_track_rts(&history).expect("RTS smoothing alias"),
        smoothed
    );
    assert_eq!(smoothed.epochs[1].state, history.epochs[1].updated);
}
