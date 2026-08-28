use sidereon::scenario::{
    scenario_source_transcript_fingerprint, simulate_scenario, simulate_scenario_with_media,
    simulate_scenario_with_source, simulate_scenario_with_source_and_media, DeclaredScenarioSource,
    Scenario, ScenarioConstellation, ScenarioEpochRange, ScenarioErrorBudget,
    ScenarioExternalProduct, ScenarioExternalProductKind, ScenarioGeodeticPosition,
    ScenarioMediaSources, ScenarioReceiver, ScenarioSignal, SyntheticKeplerOrbit,
    SyntheticKeplerSource, SyntheticObservationSet, SCENARIO_SCHEMA_VERSION,
};
use sidereon::{GnssSatelliteId, GnssSystem};

fn scenario() -> Scenario {
    Scenario {
        schema_version: SCENARIO_SCHEMA_VERSION,
        seed: 7,
        epochs: ScenarioEpochRange {
            start_j2000_s: 0.0,
            count: 1,
            cadence_s: 1.0,
        },
        receiver: ScenarioReceiver::StaticGeodetic {
            position: ScenarioGeodeticPosition {
                lat_rad: 0.0,
                lon_rad: 0.0,
                height_m: 0.0,
            },
        },
        constellation: ScenarioConstellation::SyntheticKeplerian {
            satellites: vec![SyntheticKeplerOrbit {
                satellite_id: GnssSatelliteId::new(GnssSystem::Gps, 1).expect("satellite"),
                semi_major_axis_m: 26_560_000.0,
                eccentricity: 0.0,
                inclination_rad: 0.0,
                raan_rad: 0.0,
                arg_perigee_rad: 0.0,
                mean_anomaly_rad: 0.0,
                epoch_j2000_s: 0.0,
                clock_bias_s: 0.0,
                clock_drift_s_s: 0.0,
            }],
        },
        signals: vec![ScenarioSignal::l1_ca(GnssSystem::Gps)],
        error_budget: ScenarioErrorBudget::default(),
    }
}

#[test]
fn facade_exposes_deterministic_synthetic_observables_and_truth_ledger() {
    let scenario = scenario();
    let first: SyntheticObservationSet = simulate_scenario(&scenario).expect("first simulation");
    let second = simulate_scenario_with_media(&scenario, &ScenarioMediaSources::default())
        .expect("second simulation");

    assert_eq!(first, second);
    assert_eq!(first.observation_count(), 1);
    assert_eq!(
        first.determinism_fingerprint(),
        second.determinism_fingerprint()
    );
    assert_eq!(
        first
            .truth_terms
            .pseudorange_sum_m(0)
            .expect("range terms")
            .to_bits(),
        first.observations.pseudorange_m[0].to_bits()
    );
    assert_eq!(
        first
            .truth_terms
            .carrier_phase_sum_cycles(0)
            .expect("phase terms")
            .to_bits(),
        first.observations.carrier_phase_cycles[0].to_bits()
    );
    assert_eq!(
        first
            .truth_terms
            .doppler_sum_hz(0)
            .expect("Doppler terms")
            .to_bits(),
        first.observations.doppler_hz[0].to_bits()
    );

    let source = SyntheticKeplerSource::new(
        scenario
            .satellites()
            .iter()
            .map(|sat| match &scenario.constellation {
                ScenarioConstellation::SyntheticKeplerian { satellites } => satellites
                    .iter()
                    .find(|orbit| orbit.satellite_id == *sat)
                    .copied()
                    .expect("orbit"),
                ScenarioConstellation::ExternalProducts { .. } => unreachable!(),
            })
            .collect(),
    )
    .expect("synthetic source");
    let identity = ScenarioExternalProduct {
        kind: ScenarioExternalProductKind::Sp3,
        product_id: "synthetic".into(),
        content_digest: "pending".into(),
    };
    let declared = DeclaredScenarioSource::new(&source, identity);
    assert_eq!(declared.source().satellites().len(), 1);
    assert!(ScenarioMediaSources::default().ionex.is_none());
}

#[test]
fn facade_exposes_declared_ephemeris_source_variants() {
    let synthetic = scenario();
    let (orbits, satellite_ids) = match &synthetic.constellation {
        ScenarioConstellation::SyntheticKeplerian { satellites } => (
            satellites.clone(),
            satellites.iter().map(|orbit| orbit.satellite_id).collect(),
        ),
        ScenarioConstellation::ExternalProducts { .. } => unreachable!(),
    };
    let source = SyntheticKeplerSource::new(orbits).expect("synthetic source");

    let mut external = synthetic.clone();
    let identity = ScenarioExternalProduct {
        kind: ScenarioExternalProductKind::Sp3,
        product_id: "synthetic".into(),
        content_digest: "pending".into(),
    };
    external.constellation = ScenarioConstellation::ExternalProducts {
        source: identity.clone(),
        satellites: satellite_ids,
    };

    let declared = DeclaredScenarioSource::new(&source, identity.clone());
    let digest = scenario_source_transcript_fingerprint(
        &external,
        &declared,
        &ScenarioMediaSources::default(),
    )
    .expect("source fingerprint");
    if let ScenarioConstellation::ExternalProducts { source, .. } = &mut external.constellation {
        source.content_digest = digest;
    }

    let declared = DeclaredScenarioSource::new(
        &source,
        match &external.constellation {
            ScenarioConstellation::ExternalProducts { source, .. } => source.clone(),
            ScenarioConstellation::SyntheticKeplerian { .. } => unreachable!(),
        },
    );
    let from_source = simulate_scenario_with_source(&external, &declared).expect("source run");
    let from_source_and_media = simulate_scenario_with_source_and_media(
        &external,
        &declared,
        &ScenarioMediaSources::default(),
    )
    .expect("source and media run");

    assert_eq!(from_source, from_source_and_media);
    assert_eq!(from_source.observation_count(), 1);
    assert_eq!(
        from_source.determinism_fingerprint(),
        simulate_scenario(&synthetic)
            .expect("synthetic run")
            .determinism_fingerprint()
    );
}
