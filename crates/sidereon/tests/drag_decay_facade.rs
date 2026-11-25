use sidereon::astro::forces::{
    DragForce as FacadeDragForce, SourcedDragForce as FacadeSourcedDragForce,
    SpaceWeather as FacadeSpaceWeather, SpaceWeatherSource as FacadeSpaceWeatherSource,
};
use sidereon::propagator::{
    estimate_decay, estimate_decay_with_context, estimate_decay_with_source,
    estimate_decay_with_source_and_context, propagate_states, DecayConfig, DecayEstimate,
    DragParameters, PropagationConfig, PropagationContext,
};
use sidereon::{
    DragForce, ObservationClass, SourcedDragForce, SpaceWeather, SpaceWeatherPolicy,
    SpaceWeatherSample, SpaceWeatherSource, SpaceWeatherTable,
};

#[test]
fn facade_reexports_drag_decay_and_matches_core() {
    let _: FacadeSpaceWeather = SpaceWeather::default();
    let drag = DragParameters::from_bc_factor_m2_kg(0.02, SpaceWeather::default(), 100.0)
        .expect("valid drag");
    let _: FacadeDragForce = DragForce::from_bc_factor_m2_kg(0.02, SpaceWeather::default(), 100.0)
        .expect("valid drag force");
    let source = SpaceWeatherSource::Fixed(SpaceWeather::default());
    let _: FacadeSpaceWeatherSource = source.clone();
    let _: FacadeSourcedDragForce = SourcedDragForce::new(drag, source.clone());
    let _: fn(
        sidereon::astro::state::CartesianState,
        &DecayConfig,
        &SpaceWeatherSource,
    ) -> Result<DecayEstimate, sidereon::propagator::DecayError> = estimate_decay_with_source;
    let _policy = SpaceWeatherPolicy::default();
    let _class = ObservationClass::Observed;
    let _sample_type: Option<SpaceWeatherSample> = None;
    let _table_type: Option<SpaceWeatherTable> = None;
    let _decay_api: fn(
        sidereon::astro::state::CartesianState,
        &DecayConfig,
    ) -> Result<DecayEstimate, sidereon::propagator::DecayError> = estimate_decay;
    let _decay_ctx_api: fn(
        sidereon::astro::state::CartesianState,
        &DecayConfig,
        &PropagationContext,
    ) -> Result<DecayEstimate, sidereon::propagator::DecayError> = estimate_decay_with_context;
    let _decay_src_ctx_api: fn(
        sidereon::astro::state::CartesianState,
        &DecayConfig,
        &SpaceWeatherSource,
        &PropagationContext,
    ) -> Result<DecayEstimate, sidereon::propagator::DecayError> =
        estimate_decay_with_source_and_context;
    let _decay_config = DecayConfig::new(drag);

    let facade_config = PropagationConfig::new(
        646_315_200.0,
        [6_778.137, 0.0, 0.0],
        [0.0, 7.668_558_175_407_055, 0.0],
    )
    .with_drag(drag);
    let core_config = sidereon_core::astro::propagator::PropagationConfig::new(
        646_315_200.0,
        [6_778.137, 0.0, 0.0],
        [0.0, 7.668_558_175_407_055, 0.0],
    )
    .with_drag(drag);
    let epochs = [646_315_200.0, 646_315_260.0];

    let facade = propagate_states(&facade_config, &epochs).expect("facade propagation");
    let core = sidereon_core::astro::propagator::propagate_states(&core_config, &epochs)
        .expect("core propagation");

    assert_eq!(facade, core);
}
