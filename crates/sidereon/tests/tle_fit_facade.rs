use sidereon::sgp4::{
    DecayLatch as ModuleDecayLatch, DecayLatchedError as ModuleDecayLatchedError, FitConfig,
    Loss as Sgp4Loss, XScale as Sgp4XScale,
};
use sidereon::{DecayLatch, DecayLatchedError, Loss, XScale};

#[test]
fn facade_reexports_fit_loss_and_x_scale_types() {
    let mut config = FitConfig::default();
    config.loss = Loss::Huber;
    config.x_scale = Some(XScale::Unit);

    assert_eq!(config.loss, Sgp4Loss::Huber);
    assert_eq!(config.x_scale, Some(Sgp4XScale::Unit));
}

#[test]
fn facade_reexports_sgp4_decay_latch_types() {
    let latch = DecayLatch::new();
    let module_latch = ModuleDecayLatch::new();
    assert_eq!(
        latch.first_failing_epoch(),
        module_latch.first_failing_epoch()
    );

    let _root_error: Option<DecayLatchedError> = None;
    let _module_error: Option<ModuleDecayLatchedError> = None;
}
