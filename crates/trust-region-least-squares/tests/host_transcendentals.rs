use std::cell::RefCell;

use trust_region_least_squares::loss::{rho_for_loss_with, Loss, LossFunction};
use trust_region_least_squares::trf::{BackendError, HostNumerics};

#[derive(Default)]
struct ProbeBackend {
    log1p_inputs: RefCell<Vec<f64>>,
    atan_inputs: RefCell<Vec<f64>>,
}

impl HostNumerics for ProbeBackend {
    fn svd(
        &self,
        _a: &[f64],
        _m: usize,
        _n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), BackendError> {
        Err(BackendError::Failed(
            "SVD is not part of this test".to_string(),
        ))
    }

    fn log1p(&self, value: f64) -> Result<Option<f64>, BackendError> {
        self.log1p_inputs.borrow_mut().push(value);
        Ok(Some(11.0 + value))
    }

    fn atan(&self, value: f64) -> Result<Option<f64>, BackendError> {
        self.atan_inputs.borrow_mut().push(value);
        Ok(Some(13.0 + value))
    }
}

#[test]
fn robust_loss_with_path_consults_log1p_and_atan_hooks() {
    let backend = ProbeBackend::default();

    let cauchy = LossFunction::new(Loss::Cauchy, 1.0)
        .evaluate_with(&[0.5], &backend)
        .expect("Cauchy evaluation");
    assert_eq!(backend.log1p_inputs.borrow().as_slice(), &[0.25]);
    assert_eq!(cauchy.rho0, vec![11.25]);

    let arctan =
        rho_for_loss_with(Loss::Arctan, &[0.25], false, &backend).expect("Arctan evaluation");
    assert_eq!(backend.atan_inputs.borrow().as_slice(), &[0.25]);
    assert_eq!(arctan.rho0, vec![13.25]);
}

#[test]
fn defaulted_transcendental_hooks_decline() {
    struct DefaultBackend;

    impl HostNumerics for DefaultBackend {
        fn svd(
            &self,
            _a: &[f64],
            _m: usize,
            _n: usize,
        ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), BackendError> {
            Err(BackendError::Failed(
                "SVD is not part of this test".to_string(),
            ))
        }
    }

    let backend = DefaultBackend;
    assert_eq!(backend.log1p(0.25).expect("default log1p"), None);
    assert_eq!(backend.atan(0.25).expect("default atan"), None);
}
