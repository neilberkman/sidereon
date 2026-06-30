use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum PropagationError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Numerical failure: {0}")]
    NumericalFailure(String),

    #[error("Maximum number of steps exceeded")]
    MaxStepsExceeded,

    #[error("Event failure: {0}")]
    EventFailure(String),

    #[error("Force model failure: {0}")]
    ForceModelFailure(String),
}
