#![cfg(all(test, sgp4_oracle_built))]

mod cpp_oracle;

#[path = "../../sidereon-core/src/astro/sgp4/vallado.rs"]
#[allow(
    dead_code,
    unused_variables,
    unused_assignments,
    unused_mut,
    non_snake_case,
    non_camel_case_types,
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::too_many_arguments,
    clippy::needless_return,
    clippy::assign_op_pattern,
    clippy::manual_range_contains,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::float_cmp,
    clippy::needless_late_init,
    clippy::field_reassign_with_default
)]
mod vallado;

mod vallado_comparison;
