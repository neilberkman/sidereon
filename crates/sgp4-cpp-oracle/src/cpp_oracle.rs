use std::os::raw::{c_char, c_double, c_int};

pub(super) const CPP_DUMP_DOUBLE_COUNT: usize = 112;
pub(super) const CPP_DUMP_INT_COUNT: usize = 5;

extern "C" {
    #[link_name = "sgp4_portable_init_dump"]
    pub(super) fn cpp_sgp4init_dump(
        satnum: *const c_char,
        opsmode: c_char,
        epoch_sgp4: c_double,
        bstar: c_double,
        ndot: c_double,
        nddot: c_double,
        ecco: c_double,
        argpo: c_double,
        inclo: c_double,
        mo: c_double,
        no_kozai: c_double,
        nodeo: c_double,
        epochyr: c_int,
        epochdays: c_double,
        jdsatepoch: c_double,
        jdsatepoch_frac: c_double,
        double_out: *mut c_double,
        int_out: *mut c_int,
    ) -> c_int;

    #[link_name = "sgp4_portable_step"]
    pub(super) fn cpp_sgp4_step(
        satnum: *const c_char,
        opsmode: c_char,
        epoch_sgp4: c_double,
        bstar: c_double,
        ndot: c_double,
        nddot: c_double,
        ecco: c_double,
        argpo: c_double,
        inclo: c_double,
        mo: c_double,
        no_kozai: c_double,
        nodeo: c_double,
        epochyr: c_int,
        epochdays: c_double,
        jdsatepoch: c_double,
        jdsatepoch_frac: c_double,
        tsince: c_double,
        r_out: *mut c_double,
        v_out: *mut c_double,
    ) -> c_int;
}

#[no_mangle]
extern "C" fn sgp4_oracle_sin(x: f64) -> f64 {
    libm::sin(x)
}

#[no_mangle]
extern "C" fn sgp4_oracle_cos(x: f64) -> f64 {
    libm::cos(x)
}

#[no_mangle]
extern "C" fn sgp4_oracle_atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

#[no_mangle]
extern "C" fn sgp4_oracle_pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}
