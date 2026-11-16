//! C ABI for evaluating the RTKLIB source with deterministic binary64 math.
macro_rules! unary {
    ($($name:ident => $function:ident),*) => {$ (
        #[no_mangle]
        pub extern "C" fn $name(x: f64) -> f64 { libm::$function(x) }
    )*};
}
unary!(portable_sin => sin, portable_cos => cos, portable_sqrt => sqrt,
       portable_asin => asin, portable_atan => atan);
#[no_mangle]
pub extern "C" fn portable_atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}
