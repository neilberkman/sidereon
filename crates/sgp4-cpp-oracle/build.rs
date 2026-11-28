use std::{env, path::Path, process::Command};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(sgp4_oracle_built)");
    if env::var("CARGO_CFG_TARGET_FAMILY").is_ok_and(|family| family == "wasm") {
        return;
    }

    let sources = Path::new("../sidereon-core/tests/cpp");
    println!("cargo:rerun-if-changed={}", sources.display());
    println!("cargo:rerun-if-changed=cpp/sgp4_portable_math.cpp");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .include(sources)
        .file("cpp/sgp4_portable_math.cpp")
        .file(sources.join("sgp4_dump_wrapper.cpp"))
        // The feature-gated diagnostic can link the original symbols.
        .define("SGP4Funcs", "PortableSgp4Funcs")
        .define("help", "sgp4_portable_help")
        .define("dbgfile", "sgp4_portable_dbgfile")
        .define("cpp_sgp4init_dump", "sgp4_portable_init_dump")
        .define("cpp_sgp4_step", "sgp4_portable_step")
        .flag_if_supported("-std=c++11")
        .warnings(false);
    // The Vallado expressions round after multiplication and after addition.
    if build.get_compiler().is_like_msvc() {
        build.flag("/fp:strict");
    } else {
        build.flag("-ffp-contract=off");
    }

    if cfg!(target_os = "macos")
        && env::var("CARGO_CFG_TARGET_VENDOR").is_ok_and(|vendor| vendor == "apple")
    {
        // The macOS archiver does not accept the deterministic archive flag.
        let objects = build.compile_intermediates();
        let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
        let status = Command::new("libtool")
            .args(["-static", "-no_warning_for_no_symbols", "-o"])
            .arg(Path::new(&out_dir).join("libsgp4_portable_oracle.a"))
            .args(objects)
            .status()
            .expect("libtool");
        assert!(status.success(), "libtool failed: {status}");
        println!("cargo:rustc-link-search=native={out_dir}");
        println!("cargo:rustc-link-lib=static=sgp4_portable_oracle");
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        build.compile("sgp4_portable_oracle");
    }
    println!("cargo:rustc-cfg=sgp4_oracle_built");
}
