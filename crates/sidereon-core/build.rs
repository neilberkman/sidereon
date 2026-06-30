// Build script.
//
// In normal builds this is a no-op - sidereon-core ships as a pure-Rust crate
// and downstream users compile no C++.
//
// Under the `sgp4-debug-oracle` feature (development only) we compile the
// Vallado C++ reference implementation as a static lib so the test suite can
// dump the C++ elsetrec field-by-field and diff against our pure-Rust port.
// This is purely a debug aid; the feature is off by default and not part of
// the published API.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(sgp4_oracle_built)");

    configure_repo_tests();

    let oracle_enabled = std::env::var("CARGO_FEATURE_SGP4_DEBUG_ORACLE").is_ok();
    if !oracle_enabled {
        return;
    }

    // The Vallado C++ source lives only in the git checkout - it is excluded
    // from the published tarball to keep the crate pure-Rust at the surface.
    // If a user enables `--features sgp4-debug-oracle` on a tarball-installed
    // copy of the crate, the source files won't be present. Keep the feature a
    // clean no-op instead of exporting FFI bindings that cannot be linked.
    if !std::path::Path::new("tests/cpp/SGP4.cpp").exists() {
        println!(
            "cargo:warning=sidereon-core: the `sgp4-debug-oracle` feature's \
             Vallado C++ oracle bindings are disabled because they require \
             the Vallado C++ source files in tests/cpp/, which are excluded from \
             the published crate. Clone the repo at \
             https://github.com/neilberkman/sidereon to use this feature."
        );
        return;
    }

    println!("cargo:rerun-if-changed=tests/cpp/SGP4.cpp");
    println!("cargo:rerun-if-changed=tests/cpp/SGP4.h");
    println!("cargo:rerun-if-changed=tests/cpp/sgp4_dump_wrapper.cpp");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("tests/cpp/SGP4.cpp")
        .file("tests/cpp/sgp4_dump_wrapper.cpp")
        .include("tests/cpp")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-w")
        // CRITICAL: clang ≥14 enables FP contraction (a*b+c → fma) by default
        // on macOS aarch64. Rust never auto-contracts. The Vallado port was
        // written assuming non-FMA arithmetic. The Python sgp4 PyPI wheel
        // (which generated the JSON fixture) is gcc-built and also non-FMA.
        // Disable FP contraction here so this oracle matches the JSON fixture
        // AND the pure-Rust port bit-for-bit.
        .flag_if_supported("-ffp-contract=off");

    // macOS: BSD `ar` doesn't accept the `-D` flag that newer cc-rs passes
    // for deterministic archives. Use libtool, which cc-rs has special-case
    // logic for and which is the platform-native static archiver on macOS.
    // Compile to object files only, then archive ourselves. cc-rs on
    // macOS has trouble auto-detecting the right archiver - newer versions
    // pass `-D` to BSD ar (not supported), and passing `libtool` doesn't
    // trigger libtool-style argument formatting.
    let objects = build.compile_intermediates();

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let lib_path = std::path::Path::new(&out_dir).join("libsgp4_oracle_cpp.a");

    #[cfg(target_os = "macos")]
    {
        let mut cmd = std::process::Command::new("libtool");
        cmd.arg("-static")
            .arg("-no_warning_for_no_symbols")
            .arg("-o")
            .arg(&lib_path);
        for obj in &objects {
            cmd.arg(obj);
        }
        let status = cmd
            .status()
            .expect("failed to invoke libtool to archive C++ objects");
        assert!(status.success(), "libtool failed: {:?}", status);
    }

    #[cfg(not(target_os = "macos"))]
    {
        let mut cmd = std::process::Command::new("ar");
        cmd.arg("rcs").arg(&lib_path);
        for obj in &objects {
            cmd.arg(obj);
        }
        let status = cmd
            .status()
            .expect("failed to invoke ar to archive C++ objects");
        assert!(status.success(), "ar failed: {:?}", status);
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=sgp4_oracle_cpp");
    println!("cargo:rustc-cfg=sgp4_oracle_built");
    link_cpp_runtime_for_target();
}

fn configure_repo_tests() {
    println!("cargo:rustc-check-cfg=cfg(sidereon_repo_tests)");
    println!("cargo:rerun-if-changed=tests/fixtures");

    if std::path::Path::new("tests/fixtures").is_dir() {
        println!("cargo:rustc-cfg=sidereon_repo_tests");
    }
}

fn link_cpp_runtime_for_target() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // Keep the manual archive path aligned with cc-rs' C++ standard-library
    // defaults. compile_intermediates() does not emit these link directives.
    let Some(runtime) = cpp_runtime_for_target(&target_os, &target_vendor, &target_env) else {
        return;
    };

    println!("cargo:rustc-link-lib=dylib={runtime}");
}

fn cpp_runtime_for_target(os: &str, vendor: &str, env: &str) -> Option<&'static str> {
    if env == "msvc" {
        None
    } else if vendor == "apple"
        || matches!(os, "freebsd" | "openbsd" | "aix" | "wasi")
        || (os == "linux" && env == "ohos")
    {
        Some("c++")
    } else if os == "android" {
        Some("c++_shared")
    } else {
        Some("stdc++")
    }
}
