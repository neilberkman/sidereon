fn main() {
    println!("cargo:rustc-check-cfg=cfg(sidereon_repo_tests)");
    println!("cargo:rerun-if-changed=../sidereon-core/tests/fixtures");

    if std::path::Path::new("../sidereon-core/tests/fixtures").is_dir() {
        println!("cargo:rustc-cfg=sidereon_repo_tests");
    }
}
