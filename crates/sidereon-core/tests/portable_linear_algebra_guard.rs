//! Source guard for the portable linear-algebra boundary.
//!
//! The production core must not accidentally call nalgebra's `f64`
//! decompositions or dynamic matrix multiplication again. The portable module
//! is the one deliberate exception because it is the implementation boundary.

use std::fs;
use std::path::{Path, PathBuf};

const FORBIDDEN: &[&str] = &[
    ".svd(",
    ".lu(",
    ".cholesky(",
    ".qr(",
    ".symmetric_eigen(",
    ".try_inverse(",
];

fn rust_files(root: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).expect("read source directory") {
        let path = entry.expect("read source entry").path();
        if path.is_dir() {
            rust_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

#[test]
fn production_linear_algebra_stays_on_portable_boundary() {
    let source_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&source_root, &mut files);

    let mut violations = Vec::new();
    for path in files {
        if path.ends_with("astro/math/portable.rs")
            || path
                .components()
                .any(|component| component.as_os_str() == "tests")
        {
            continue;
        }

        let source = fs::read_to_string(&path).expect("read Rust source");
        let mut in_test_module = false;
        for (line_number, line) in source.lines().enumerate() {
            if line.trim_start().starts_with("#[cfg(test)]") {
                in_test_module = true;
            }
            if in_test_module {
                continue;
            }
            if FORBIDDEN.iter().any(|needle| line.contains(needle))
                || line.contains("transpose() *")
                || line.contains("* transpose()")
            {
                violations.push(format!("{}:{}: {}", path.display(), line_number + 1, line));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "production linear algebra escaped the portable boundary:\n{}",
        violations.join("\n")
    );
}
