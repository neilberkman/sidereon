# Hotpath benchmark

`hotpath.rs` measures the observable batch paths and the dynamic linear
algebra sizes used by precise positioning: normal products and Cholesky
decompositions at orders 50, 200, and 500, plus `J^T J` for a 2000 by 200
Jacobian.

The CI performance job runs:

```text
cargo bench -p sidereon-core --bench hotpath -- --noplot --sample-size 10 --measurement-time 1
python3 crates/sidereon-core/benches/check_hotpath.py target/criterion crates/sidereon-core/benches/hotpath_baseline.json
```

To refresh the baseline on the same `ubuntu-latest` runner class, run the
benchmark command above and add `--update` to the checker. Commit the resulting
`hotpath_baseline.json` together with the benchmark change, and inspect the
reported ratios before accepting the new reference.
