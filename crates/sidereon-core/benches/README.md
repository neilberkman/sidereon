# Hotpath benchmark

`hotpath.rs` measures the observable batch paths and the dynamic linear
algebra sizes used by precise positioning: normal products and Cholesky
decompositions at orders 50, 200, and 500, plus `J^T J` for a 2000 by 200
Jacobian.

The CI performance job checks the merge-base implementation and the branch
implementation back to back on the same `ubuntu-latest` runner. It runs the
same benchmark source against both trees, including the bundle, in-place,
cached, solve, product, Jacobian, and Cholesky cases, then fails if any branch
mean exceeds the merge-base mean by more than 1.25x:

```text
cargo bench -p sidereon-core --bench hotpath -- --noplot --sample-size 10 --measurement-time 1
python3 crates/sidereon-core/benches/check_hotpath.py BASE/criterion BRANCH/criterion
```

The gate covers the 12 application cases and the seven portable
linear-algebra cases. `BASE/criterion` and `BRANCH/criterion` are the two
output trees produced in that job. Both trees are uploaded as the
`hotpath-criterion` artifact so a failure can be inspected without rerunning
the benchmark. There is no committed absolute performance baseline to
refresh; local comparisons should use the same merge-base procedure and this
checker.
