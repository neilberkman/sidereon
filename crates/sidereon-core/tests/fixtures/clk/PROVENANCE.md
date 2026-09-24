# Clock fixture provenance: AIUB short-name CODE excerpts

Where the short-name CODE clock excerpts in this directory came from and how
they were cut. Sizes and SHA-256 digests are of the committed files. Both
keep the source header verbatim and the source's data records, in file
order, up to the time stated; no record was altered.

## `COM17733_0000-0010.CLK`

- **Upstream product:** AIUB's short-name CODE MGEX final clock product for
  2014-01-01 (GPS week 1773 day 3), the first file of that series,
  `https://www.aiub.unibe.ch/download/CODE_MGEX/CODE/2014/COM17733.CLK.Z`
  (1150957 bytes, MD5 `4e21a4d9272d642f2452c5766f02d66f` as AIUB's
  `full_listing.csv` states it, SHA-256
  `17835842e73d4cc49370648c8cb8e25b1e3bed20c8a426404d0c935ae473b3b0`;
  decompressed 5157540 bytes, SHA-256
  `983de55cfbda4792fafc224af27e8cf7626533efad269d1d606dc9b3566e13fc`),
  fetched 2026-09-24. RINEX clock 2.00, GPS time, receiver and satellite
  clocks every 300 s from 00:00 through 23:55.
- **Trim:** the header and the `AR` and `AS` records of the epochs 00:00,
  00:05 and 00:10: 69, 70 and 70 satellites (G22 has no record at 00:00).
- **Committed file:** 751 lines, 66150 bytes, SHA-256
  `20484c9aa8af673da5b68739abd977954a05d9fbc39079731885ed6c62afbdcf`.
- **Used by:** `tests/code_legacy_products.rs`.

## `COM19620_0000-0100.CLK`

- **Upstream product:** AIUB's short-name CODE MGEX final clock product for
  2017-08-13 (GPS week 1962 day 0), the first file with 30-second satellite
  clocks,
  `https://www.aiub.unibe.ch/download/CODE_MGEX/CODE/2017/COM19620.CLK.Z`
  (3727183 bytes, MD5 `e89bbc50caa57f7e129c84461f5bdd1e` as AIUB's
  `full_listing.csv` states it, SHA-256
  `a70c8ce1726c86787f4016cd583ae49a18bd34f091a18bbeba4a702b99513628`;
  decompressed 24920775 bytes, SHA-256
  `229758342f84dd4ba421fe6847d4b808e1c711421e01f0ee13b7d3e935ae90a7`),
  fetched 2026-09-24. RINEX clock 2.00, GPS time, satellite clocks every
  30 s, receiver clocks every 300 s except the reference clock.
- **Trim:** the header and the records of the epochs 00:00:00, 00:00:30 and
  00:01:00: 82 satellites at each; 132 receivers at 00:00:00 and the
  reference receiver at the other two.
- **Committed file:** 545 lines, 47565 bytes, SHA-256
  `3189c5e70f9d233c5228129de86e2fd946ead4f5025f1175fdd4131eaeb61103`.
- **Used by:** `tests/code_legacy_products.rs`.
