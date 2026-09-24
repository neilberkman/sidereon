# SP3 fixture provenance

Where the precise-orbit files in this directory came from and how the
committed copies were derived. Epoch counts, sizes and SHA-256 digests
are of the committed files.

## `GRG0MGXFIN_20201760000_01D_15M_ORB.SP3`

- **Product:** IGS MGEX final precise orbit and clock product from the CNES/CLS
  analysis center (`GRG`), 2020 day-of-year 176 (2020-06-24), GPS week 2111,
  SP3-c, GPS time, 15-minute grid. A public IGS MGEX product, committed
  verbatim after decompression, from the `nav-solutions/data` repository:
  `https://raw.githubusercontent.com/nav-solutions/data/main/SP3/C/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3.gz`.
- **Content:** 96 epochs, 00:00 through 23:45 GPST; 75 satellites across GPS,
  GLONASS and Galileo (no BeiDou); position and clock records only.
- **Committed file:** 443618 bytes, SHA-256
  `e123cedd659bf83eadaf40aa3845d58457e1f2596ffbd196de433a79898cffab`.
- **Used by:** the SP3 parser and interpolation tests, the reduced-orbit tests,
  the observables, velocity, geometry and DGNSS tests, and the SPP trace
  fixtures (`../spp_trace_*.json`), whose receiver epochs fall on this day.

## `GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3`

- **Upstream product:** `GBM0MGXRAP_20201770000_01D_05M_ORB.SP3.gz`, the GFZ
  MGEX rapid precise orbit and clock product for 2020 day-of-year 177
  (2020-06-25), GPS week 2111, 5-minute grid, GPS time.
- **Trim:** decompressed; kept the header verbatim and the first 24 epochs
  (00:00 through 01:55 GPST), set the epoch count on the first line to 24, and
  appended `EOF`.
- **Content:** 123 satellites across GPS, GLONASS, Galileo, BeiDou and QZSS.
- **Committed file:** 243328 bytes, SHA-256
  `769e61ab9153cac0c9103df1b1721cda8a8e04457188b862a5f63c431ca3cba2`.

## `GBM_BDS_C21_C08_trim.sp3`

- **Upstream product:** the same GFZ rapid product,
  `GBM0MGXRAP_20201770000_01D_05M_ORB.SP3`.
- **Trim:** kept the header verbatim (it still lists all 123 satellites) and
  only the position records of BeiDou C21 (MEO) and C08 (IGSO) across all 288
  epochs. No value was altered.
- **Committed file:** 72293 bytes, SHA-256
  `f77d83a0da91e7112c2890ba7aae29326b8c621cfee58ac18e4243d86e40238b`.

## `COD0MGXFIN_20201770000_01D_05M_ORB.SP3`

- **Product:** `COD0MGXFIN_20201770000_01D_05M_ORB.SP3.gz`, the CODE MGEX final
  precise orbit and clock product for 2020 day-of-year 177, GPS week 2111,
  committed decompressed and otherwise verbatim.
- **Content:** 289 epochs at 5-minute spacing, 2020-06-25 00:00 through
  2020-06-26 00:00 GPST; 90 satellites across GPS, GLONASS, Galileo, BeiDou and
  QZSS.
- **Committed file:** 1597406 bytes, SHA-256
  `54b70fa009a840ecf8cec25fbd4d749c9aaef7c95bdf463484e115f74d802215`.

## `IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3`

- **Upstream product:** IGS final GPS orbits for 2026 day-of-year 120, GPS week
  2416,
  `https://igs.bkg.bund.de/root_ftp/IGS/products/2416/IGS0OPSFIN_20261200000_01D_15M_ORB.SP3.gz`.
- **Trim:** decompressed; kept the header verbatim (its epoch count still reads
  96) and the 11 epochs from 09:45 through 12:15 GPST, then appended `EOF`.
- **Content:** 31 GPS satellites.
- **Committed file:** 29319 bytes, SHA-256
  `8d3896583b8d2662d3012485c5c92f52a72124ccf199eb24760464411f968d6b`.

## `GAP_G01_20201760000_15M.sp3`

- **Derived from:** `GRG0MGXFIN_20201760000_01D_15M_ORB.SP3` above.
- **Change:** G01's position records at the 11 epochs 07:30 through 10:00 GPST
  are removed; every other line, the header included, is identical to the
  source. G01 keeps records at the other 85 of the 96 epochs.
- **Committed file:** 442947 bytes, SHA-256
  `f45ce1a1dc5006112412c354bca7402ea7ec92a23114b74272798c30065b2584`.
- **Used by:** the SP3 interpolation tests that check no window spans the
  gap, among others.

## `COD0OPSFIN_20261200945_02H30M_15M_ORB_trim.SP3`, `GFZ0OPSFIN_20261200945_02H30M_15M_ORB_trim.SP3`, `JPL0OPSFIN_20261200945_02H30M_15M_ORB_trim.SP3`

- **Upstream products:** the IGS final orbit products of CODE (AIUB, SP3-d),
  GFZ (SP3-c) and JPL (SP3-c) for 2026 day-of-year 120 (2026-04-30), GPS week
  2416, frame IGc20, GPS time, 5-minute grid:
  `COD0OPSFIN_20261200000_01D_05M_ORB.SP3.gz`,
  `GFZ0OPSFIN_20261200000_01D_05M_ORB.SP3.gz` and
  `JPL0OPSFIN_20261200000_01D_05M_ORB.SP3.gz` from the IGS product directory
  for week 2416 (for example `ftp://igs.gnsswhu.cn/pub/gps/products/2416/`).
- **Trim:** each keeps its header verbatim (the first line still gives 289
  epochs) and only the 11 epochs 09:45 through 12:15 GPST that fall on a
  15-minute grid, for the eight GPS satellites G02, G03, G04, G05, G09, G17,
  G25 and G31. No value was altered.
- **Committed files:**
  - COD: 7227 bytes, SHA-256
    `f3ad3f637134651d086815345f3e5f531a9dbacb6f739b7dddf664e0ab3a1795`
  - GFZ: 9805 bytes, SHA-256
    `9e50edc53ac42791923fd71c39b49a97bf516084f1d2b1dcb260685d2a8f11cc`
  - JPL: 8210 bytes, SHA-256
    `9ac5aafdabed38679892f57b42864cc3716d997400280f29ee8049a37057adf4`
- **Used by:** the multi-center SP3 combination test in `src/sp3/combine.rs`,
  against `IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3`.

## `IGS0OPSFIN_20261330000_03H_15M_ORB.SP3`

- **Upstream product:** IGS final GPS orbits for 2026 day-of-year 133
  (2026-05-13), GPS week 2418, SP3-c, frame IGc20, 15-minute grid,
  `https://igs.bkg.bund.de/root_ftp/IGS/products/2418/IGS0OPSFIN_20261330000_01D_15M_ORB.SP3.gz`.
- **Trim:** the header and the first 13 epochs, 00:00 through 03:00 GPST (the
  first line gives 13), then `EOF`.
- **Content:** 32 GPS satellites.
- **Committed file:** 35458 bytes, SHA-256
  `94f130b471882c99e7b38fcd32e7fe175a2dd47c8608ad6e779c44518a42abab`.
- **Used by:** the ZIM2 decimeter PPP arc test (`tests/ppp_decimeter_arc.rs`),
  among others.

## `GRG0OPSULT_20261880600_02D_05M_ORB_C19_C23_E02_E05_1500_1730.SP3`

- **Upstream product:** `GRG0OPSULT_20261880600_02D_05M_ORB.SP3.gz`, the
  CNES/CLS ultra-rapid GPS, BeiDou and Galileo orbits issued 2026-07-07 06:00
  (2026 day-of-year 188), SP3-c, frame IGS20, 5-minute grid. The trim is
  described in the file's own `/*` comment lines.
- **Trim:** BeiDou C19 through C23 and Galileo E02 through E05, 31 epochs from
  2026-07-08 15:00 through 17:30 GPST; the header lists only those nine
  satellites and gives 31 epochs.
- **Committed file:** 18831 bytes, SHA-256
  `c2ed729dd53502c514d3a0d537525b7fa1ae4920a0c12f35f5ce4525a2f17c09`.

## `GRG0OPSULT_20261880600_02D_05M_ORB_E03_E10_1300_1430.SP3`

- **Upstream product:** the same ultra-rapid product, from the CDDIS product
  directory for GPS week 2426, as the file's `/*` comment lines state.
- **Trim:** Galileo E03 through E10, 19 epochs from 2026-07-08 13:00 through
  14:30 GPST; the header lists only those eight satellites and gives 19
  epochs.
- **Committed file:** 10683 bytes, SHA-256
  `0b2e752b8d69800cd1ef9b774d4d5879e2df049d776a14d8d228338d11b15245`.
- **Used by:** the real RTCM Galileo broadcast test
  (`tests/rtcm_galileo_real.rs`); the C19-C23/E02-E05 trim above is used by
  the real RTCM multi-GNSS broadcast tests (`tests/rtcm_multignss_real.rs`).

## `qzu24263_06_J02_J04_J08_1500_1900.sp3`

- **Upstream product:** `qzu24263_06.sp3`, the QZSS ultra-rapid orbit (QZU)
  from the public QZSS (Cabinet Office) archive, GPS week 2426 day 3, issue
  06, SP3-c, 15-minute grid, as the file's `/*` comment lines state.
- **Trim:** QZSS J02, J03, J04 and J08, 17 epochs from 2026-07-08 15:00
  through 19:00 GPST; the header lists only those four satellites and gives
  17 epochs.
- **Committed file:** 6836 bytes, SHA-256
  `cdb11b70669891ccb0e8cd7e3ad9a41d49a7409051dfc8e574e7a516fefe7a83`.
- **Used by:** the real RTCM multi-GNSS broadcast tests
  (`tests/rtcm_multignss_real.rs`).

## `trimmed_go_static.sp3`

- **Derived from:** `GRG0MGXFIN_20201760000_01D_15M_ORB.SP3` above.
- **Content:** a hand-assembled SP3-c file (agency and data-used fields
  `TEST`) with the source's position and clock records for G08, G10, G16,
  G18, G20, G21, G26 and G27 at the five epochs 11:45 through 12:45 GPST on
  2020-06-24. The values are the source's; one record (G26 at 12:15) has its
  clock field one column left of the source's. The file ends at the last
  record, without `EOF`.
- **Committed file:** 3192 bytes, SHA-256
  `8f71a12445f3067eea25b06617926da25baceffa29f87587a836252f93bd38ae`.
- **Used by:** `tests/go_fixture_parity.rs` and the static-positioning tests.

## `degenerate_coincident_5sat.sp3`

Hand-written, not a redistributed product: five GPS satellites placed at one
ECEF point, for the rank-deficient geometry path.
