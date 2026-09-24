# RINEX navigation fixture provenance

Where the broadcast navigation files in this directory came from and how the
committed copies were derived. Sizes and SHA-256 digests are of the committed
files.

## `BRDM00DLR_S_20201760000_01D_MN.rnx`

- **Product:** DLR/GSOC merged multi-GNSS broadcast navigation file, RINEX
  3.04, 2020 day-of-year 176 (2020-06-24), GPS week 2111.
- **Source:** IGS BKG archive,
  `https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2020/176/BRDM00DLR_S_20201760000_01D_MN.rnx.gz`.
- **Content:** uncompressed RINEX product with 384 GPS ephemeris records and
  mixed-constellation records. G08, G10, G16, G18, G20, G21, G26 and G27 have
  broadcast ephemerides at or immediately around 2020-06-24 12:00 GPST.
- **Committed file:** 8619012 bytes, SHA-256
  `778e99a30b9fc3f2ea2844219f459535a9060645ec6accf89d83410154db5c53`.
- **Used by:** `fixtures-generators/rtklib_sp3_oracle/generate.sh` for RTKLIB
  `pntpos`; `satposs` requires broadcast ephemerides for transmit-time
  estimation before using precise SP3 positions.

## `ESBC00DNK_R_20201770000_01D_MN.rnx`

- **Product:** IGS MGEX daily mixed broadcast navigation file for station
  ESBC00DNK (Esbjerg, Denmark), 2020 day-of-year 177 (2020-06-25), GPS week
  2111. RINEX 3.05 `NAVIGATION DATA MIXED`; the header records `sbf2rin-13.4.5`
  as the converting program and a `gfzrnx-1.16` file merge.
- **Source:** the public `nav-solutions/data` repository, which redistributes
  IGS/MGEX products:
  `https://raw.githubusercontent.com/nav-solutions/data/main/NAV/V3/ESBC00DNK_R_20201770000_01D_MN.rnx.gz`.
  The original product carries GPS, Galileo, BeiDou, GLONASS, QZSS and SBAS
  records.
- **Derivation:** decompressed, then filtered to the GPS, Galileo and BeiDou
  records with the header kept verbatim through `END OF HEADER`:

  ```sh
  awk '
    BEGIN { inhdr=1; keep=0 }
    inhdr { print; if ($0 ~ /END OF HEADER/) inhdr=0; next }
    /^[A-Z][0-9][0-9] / { keep = ($0 ~ /^[GEC]/) ? 1 : 0 }
    { if (keep) print }
  ' <decompressed> > ESBC00DNK_R_20201770000_01D_MN.rnx
  ```

- **Content:** 257 GPS, 1602 Galileo and 357 BeiDou records. BeiDou PRNs run
  from C05 (a BDS-2 geostationary satellite) to C37. Record epochs run from
  2020-06-24 to 2020-06-26; the 2020-06-24 records let broadcast orbits be
  compared against the day-176 precise product
  `../sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3`.
- **Committed file:** 1452728 bytes, SHA-256
  `069f73afc10e9c1a8b87b7fbbb774f3eb9be94fb4da4ac365cfd4356c6ebfd36`.
- **Used by:** the RINEX NAV parser tests and the broadcast-ephemeris goldens
  (`../broadcast_golden.json`, `../broadcast_comparison_golden.json`).

## `ESBC00DNK_R_20201770000_01D_RN.rnx`

- **Derivation:** the GLONASS records of the same decompressed ESBC00DNK
  product, with the same verbatim header (identical to the MN file's header)
  and the same filter keeping `^R` records instead of `^[GEC]`.
- **Content:** 510 GLONASS state-vector records (PZ-90 position, velocity and
  lunisolar acceleration, `-TauN`/`+GammaN` clock terms and the frequency
  channel). The header carries `LEAP SECONDS` 18.
- **Committed file:** 223310 bytes, SHA-256
  `b3cbf368d8784b9fa9e77e8025be63a961fd53db7901bca7b3dd9ada7707a5a9`.

## `KMS300DNK_R_20221591000_01H_MN.rnx`

- **Source:** `nav-solutions/data`,
  `https://raw.githubusercontent.com/nav-solutions/data/main/NAV/V4/KMS300DNK_R_20221591000_01H_MN.rnx.gz`,
  committed decompressed and otherwise verbatim.
- **Content:** RINEX 4.00 mixed navigation, one hour of 2022 day-of-year 159.
  EPH frames: GPS LNAV 30, Galileo INAV 55 and FNAV 53, BeiDou D1 33 and D2 3,
  GLONASS FDMA 24, QZSS LNAV 1, SBAS 158; plus ION (3) and STO (3) frames.
- **Committed file:** 169572 bytes, SHA-256
  `9afdb2e289aadfb7de72b8401652d01598fa58e231c1370fe4700f4a11f1606d`.

## `BRDC00GOP_R_20210010000_01D_MN.rnx`

- **Source:** `nav-solutions/data`,
  `https://raw.githubusercontent.com/nav-solutions/data/main/NAV/V3/BRDC00GOP_R_20210010000_01D_MN.rnx.gz`.
  A merged multi-GNSS broadcast file from GOP/RIGTC (Pecny), RINEX 3.04, 2021
  day-of-year 1.
- **Content:** the full header, with `IONOSPHERIC CORR` lines for GAL, GPSA/GPSB,
  QZSA/QZSB, BDSA/BDSB and IRNA/IRNB, followed by four records (C01, E03, R10,
  S36).
- **Committed file:** 3955 bytes, SHA-256
  `eb566f1a5de27126f52ea54b45aabdc831fa3340d4a42d45197cf720a4fe23f5`.

## `BRD400DLR_S_20261800000_01H_MN_trim.rnx`

- **Upstream product:** `BRD400DLR_S_20261800000_01D_MN.rnx.gz`, the DLR/GSOC
  merged multi-GNSS broadcast navigation file (header program `BCEmerge`,
  DOI 10.57677/BRD400DLR), RINEX 4.02, 2026 day-of-year 180, from
  `https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2026/180/BRD400DLR_S_20261800000_01D_MN.rnx.gz`.
- **Trim:** the header through `END OF HEADER` plus eight EPH frames: G01 and
  G03 LNAV and CNAV, J02 LNAV, CNAV and CNV2, and C19 CNV2. Their epochs fall
  on 2026-06-29 between 00:00 and 02:00. The upstream product has no GPS CNV2
  frames.
- **Committed file:** 6253 bytes, SHA-256
  `f31502c3206b5f3edcfaeab3c7cc084514f51fb67cdef062806294f5cb804de2`.
- **Used by:** the RINEX 4 CNAV/CNV2 parsing tests in `src/rinex_nav/tests.rs`.
