# IONEX fixture provenance: AIUB short-name CODE excerpts

Where the short-name CODE IONEX excerpts in this directory came from and how
they were cut. Upstream compressed sizes, listing MD5 values and SHA-256
digests are recorded separately from the committed excerpt sizes and digests.
Each excerpt keeps the source header and selected TEC/RMS maps verbatim except
for the map count and last-map epoch needed to describe the excerpt; products
without RMS maps retain none. Each ends with `END OF FILE`.

## `CODG0010.95I`

- **Upstream product:** AIUB's short-name CODE final ionosphere product for
  1995-01-01, first day of the series,
  `https://www.aiub.unibe.ch/download/CODE/1995/CODG0010.95I.Z` (10282 bytes,
  MD5 `f6257b590e8ae52f8347b825a1299f6f` in AIUB's `full_listing.csv`, SHA-256
  `dd27804b50fcf5a42bd3998d7192b4a44f276214332772c7e0f38d580da3753e`;
  decompressed 34532 bytes, SHA-256
  `fe02575ef376c2d550970e861d8cf6987cdf9ee4133d667ad180973c0a51beaa`),
  fetched 2026-09-24. `INTERVAL` 86400, one TEC map at 12:00, no RMS maps.
- **Trim:** no maps removed; no header values changed.
- **Committed file:** 457 lines, 34532 bytes, SHA-256
  `fe02575ef376c2d550970e861d8cf6987cdf9ee4133d667ad180973c0a51beaa`.
- **Used by:** `tests/code_legacy_products.rs`.

## `CODG0330_maps1-2.97I`

- **Upstream product:** AIUB's short-name CODE final ionosphere product for
  1997-02-02, first two-hourly era day,
  `https://www.aiub.unibe.ch/download/CODE/1997/CODG0330.97I.Z` (104352 bytes,
  MD5 `1f7de7ac61475626e5585b9df5bb8dbe` in AIUB's `full_listing.csv`, SHA-256
  `53b8f6f6bca126f38e0964a27b734d04a0889f8fab0ca7e92b0185ce862ceaba`;
  decompressed 779034 bytes, SHA-256
  `2a33222a34a999088ddafa253a86ca6c66ec9c9a817c836b69a5b48223cfa0db`),
  fetched 2026-09-24. `INTERVAL` 7200, 12 maps from 01:00 through 23:00.
- **Trim:** TEC and RMS maps 1 and 2 (01:00 and 03:00); `EPOCH OF LAST MAP`
  set to 03:00 and `# OF MAPS IN FILE` to 2.
- **Committed file:** 1774 lines, 133754 bytes, SHA-256
  `464af65493c50972dbe6a91951394a0226cfa8f489d40a72aa15639543da6e09`.
- **Used by:** `tests/code_legacy_products.rs`.

## `CODG0550.97I`

- **Upstream product:** AIUB's short-name CODE final ionosphere product for
  1997-02-24, the return to daily maps,
  `https://www.aiub.unibe.ch/download/CODE/1997/CODG0550.97I.Z` (10245 bytes,
  MD5 `938ecc204b92b24256a7c021e546b5d4` in AIUB's `full_listing.csv`, SHA-256
  `c64ad3e2462a1a1546fda2bd8306b839f66600df800616f404c90f90f2ee1618`;
  decompressed 34613 bytes, SHA-256
  `93b0106a056504279542862b5cef9816797799cadc0f3f0e3ae32adbf45b2a17`),
  fetched 2026-09-24. `INTERVAL` 86400, one TEC map at 12:00, no RMS maps.
- **Trim:** no maps removed; no header values changed.
- **Committed file:** 458 lines, 34613 bytes, SHA-256
  `93b0106a056504279542862b5cef9816797799cadc0f3f0e3ae32adbf45b2a17`.
- **Used by:** `tests/code_legacy_products.rs`.

## `CODG0870_maps1-2.98I`

- **Upstream product:** AIUB's short-name CODE final ionosphere product for
  1998-03-28, start of the second two-hourly era,
  `https://www.aiub.unibe.ch/download/CODE/1998/CODG0870.98I.Z` (99082 bytes,
  MD5 `1918c3b4a1d2c526ae8ec4deb8ee387b` in AIUB's `full_listing.csv`, SHA-256
  `c47b3f3e570f2b66711f13fb3ddd9e7c25ea82010909002d87791ef4072d6735`;
  decompressed 391947 bytes, SHA-256
  `d621d1fec8186236c98199072e23fec213f7374a3a22afceacdf14c0ef2bcc9a`),
  fetched 2026-09-24. `INTERVAL` 7200, 12 TEC maps from 01:00 through 23:00;
  the upstream file has no RMS maps.
- **Trim:** TEC maps 1 and 2 (01:00 and 03:00); `EPOCH OF LAST MAP` set to
  03:00 and `# OF MAPS IN FILE` to 2.
- **Committed file:** 917 lines, 69307 bytes, SHA-256
  `2482130ee6ab9826195d5c4870c71b634000395f29a0b9a7586c906648c9f9d2`.
- **Used by:** `tests/code_legacy_products.rs`.

## `CODG3070_maps1-2.02I`

- **Upstream product:** AIUB's short-name CODE final ionosphere product for
  2002-11-03, the first day whose two-hourly series contains 13 maps,
  `https://www.aiub.unibe.ch/download/CODE/2002/CODG3070.02I.Z` (221364 bytes,
  MD5 `9296703efbe0093baa571ed4fb469ce2` in AIUB's `full_listing.csv`, SHA-256
  `e974cc5364e9d48b37204c39bd20c6e04afa52f21954370acfd66c28ea410510`;
  decompressed 858871 bytes, SHA-256
  `2c85904a0d4b312763d9e1d9b2f111eb13d68f19844a04ed2d7f75f8bf887c14`),
  fetched 2026-09-24. The upstream header states `INTERVAL` 7200 and 13 maps
  from 00:00 through 24:00; it has 13 TEC and 13 RMS maps.
- **Trim:** TEC and RMS maps 1 and 2 (00:00 and 02:00); `EPOCH OF LAST MAP`
  set to 02:00 and `# OF MAPS IN FILE` to 2.
- **Committed file:** 1963 lines, 149063 bytes, SHA-256
  `bbb4b45f993103cd6f39a691bc3b4d702428f857530a639b1ddfe7399ca55a28`.
- **Used by:** `tests/code_legacy_products.rs`.

## `CODG2910_maps1-3.14I`

- **Upstream product:** AIUB's short-name CODE final ionosphere product for
  2014-10-18, the last two-hourly day of that series,
  `https://www.aiub.unibe.ch/download/CODE/2014/CODG2910.14I.Z` (198869
  bytes, MD5 `48129f0cddf10c27a7facdc23569512c` as AIUB's `full_listing.csv`
  states it, SHA-256
  `be1b4f19d7e63de567813f59728510ea30a79b21952406a3df1798959140d787`;
  decompressed 887383 bytes, SHA-256
  `24d0d6f8747e95bf15fc6555524c873d093b60e39c176fea3ee674d4ab2a2e92`),
  fetched 2026-09-24. `INTERVAL` 7200, 13 maps from 00:00 through 24:00.
- **Trim:** maps 1 to 3 (00:00, 02:00, 04:00); `EPOCH OF LAST MAP` set to
  04:00 and `# OF MAPS IN FILE` to 3.
- **Committed file:** 3173 lines, 242103 bytes, SHA-256
  `17fff5734f3933e4e9c03420122f1d6fef89560a9015998ae0ace682141e8590`.
- **Used by:** `tests/code_legacy_products.rs`.

## `CODG2920_maps1-3.14I`

- **Upstream product:** AIUB's short-name CODE final ionosphere product for
  2014-10-19, the first hourly day of that series,
  `https://www.aiub.unibe.ch/download/CODE/2014/CODG2920.14I.Z` (378925
  bytes, MD5 `b18afae75329f0f56ecdf46d3d31eca4` as AIUB's `full_listing.csv`
  states it, SHA-256
  `bff0dc183f7d9962d23ed789d927a8dc05776b9de6ee188ccd2208b91a8a5d2f`;
  decompressed 1662043 bytes, SHA-256
  `64dee71070ab3d7dbcec6caa8239afe64d368bf55f4e65847524a47d9061c502`),
  fetched 2026-09-24. `INTERVAL` 3600, 25 maps from 00:00 through 24:00.
- **Trim:** maps 1 to 3 (00:00, 01:00, 02:00); `EPOCH OF LAST MAP` set to
  02:00 and `# OF MAPS IN FILE` to 3.
- **Committed file:** 3177 lines, 242427 bytes, SHA-256
  `f1eadde3fb4bcd0572c40b73f65a0e785c5dab5ea2c28aecadfbc663f1ecdcf6`.
- **Used by:** `tests/code_legacy_products.rs`.
