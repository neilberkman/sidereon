# Third-Party Notices

sidereon is licensed under the MIT License (see LICENSE). It contains, ports,
or reimplements algorithms from the following third-party sources, and vendors
test data from some of them. All are permissive licenses; their required
attributions are reproduced below. No copyleft (GPL/LGPL/AGPL/MPL/EUPL/CDDL)
code or dependencies are included.

--------------------------------------------------------------------------------
## Orekit (Apache License 2.0) — test data

Four CCSDS TDM KVN files under `crates/sidereon-core/tests/fixtures/tdm/`, named
`orekit_*.kvn`, are copied byte for byte from Orekit's
`src/test/resources/ccsds/tdm/kvn/`. No Orekit source code is used. Each file
carries whitespace or a line length CCSDS 503.0-B-2 forbids, which is what the
tests measure, so the bytes are kept as published rather than cleaned up.

  OREKIT Copyright 2002-2026 CS GROUP

  Licensed under the Apache License, Version 2.0 (the "License"); you may not
  use these files except in compliance with the License. You may obtain a copy
  of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  License for the specific language governing permissions and limitations under
  the License.

--------------------------------------------------------------------------------
## RTKLIB (BSD 2-Clause)

The integer least-squares (MLAMBDA/LAMBDA) routine is a Rust port of RTKLIB's
`lambda.c`. The broadcast ephemeris evaluators follow RTKLIB's `ephemeris.c`
(`eph2pos`, `eph2clk`, `geph2pos`, `geph2clk`, `seph2pos`, `deq`, `glorbit`) and its
ephemeris selection (`seleph`, `selgeph`, `selseph`, `satexclude`, `uniqnav`), and
`crates/sidereon-core/fixtures-generators/rtklib_oracle/rtklib_ephemeris_oracle.c`
contains those functions and `rtkcmn.c` time helpers copied from RTKLIB demo5 to
check the evaluators against RTKLIB. The full licence text is in
`crates/sidereon-core/RTKLIB-LICENSE.txt`.

  Copyright (c) 2007-2020, T. Takasu, All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES ARISING IN ANY WAY
  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
  DAMAGE.

--------------------------------------------------------------------------------
## ERFA (BSD 3-Clause)

Nutation/precession coefficient tables and conventions are derived from ERFA
(Essential Routines for Fundamental Astronomy), itself derived from IAU SOFA.

  Copyright (C) 2013-2021, NumFOCUS Foundation. All rights reserved.
  Licensed under the BSD 3-Clause License.

--------------------------------------------------------------------------------
## SciPy (BSD 3-Clause)

The trust-region least-squares solver (`trust-region-least-squares`) contains
Rust ports of corresponding SciPy least-squares routines.

  Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
  All rights reserved.
  Licensed under the BSD 3-Clause License.

--------------------------------------------------------------------------------
## IERS Conventions Software

The solid-earth / ocean / pole tide displacement follows the IERS Conventions
reference routines (e.g. DEHANTTIDEINEL), used under the IERS Conventions
Software License. The IERS acknowledgment and derived-work description are
retained in the relevant source, and the intact license notice is reproduced in
`crates/sidereon-core/IERS-CONVENTIONS-SOFTWARE-LICENSE.txt`.
This Sidereon derived work is neither distributed by nor endorsed by the IERS
Conventions Center.

--------------------------------------------------------------------------------
## Reference algorithms (no code copied)

The following informed reimplementations from public specifications/literature;
no source code was copied:

- SGP4 / SDP4: D. Vallado et al., "Revisiting Spacetrack Report #3" (AIAA), and
  the CelesTrak reference vectors (validation only).
- Frame/time-scale conventions cross-checked against Skyfield (MIT) and the IAU
  conventions.
- Galileo NeQuick-G: reimplemented from the Galileo OS SIS ICD "Ionospheric
  Correction Algorithm for Galileo Single Frequency Users"; MODIP and CCIR data
  tables transcribed as ITU-R / EU-JRC reference data (facts).
- NRLMSISE-00: U.S. Naval Research Laboratory (public domain).
- EGM96 geoid undulation grid (`crates/sidereon-core/src/egm96_geoid_1deg.bin`):
  Earth Gravitational Model 1996, a joint NIMA (now NGA) / NASA GSFC / Ohio State
  University model. As a work of the U.S. Government it is in the public domain
  and is distributed by NGA without restriction. The embedded file is the
  official 15-arcminute grid (`WW15MGH.DAC`) decimated to a 1-degree lattice;
  each sample is a genuine EGM96 undulation value at the corresponding node.
