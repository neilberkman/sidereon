# Third-Party Notices

`sidereon-core` is licensed under the MIT License, reproduced in `LICENSE`.

The crate contains or derives code and data from these permissively licensed
public sources:

- IERS Conventions software: the derived-work description and renamed Rust
  routines are in `src/tides/`; the intact terms are in
  `IERS-CONVENTIONS-SOFTWARE-LICENSE.txt` (official source accessed
  2026-07-20: https://iers-conventions.obspm.fr/content/chapter7/software/dehanttideinel/DEHANTTIDEINEL.F).
  This Sidereon derived work is neither distributed by nor endorsed by the IERS
  Conventions Center.
- ERFA/SOFA: precession/nutation conventions and coefficient tables; the
  applicable terms are in `ERFA-LICENSE.txt`.
- RTKLIB: the MLAMBDA/LAMBDA integer least-squares port; the applicable terms
  are in `RTKLIB-LICENSE.txt`.
- SciPy: trust-region least-squares routines arrive through the separately
  packaged `trust-region-least-squares` dependency, which carries the SciPy
  license and derivation notice.

No copyleft code is included.

## Orekit 13.1.10 and Hipparchus 4.0.3

The IERS 2010 Earth-orientation module includes Orekit's IERS coefficient
tables and nutation arguments. Its date, rotation and interpolation code
includes adaptations of the following Apache-2.0 sources.

- `AbsoluteDate.java`, `DateComponents.java`, `TimeComponents.java`,
  `UTCScale.java`, `UTCTAIOffset.java`, `UT1Scale.java`, `CIRFProvider.java`,
  `TIRFProvider.java`, `ITRFProvider.java`, `Transform.java`,
  `EOPHistory.java`, `IERSConventions.java`, `FundamentalNutationArguments.java`,
  `NutationCodec.java`, `PoissonSeries.java`, `PoissonSeriesParser.java`,
  `PolynomialParser.java`, `PolynomialNutation.java`, `SeriesTerm.java`,
  `TideTerm.java`, `LuniSolarTerm.java`, `PlanetaryTerm.java`,
  `GeneralTerm.java` and `NoFarPlanetsTerm.java`. Copyright 2002-2025 CS GROUP.
- `TimeOffset.java`. Copyright 2022-2025 Luc Maisonobe.
- `KinematicTransform.java`. Copyright 2022-2025 Romain Serra.
- `Rotation.java`, `Vector3D.java`, `HermiteInterpolator.java`, `FastMath.java`
  and `MathArrays.java`. Licensed to the Apache Software Foundation (ASF)
  under one or more contributor license agreements. Their headers refer
  to the Hipparchus NOTICE for copyright ownership and identify the
  modifications by the Hipparchus project.

See the [Apache License, Version 2.0](APACHE-2.0.txt), the
[Orekit NOTICE](OREKIT-NOTICE.txt), and the
[Hipparchus NOTICE](HIPPARCHUS-NOTICE.txt).
