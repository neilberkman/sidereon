// Orekit batch least-squares oracle for sidereon-core's force-parameter orbit
// fit (crates/sidereon-core/src/orbit_determination/parameters.rs).
//
// One GPS satellite of an in-repository SP3 product is fitted by Orekit's
// BatchLSEstimator (Gauss-Newton with a QR decomposer) with ParameterDrivers
// selected on the force models:
//   - the reflection coefficient C_R of cannonball solar radiation pressure
//     (SolarRadiationPressure with IsotropicRadiationSingleCoefficient);
//   - constant empirical accelerations along the radial, along-track and
//     cross-track axes (ParametricAcceleration with a degree-0
//     PolynomialAccelerationModel, directed by a QSW LofOffset);
//   - once-per-revolution empirical accelerations, the cosine and sine of the
//     argument of latitude along the same axes (ParametricAcceleration with
//     the ArgumentOfLatitudeHarmonic model below);
//   - ECOM2 empirical solar radiation pressure (org.orekit.forces.radiation.ECOM2,
//     used unchanged because G03 stays in full sunlight across the entire arc:
//     D0/Y0/B0, D2B1, and D4B1 replacing cannonball SRP).
// Orekit's HarmonicAccelerationModel is a harmonic of elapsed time,
// gamma sin(2 pi n (t - t0) / P + phi), with a phase parameter; the model
// fitted here is linear in a cosine and a sine of the argument of latitude,
// so it is written as an AccelerationModel. Orekit differentiates it, like
// every force, by automatic differentiation for the state transition matrix
// and the parameter Jacobian.
//
// The force models are the ones sidereon's Phase A composite evaluates, built
// from the same constants:
//   - central gravity with GM_EARTH;
//   - zonal harmonics J2 through J6 about the inertial Z axis
//     (HolmesFeatherstoneAttractionModel in GCRF with normalized C_n0 =
//     -J_n / sqrt(2n + 1));
//   - Sun and Moon point masses (ThirdBodyAttraction) at the GCRS positions
//     of sidereon's low-precision analytic series (Montenbruck & Gill,
//     "Satellite Orbits", sections 3.3.2 and 3.3.3, as carried in
//     src/astro/bodies/sun_moon.rs), evaluated at TT and rotated from mean of
//     date to GCRF by (P * B)^T (IAU 2006 precession at TDB and IAU 2006 frame
//     bias, matching sun_moon_gcrs_at_tdb_seconds), ported below with the same
//     operation order and StrictMath so the estimator comparison isolates the
//     fit rather than the 6.82e-13 rad frame-bias rounding convention between
//     Orekit's getMOD(IERS_2010) and sidereon's bias matrix;
//   - cannonball solar radiation pressure with P = 4.56e-6 N/m^2 at
//     1 AU = 149597870700 m, A/m = 0.02 m^2/kg (unit mass), flux scaled by
//     the satellite-Sun distance;
//   - the Schwarzschild term (Relativity).
// The arc is checked to stay in sunlight, so the two shadow models never
// differ.
//
// Observations are the SP3 ITRF positions rotated to GCRF by Orekit
// (IERS 2010 conventions, EOP from the orekit-data directory), weighted with
// sigma = 1 m per component. The initial guess is the first GCRF position and
// the derivative there of the degree-8 Lagrange polynomial through the first
// nine. Both are written to the fixture so the Rust test fits the same
// observations from the same guess.
//
// usage: java OrbitParameterOracle <orekit-data dir> <provenance json> <sp3 file> <satellite>
// Writes the fixture JSON to standard output.

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.hipparchus.CalculusFieldElement;
import org.hipparchus.geometry.euclidean.threed.FieldVector3D;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.linear.QRDecomposer;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.optim.nonlinear.vector.leastsquares.GaussNewtonOptimizer;
import org.orekit.attitudes.LofOffset;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.estimation.leastsquares.BatchLSEstimator;
import org.orekit.estimation.measurements.ObservableSatellite;
import org.orekit.estimation.measurements.Position;
import org.orekit.forces.ForceModel;
import org.orekit.forces.empirical.AccelerationModel;
import org.orekit.forces.empirical.ParametricAcceleration;
import org.orekit.forces.empirical.PolynomialAccelerationModel;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.NewtonianAttraction;
import org.orekit.forces.gravity.Relativity;
import org.orekit.forces.gravity.ThirdBodyAttraction;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.forces.gravity.potential.TideSystem;
import org.orekit.forces.radiation.ECOM2;
import org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient;
import org.orekit.forces.radiation.SolarRadiationPressure;
import org.orekit.frames.EOPHistory;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.LOFType;
import org.orekit.frames.StaticTransform;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.Propagator;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.FieldSpacecraftState;
import org.orekit.propagation.conversion.DormandPrince853IntegratorBuilder;
import org.orekit.propagation.conversion.NumericalPropagatorBuilder;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.FieldAbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.ExtendedPositionProvider;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.ParameterDriver;
import org.orekit.utils.PVCoordinates;
import org.orekit.utils.TimeStampedPVCoordinates;

public final class OrbitParameterOracle {

    // sidereon-core constants (src/astro/constants.rs), SI units.
    static final double GM_EARTH = 398_600.4418e9;
    static final double RE_EARTH = 6_378.137 * 1000.0;
    static final double[] J = {0.0, 0.0, 1.082_626_68e-3, -2.532_656_485_332_235_5e-6,
        -1.619_621_591_367e-6, -2.272_960_828_686_982e-7, 5.406_812_391_070_848e-7};
    static final double GM_SUN = 132_712_440_041.939_38e9;
    static final double GM_MOON = 4_902.800_066e9;
    static final double AU_M = 149_597_870.700 * 1000.0;
    static final double SOLAR_PRESSURE = 4.56e-6;
    static final double MEAN_EARTH_RADIUS = 6_371.0 * 1000.0;
    static final double MONTENBRUCK_AU_M = 149_597_870_691.0;
    static final double WGS84_A_M = 6_378.137 * 1000.0;
    static final double DEG_TO_RAD = Math.PI / 180.0;
    static final double ARCSEC_TO_RAD = 4.848_136_811_095_359_935_899_141e-6;
    static final double J2000_JD = 2_451_545.0;
    static final double DAYS_PER_JULIAN_CENTURY = 36_525.0;
    static final double SECONDS_PER_DAY = 86_400.0;

    static final double AREA_TO_MASS = 0.02;
    static final double CR_START = 1.0;
    static final double SIGMA_M = 1.0;
    static final double POSITION_TOLERANCE_M = 1.0e-6;
    static final double CONVERGENCE_THRESHOLD = 1.0e-3;
    static final String CR_DRIVER = "reflection coefficient";

    static final String[] AXES = {"radial", "along", "cross"};
    static final Vector3D[] QSW = {Vector3D.PLUS_I, Vector3D.PLUS_J, Vector3D.PLUS_K};

    private OrbitParameterOracle() {}

    public static void main(String[] args) throws IOException {
        Locale.setDefault(Locale.ROOT);
        DataContext.getDefault().getDataProvidersManager()
            .addProvider(new DirectoryCrawler(new File(args[0])));
        String provenance = args[1];
        String sp3Path = args[2];
        String satellite = args[3];

        Frame gcrf = FramesFactory.getGCRF();
        Frame itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, false);
        TimeScale gps = TimeScalesFactory.getGPS();
        List<Sample> samples = readSp3(sp3Path, satellite, gps);
        for (Sample sample : samples) {
            StaticTransform toGcrf = itrf.getStaticTransformTo(gcrf, sample.date);
            sample.gcrf = toGcrf.transformPosition(sample.itrf);
        }
        AbsoluteDate t0 = samples.get(0).date;

        SeriesBody sun = new SeriesBody(false, gcrf);
        SeriesBody moon = new SeriesBody(true, gcrf);
        OneAxisEllipsoid shadowBody = new OneAxisEllipsoid(MEAN_EARTH_RADIUS, 0.0, gcrf);

        Vector3D guessPosition = samples.get(0).gcrf;
        Vector3D guessVelocity = lagrangeVelocity(samples, 9);
        CartesianOrbit guess = new CartesianOrbit(
            new TimeStampedPVCoordinates(t0, guessPosition, guessVelocity), gcrf, GM_EARTH);

        // The arc must stay in sunlight: both shadow models then give 1.
        SolarRadiationPressure lightingCheck = new SolarRadiationPressure(AU_M, SOLAR_PRESSURE, sun,
            shadowBody, new IsotropicRadiationSingleCoefficient(AREA_TO_MASS, CR_START));
        for (Sample sample : samples) {
            SpacecraftState state = new SpacecraftState(new CartesianOrbit(
                new TimeStampedPVCoordinates(sample.date, sample.gcrf, guessVelocity), gcrf,
                GM_EARTH));
            double ratio = lightingCheck.getLightingRatio(state);
            if (ratio != 1.0) {
                throw new IllegalStateException("satellite in shadow at " + sample.date
                    + ": lighting ratio " + ratio);
            }
        }

        StringBuilder out = new StringBuilder();
        out.append("{\n");
        out.append("  \"provenance\": ").append(provenance).append(",\n");
        out.append("  \"sp3\": ").append(quote(new File(sp3Path).getName())).append(",\n");
        out.append("  \"satellite\": ").append(quote(satellite)).append(",\n");
        out.append("  \"force_model\": {\"gm_earth_m3_s2\": ").append(num(GM_EARTH))
            .append(", \"re_earth_m\": ").append(num(RE_EARTH))
            .append(", \"zonal_degrees\": [2, 3, 4, 5, 6]")
            .append(", \"gm_sun_m3_s2\": ").append(num(GM_SUN))
            .append(", \"gm_moon_m3_s2\": ").append(num(GM_MOON))
            .append(", \"srp_pressure_n_m2\": ").append(num(SOLAR_PRESSURE))
            .append(", \"srp_reference_distance_m\": ").append(num(AU_M))
            .append(", \"area_to_mass_m2_kg\": ").append(num(AREA_TO_MASS))
            .append(", \"cr_start\": ").append(num(CR_START))
            .append(", \"relativity\": true},\n");
        out.append("  \"estimator\": {\"optimizer\": \"GaussNewtonOptimizer(QRDecomposer(1e-11), false)\"")
            .append(", \"integrator\": \"DormandPrince853, position tolerance ")
            .append(num(POSITION_TOLERANCE_M)).append(" m\"")
            .append(", \"sigma_m\": ").append(num(SIGMA_M))
            .append(", \"parameters_convergence_threshold\": ").append(num(CONVERGENCE_THRESHOLD))
            .append("},\n");

        // Earth orientation the observations were rotated with, at 0h UTC of
        // the days around the arc.
        EOPHistory eop = FramesFactory.getEOPHistory(IERSConventions.IERS_2010, false);
        TimeScale utc = TimeScalesFactory.getUTC();
        out.append("  \"earth_orientation\": [\n");
        AbsoluteDate firstDay = new AbsoluteDate(t0.getComponents(utc).getDate(), utc).shiftedBy(-86400.0);
        for (int day = 0; day < 4; day++) {
            AbsoluteDate date = firstDay.shiftedBy(86400.0 * day);
            double[] nro = eop.getNonRotatinOriginNutationCorrection(date);
            out.append("    {\"utc\": ").append(quote(date.toString(utc)))
                .append(", \"xp_arcsec\": ").append(num(eop.getPoleCorrection(date).getXp() / ARCSEC_TO_RAD))
                .append(", \"yp_arcsec\": ").append(num(eop.getPoleCorrection(date).getYp() / ARCSEC_TO_RAD))
                .append(", \"ut1_minus_utc_s\": ").append(num(eop.getUT1MinusUTC(date)))
                .append(", \"dx_arcsec\": ").append(num(nro[0] / ARCSEC_TO_RAD))
                .append(", \"dy_arcsec\": ").append(num(nro[1] / ARCSEC_TO_RAD))
                .append("}").append(day < 3 ? ",\n" : "\n");
        }
        out.append("  ],\n");

        out.append("  \"observations\": [\n");
        for (int i = 0; i < samples.size(); i++) {
            Sample sample = samples.get(i);
            out.append("    {\"gps\": ").append(quote(sample.date.toString(gps)))
                .append(", \"tt_j2000_s\": ").append(num(ttSeconds(sample.date)))
                .append(", \"itrf_m\": ").append(vector(sample.itrf))
                .append(", \"gcrf_m\": ").append(vector(sample.gcrf))
                .append("}").append(i + 1 < samples.size() ? ",\n" : "\n");
        }
        out.append("  ],\n");
        out.append("  \"initial_guess\": {\"position_m\": ").append(vector(guessPosition))
            .append(", \"velocity_m_s\": ").append(vector(guessVelocity)).append("},\n");

        // Body positions and per-force accelerations at the initial guess, so
        // the Rust test can compare the force models one by one.
        out.append("  \"bodies\": [\n");
        for (int i = 0; i < samples.size(); i += 96) {
            AbsoluteDate date = samples.get(i).date;
            out.append("    {\"tt_j2000_s\": ").append(num(ttSeconds(date)))
                .append(", \"sun_m\": ").append(vector(sun.getPosition(date, gcrf)))
                .append(", \"moon_m\": ").append(vector(moon.getPosition(date, gcrf)))
                .append("},\n");
        }
        out.setLength(out.length() - 2);
        out.append("\n  ],\n");
        SpacecraftState guessState = new SpacecraftState(guess).withMass(1.0);
        Map<String, ForceModel> forces = new LinkedHashMap<>();
        forces.put("central", new NewtonianAttraction(GM_EARTH));
        forces.put("zonal", zonal(gcrf));
        forces.put("sun", new ThirdBodyAttraction(sun, "Sun", GM_SUN));
        forces.put("moon", new ThirdBodyAttraction(moon, "Moon", GM_MOON));
        forces.put("srp_unit_cr", new SolarRadiationPressure(AU_M, SOLAR_PRESSURE, sun, shadowBody,
            new IsotropicRadiationSingleCoefficient(AREA_TO_MASS, 1.0)));
        forces.put("relativity", new Relativity(GM_EARTH));
        for (int axis = 0; axis < 3; axis++) {
            forces.put("empirical." + AXES[axis] + ".constant", constant(gcrf, axis, t0, 1.0e-9));
            forces.put("empirical." + AXES[axis] + ".cos", harmonic(gcrf, axis, true, 1.0e-9));
            forces.put("empirical." + AXES[axis] + ".sin", harmonic(gcrf, axis, false, 1.0e-9));
        }
        ECOM2 guessEcom2 = new ECOM2(2, 1, 0.0, sun, MEAN_EARTH_RADIUS);
        double[] guessEcom2Params = {
            -1.0e-9, 3.0e-10, -4.0e-10,
            -1.0e-7, 2.0e-9, 1.0e-10, -5.0e-10, 2.0e-10,
            5.0e-10
        };
        for (int k = 0; k < guessEcom2Params.length; k++) {
            guessEcom2.getParametersDrivers().get(k).setValue(guessEcom2Params[k]);
        }
        forces.put("ecom2_d4b1", guessEcom2);
        out.append("  \"accelerations_at_guess_m_s2\": {\n");
        int written = 0;
        for (Map.Entry<String, ForceModel> entry : forces.entrySet()) {
            ForceModel model = entry.getValue();
            Vector3D acceleration = model.acceleration(guessState, model.getParameters(t0));
            out.append("    ").append(quote(entry.getKey())).append(": ").append(vector(acceleration))
                .append(++written < forces.size() ? ",\n" : "\n");
        }
        out.append("  },\n");

        String[][] cases = {
            {"state_cr", "srp.cr"},
            {"state_cr_rtn_constant", "srp.cr", "empirical.radial.constant",
                "empirical.along.constant", "empirical.cross.constant"},
            {"state_rtn_once_per_rev", "empirical.radial.constant", "empirical.radial.cos",
                "empirical.radial.sin", "empirical.along.constant", "empirical.along.cos",
                "empirical.along.sin", "empirical.cross.constant", "empirical.cross.cos",
                "empirical.cross.sin"},
            {"state_ecom2_d0b0", "ecom2.b0", "ecom2.d0", "ecom2.y0"},
            {"state_ecom2_d2b1", "ecom2.b0", "ecom2.b1c", "ecom2.b1s",
                "ecom2.d0", "ecom2.d2c", "ecom2.d2s", "ecom2.y0"},
            {"state_ecom2_d4b1", "ecom2.b0", "ecom2.b1c", "ecom2.b1s",
                "ecom2.d0", "ecom2.d2c", "ecom2.d4c", "ecom2.d2s", "ecom2.d4s", "ecom2.y0"},
        };
        out.append("  \"cases\": [\n");
        for (int c = 0; c < cases.length; c++) {
            String[] estimated = Arrays.copyOfRange(cases[c], 1, cases[c].length);
            runCase(out, cases[c][0], estimated, guess, samples, gcrf, sun, moon, shadowBody, t0);
            out.append(c + 1 < cases.length ? ",\n" : "\n");
        }
        out.append("  ]\n}\n");
        System.out.print(out);
    }

    static void runCase(StringBuilder out, String name, String[] estimated, CartesianOrbit guess,
                        List<Sample> samples, Frame gcrf, SeriesBody sun, SeriesBody moon,
                        OneAxisEllipsoid shadowBody, AbsoluteDate t0) {
        List<String> selected = Arrays.asList(estimated);
        NumericalPropagatorBuilder builder = new NumericalPropagatorBuilder(guess,
            new DormandPrince853IntegratorBuilder(1.0e-3, 300.0, POSITION_TOLERANCE_M),
            PositionAngleType.TRUE, 1.0e-3);
        builder.setMass(1.0);
        builder.addForceModel(zonal(gcrf));
        builder.addForceModel(new ThirdBodyAttraction(sun, "Sun", GM_SUN));
        builder.addForceModel(new ThirdBodyAttraction(moon, "Moon", GM_MOON));
        boolean isEcom2 = selected.stream().anyMatch(s -> s.startsWith("ecom2."));
        if (isEcom2) {
            int nD = selected.contains("ecom2.d4c") ? 2 : (selected.contains("ecom2.d2c") ? 1 : 0);
            int nB = selected.contains("ecom2.b1c") ? 1 : 0;
            // The arc is verified above to stay in full sunlight at every epoch,
            // so Orekit's ECOM2 is used unchanged.
            ECOM2 ecom2 = new ECOM2(nD, nB, 0.0, sun, MEAN_EARTH_RADIUS);
            for (ParameterDriver driver : ecom2.getParametersDrivers()) {
                driver.setSelected(false);
            }
            builder.addForceModel(ecom2);
        } else {
            builder.addForceModel(new SolarRadiationPressure(AU_M, SOLAR_PRESSURE, sun, shadowBody,
                new IsotropicRadiationSingleCoefficient(AREA_TO_MASS, CR_START)));
        }
        builder.addForceModel(new Relativity(GM_EARTH));
        for (int axis = 0; axis < 3; axis++) {
            if (selected.contains("empirical." + AXES[axis] + ".constant")) {
                builder.addForceModel(constant(gcrf, axis, t0, 0.0));
            }
            if (selected.contains("empirical." + AXES[axis] + ".cos")) {
                builder.addForceModel(harmonic(gcrf, axis, true, 0.0));
            }
            if (selected.contains("empirical." + AXES[axis] + ".sin")) {
                builder.addForceModel(harmonic(gcrf, axis, false, 0.0));
            }
        }
        // Scales: the normalized step that ends the iteration is
        // CONVERGENCE_THRESHOLD times these, 1 um in position, 1e-8 in C_R and
        // 1e-15 m/s^2 in an empirical acceleration, below 1e-5 of the formal
        // standard deviations and above the integration noise of the
        // estimates.
        for (ParameterDriver driver : builder.getPropagationParametersDrivers().getDrivers()) {
            String sidereonName = sidereonName(driver.getName());
            if (sidereonName != null && selected.contains(sidereonName)) {
                driver.setSelected(true);
                driver.setScale(sidereonName.equals("srp.cr") ? 1.0e-5 : 1.0e-12);
            }
        }

        BatchLSEstimator estimator = new BatchLSEstimator(
            new GaussNewtonOptimizer(new QRDecomposer(1.0e-11), false), builder);
        ObservableSatellite observable = new ObservableSatellite(0);
        for (Sample sample : samples) {
            estimator.addMeasurement(new Position(sample.date, sample.gcrf, SIGMA_M, 1.0, observable));
        }
        // Iteration log on standard error: RMS of the weighted residuals and
        // the estimated force parameters after each evaluation.
        estimator.setObserver((iteration, evaluation, orbits, orbital, propagation, measurement,
                               provider, lsp) -> {
            StringBuilder line = new StringBuilder(name).append(" iteration ").append(iteration)
                .append(" rms ").append(num(lsp.getRMS()));
            for (ParameterDriver driver : orbital.getDrivers()) {
                line.append(' ').append(driver.getName()).append('=').append(num(driver.getValue(t0)));
            }
            for (ParameterDriver driver : propagation.getDrivers()) {
                line.append(' ').append(sidereonName(driver.getName())).append('=')
                    .append(num(driver.getValue(t0)));
            }
            System.err.println(line);
        });
        estimator.setParametersConvergenceThreshold(CONVERGENCE_THRESHOLD);
        estimator.setMaxIterations(40);
        estimator.setMaxEvaluations(80);
        Propagator[] estimatedPropagators = estimator.estimate();

        List<String> order = new ArrayList<>(Arrays.asList("x", "y", "z", "vx", "vy", "vz"));
        List<? extends ParameterDriver> propagationDrivers =
            estimator.getPropagatorParametersDrivers(true).getDrivers();
        Map<String, Double> values = new LinkedHashMap<>();
        for (ParameterDriver driver : propagationDrivers) {
            String sidereonName = sidereonName(driver.getName());
            order.add(sidereonName);
            values.put(sidereonName, driver.getValue(t0));
        }
        RealMatrix covariance = estimator.getPhysicalCovariances(1.0e-14);
        int m = 3 * samples.size();
        int n = order.size();
        double chiSquare = estimator.getOptimum().getChiSquare();

        SpacecraftState initial = estimatedPropagators[0].getInitialState();
        PVCoordinates initialPv = initial.getPVCoordinates(gcrf);

        // Post-fit residuals, observed minus computed, in the QSW (RTN) frame
        // of the fitted trajectory.
        Propagator propagator = estimatedPropagators[0];
        double[] sumSquares = new double[3];
        StringBuilder residuals = new StringBuilder();
        for (int i = 0; i < samples.size(); i++) {
            Sample sample = samples.get(i);
            SpacecraftState state = propagator.propagate(sample.date);
            PVCoordinates pv = state.getPVCoordinates(gcrf);
            Vector3D difference = sample.gcrf.subtract(pv.getPosition());
            Vector3D radial = pv.getPosition().normalize();
            Vector3D cross = pv.getMomentum().normalize();
            Vector3D along = Vector3D.crossProduct(cross, radial);
            double[] rtn = {difference.dotProduct(radial), difference.dotProduct(along),
                difference.dotProduct(cross)};
            for (int k = 0; k < 3; k++) {
                sumSquares[k] += rtn[k] * rtn[k];
            }
            residuals.append("        [").append(num(rtn[0])).append(", ").append(num(rtn[1]))
                .append(", ").append(num(rtn[2])).append("]")
                .append(i + 1 < samples.size() ? ",\n" : "\n");
        }

        out.append("    {\n");
        out.append("      \"name\": ").append(quote(name)).append(",\n");
        out.append("      \"estimated\": ").append(stringList(Arrays.asList(estimated))).append(",\n");
        out.append("      \"iterations\": ").append(estimator.getIterationsCount()).append(",\n");
        out.append("      \"evaluations\": ").append(estimator.getEvaluationsCount()).append(",\n");
        out.append("      \"initial_position_m\": ").append(vector(initialPv.getPosition())).append(",\n");
        out.append("      \"initial_velocity_m_s\": ").append(vector(initialPv.getVelocity())).append(",\n");
        out.append("      \"parameters\": {");
        int index = 0;
        for (Map.Entry<String, Double> entry : values.entrySet()) {
            out.append(index++ == 0 ? "" : ", ").append(quote(entry.getKey())).append(": ")
                .append(num(entry.getValue()));
        }
        out.append("},\n");
        out.append("      \"parameter_order\": ").append(stringList(order)).append(",\n");
        out.append("      \"covariance_unscaled\": [\n");
        for (int row = 0; row < n; row++) {
            out.append("        [");
            for (int column = 0; column < n; column++) {
                out.append(column == 0 ? "" : ", ").append(num(covariance.getEntry(row, column)));
            }
            out.append("]").append(row + 1 < n ? ",\n" : "\n");
        }
        out.append("      ],\n");
        out.append("      \"residual_components\": ").append(m).append(",\n");
        out.append("      \"chi_square\": ").append(num(chiSquare)).append(",\n");
        out.append("      \"variance_factor\": ").append(num(chiSquare / (m - n))).append(",\n");
        out.append("      \"rms_m\": ").append(num(estimator.getOptimum().getRMS() * SIGMA_M)).append(",\n");
        out.append("      \"rtn_rms_m\": [").append(num(Math.sqrt(sumSquares[0] / samples.size())))
            .append(", ").append(num(Math.sqrt(sumSquares[1] / samples.size())))
            .append(", ").append(num(Math.sqrt(sumSquares[2] / samples.size()))).append("],\n");
        out.append("      \"rtn_residuals_m\": [\n").append(residuals).append("      ]\n");
        out.append("    }");
    }

    /** The sidereon parameter name of an Orekit driver, or null for others. */
    static String sidereonName(String driver) {
        if (driver.equals(CR_DRIVER)) {
            return "srp.cr";
        }
        if (driver.startsWith("empirical.")) {
            return driver.endsWith("[0]") ? driver.substring(0, driver.length() - 3) : driver;
        }
        String ecomPrefix = ECOM2.ECOM_COEFFICIENT + " ";
        if (driver.startsWith(ecomPrefix)) {
            String suffix = driver.substring(ecomPrefix.length());
            if (suffix.equals("B0")) {
                return "ecom2.b0";
            }
            if (suffix.equals("D0")) {
                return "ecom2.d0";
            }
            if (suffix.equals("Y0")) {
                return "ecom2.y0";
            }
            if (suffix.startsWith("Bcos")) {
                int idx = Integer.parseInt(suffix.substring(4));
                return "ecom2.b" + (2 * idx + 1) + "c";
            }
            if (suffix.startsWith("Bsin")) {
                int idx = Integer.parseInt(suffix.substring(4));
                return "ecom2.b" + (2 * idx + 1) + "s";
            }
            if (suffix.startsWith("Dcos")) {
                int idx = Integer.parseInt(suffix.substring(4));
                return "ecom2.d" + (2 * idx + 2) + "c";
            }
            if (suffix.startsWith("Dsin")) {
                int idx = Integer.parseInt(suffix.substring(4));
                return "ecom2.d" + (2 * idx + 2) + "s";
            }
        }
        return null;
    }

    static HolmesFeatherstoneAttractionModel zonal(Frame gcrf) {
        double[][] c = new double[7][];
        double[][] s = new double[7][];
        for (int degree = 0; degree <= 6; degree++) {
            c[degree] = new double[degree + 1];
            s[degree] = new double[degree + 1];
            if (degree >= 2) {
                c[degree][0] = -J[degree] / Math.sqrt(2.0 * degree + 1.0);
            }
        }
        NormalizedSphericalHarmonicsProvider provider = GravityFieldFactory.getNormalizedProvider(
            RE_EARTH, GM_EARTH, TideSystem.TIDE_FREE, c, s);
        return new HolmesFeatherstoneAttractionModel(gcrf, provider);
    }

    static ParametricAcceleration constant(Frame gcrf, int axis, AbsoluteDate t0, double value) {
        PolynomialAccelerationModel model =
            new PolynomialAccelerationModel("empirical." + AXES[axis] + ".constant", t0, 0);
        model.getParametersDrivers().get(0).setValue(value);
        return new ParametricAcceleration(QSW[axis], new LofOffset(gcrf, LOFType.QSW), model);
    }

    static ParametricAcceleration harmonic(Frame gcrf, int axis, boolean cosine, double value) {
        ArgumentOfLatitudeHarmonic model = new ArgumentOfLatitudeHarmonic(
            "empirical." + AXES[axis] + (cosine ? ".cos" : ".sin"), cosine);
        model.getParametersDrivers().get(0).setValue(value);
        return new ParametricAcceleration(QSW[axis], new LofOffset(gcrf, LOFType.QSW), model);
    }

    /**
     * Signed amplitude {@code a cos u} or {@code a sin u}, with {@code u} the argument of
     * latitude from the ascending node on the equator of the propagation frame:
     * {@code cos u = r_hat . n}, {@code sin u = r_hat . (h_hat x n)},
     * {@code n = (z x h_hat) / |z x h_hat|}.
     */
    static final class ArgumentOfLatitudeHarmonic implements AccelerationModel {
        private final boolean cosine;
        private final ParameterDriver driver;

        ArgumentOfLatitudeHarmonic(String name, boolean cosine) {
            this.cosine = cosine;
            this.driver = new ParameterDriver(name, 0.0, 1.0e-12, Double.NEGATIVE_INFINITY,
                Double.POSITIVE_INFINITY);
        }

        @Override
        public List<ParameterDriver> getParametersDrivers() {
            return List.of(driver);
        }

        @Override
        public double signedAmplitude(SpacecraftState state, double[] parameters) {
            Vector3D r = state.getPosition();
            Vector3D h = r.crossProduct(state.getPVCoordinates().getVelocity()).normalize();
            Vector3D node = Vector3D.PLUS_K.crossProduct(h).normalize();
            Vector3D radial = r.normalize();
            double value = cosine ? radial.dotProduct(node) : radial.dotProduct(h.crossProduct(node));
            return parameters[0] * value;
        }

        @Override
        public <T extends CalculusFieldElement<T>> T signedAmplitude(FieldSpacecraftState<T> state,
                                                                      T[] parameters) {
            FieldVector3D<T> r = state.getPosition();
            FieldVector3D<T> h = r.crossProduct(state.getPVCoordinates().getVelocity()).normalize();
            FieldVector3D<T> z = FieldVector3D.getPlusK(r.getX().getField());
            FieldVector3D<T> node = z.crossProduct(h).normalize();
            FieldVector3D<T> radial = r.normalize();
            T value = cosine ? radial.dotProduct(node) : radial.dotProduct(h.crossProduct(node));
            return parameters[0].multiply(value);
        }
    }

    /**
     * Sun or Moon at the GCRS position of sidereon's analytic series
     * (mean-of-date series at TT rotated to GCRF by (P * B)^T at TDB, matching
     * sun_moon_gcrs_at_tdb_seconds), metres in GCRF.
     */
    static final class SeriesBody implements ExtendedPositionProvider {
        private final boolean moon;
        private final Frame gcrf;

        SeriesBody(boolean moon, Frame gcrf) {
            this.moon = moon;
            this.gcrf = gcrf;
        }

        @Override
        public Vector3D getPosition(AbsoluteDate date, Frame frame) {
            double tdbSeconds = tdbFromTtJ2000Seconds(ttSeconds(date));
            double tTtCenturies =
                ttFromTdbJ2000Seconds(tdbSeconds) / (DAYS_PER_JULIAN_CENTURY * SECONDS_PER_DAY);
            double jdTdb = J2000_JD + tdbSeconds / SECONDS_PER_DAY;
            double[] mean = series(tTtCenturies, moon);
            double[] p = mat3Vec3Mul(meanOfDateToGcrsMatrix(jdTdb), mean);
            Vector3D position = new Vector3D(p[0], p[1], p[2]);
            return frame == gcrf ? position
                : gcrf.getStaticTransformTo(frame, date).transformPosition(position);
        }

        @Override
        public TimeStampedPVCoordinates getPVCoordinates(AbsoluteDate date, Frame frame) {
            // Velocity by a central difference; none of the force models here
            // reads it.
            Vector3D velocity = getPosition(date.shiftedBy(1.0), frame)
                .subtract(getPosition(date.shiftedBy(-1.0), frame)).scalarMultiply(0.5);
            return new TimeStampedPVCoordinates(date, getPosition(date, frame), velocity);
        }

        @Override
        public <T extends CalculusFieldElement<T>> FieldVector3D<T> getPosition(
                FieldAbsoluteDate<T> date, Frame frame) {
            // The dates of the propagation carry no derivatives with respect
            // to the estimated parameters.
            return new FieldVector3D<>(date.getField(), getPosition(date.toAbsoluteDate(), frame));
        }
    }

    /** TT seconds past J2000 (2000-01-01T12:00:00 TT). */
    static double ttSeconds(AbsoluteDate date) {
        return date.durationFrom(AbsoluteDate.J2000_EPOCH);
    }

    /** TDB - TT series (seconds) at t Julian centuries of TT past J2000. */
    static double tdbMinusTtSeries(double t) {
        return 0.001657 * StrictMath.sin(628.3076 * t + 6.2401)
            + 0.000022 * StrictMath.sin(575.3385 * t + 4.2970)
            + 0.000014 * StrictMath.sin(1256.6152 * t + 6.1969)
            + 0.000005 * StrictMath.sin(606.9777 * t + 4.0212)
            + 0.000005 * StrictMath.sin(52.9691 * t + 0.4444)
            + 0.000002 * StrictMath.sin(21.3299 * t + 5.5431)
            + 0.000010 * t * StrictMath.sin(628.3076 * t + 4.2490);
    }

    /** sidereon-core tdb_minus_tt_seconds. */
    static double tdbMinusTtSeconds(double ttSeconds) {
        return tdbMinusTtSeries(ttSeconds / (DAYS_PER_JULIAN_CENTURY * SECONDS_PER_DAY));
    }

    /** sidereon-core tdb_from_tt_j2000_seconds. */
    static double tdbFromTtJ2000Seconds(double ttSeconds) {
        return ttSeconds + tdbMinusTtSeconds(ttSeconds);
    }

    /** sidereon-core tt_from_tdb_j2000_seconds. */
    static double ttFromTdbJ2000Seconds(double tdbSeconds) {
        double first = tdbSeconds - tdbMinusTtSeconds(tdbSeconds);
        return tdbSeconds - tdbMinusTtSeconds(first);
    }

    /** sidereon-core compute_skyfield_precession_matrix_unchecked (IAU 2006 P). */
    static double[][] precessionMatrix(double jdTdb) {
        double eps0Arcsec = 84381.406;
        double t = (jdTdb - J2000_JD) / DAYS_PER_JULIAN_CENTURY;
        double psia = ((((-0.0000000951 * t + 0.000132851) * t - 0.00114045) * t - 1.0790069) * t
            + 5038.481507) * t;
        double omegaa = ((((0.0000003337 * t - 0.000000467) * t - 0.00772503) * t + 0.0512623) * t
            - 0.025754) * t + eps0Arcsec;
        double chia = ((((-0.0000000560 * t + 0.000170663) * t - 0.00121197) * t - 2.3814292) * t
            + 10.556403) * t;

        double eps0 = eps0Arcsec * ARCSEC_TO_RAD;
        double psiaRad = psia * ARCSEC_TO_RAD;
        double omegaaRad = omegaa * ARCSEC_TO_RAD;
        double chiaRad = chia * ARCSEC_TO_RAD;

        double sa = StrictMath.sin(eps0);
        double ca = StrictMath.cos(eps0);
        double sb = StrictMath.sin(-psiaRad);
        double cb = StrictMath.cos(-psiaRad);
        double sc = StrictMath.sin(-omegaaRad);
        double cc = StrictMath.cos(-omegaaRad);
        double sd = StrictMath.sin(chiaRad);
        double cd = StrictMath.cos(chiaRad);

        return new double[][] {
            {
                cd * cb - sb * sd * cc,
                cd * sb * ca + sd * cc * cb * ca - sa * sd * sc,
                cd * sb * sa + sd * cc * cb * sa + ca * sd * sc,
            },
            {
                -sd * cb - sb * cd * cc,
                -sd * sb * ca + cd * cc * cb * ca - sa * cd * sc,
                -sd * sb * sa + cd * cc * cb * sa + ca * cd * sc,
            },
            {sb * sc, -sc * cb * ca - sa * cc, -sc * cb * sa + cc * ca},
        };
    }

    /** sidereon-core build_icrs_to_j2000 (IAU 2006 frame bias B). */
    static double[][] frameBiasMatrix() {
        double xi0 = -0.0166170 * ARCSEC_TO_RAD;
        double eta0 = -0.0068192 * ARCSEC_TO_RAD;
        double da0 = -0.01460 * ARCSEC_TO_RAD;

        double yx = -da0;
        double zx = xi0;
        double xy = da0;
        double zy = eta0;
        double xz = -xi0;
        double yz = -eta0;

        return new double[][] {
            {1.0 - 0.5 * (yx * yx + zx * zx), xy, xz},
            {yx, 1.0 - 0.5 * (yx * yx + zy * zy), yz},
            {zx, zy, 1.0 - 0.5 * (zy * zy + zx * zx)},
        };
    }

    /** sidereon-core mean_of_date_to_gcrs_matrix_unchecked: (P * B)^T. */
    static double[][] meanOfDateToGcrsMatrix(double jdTdb) {
        double[][] p = precessionMatrix(jdTdb);
        double[][] b = frameBiasMatrix();
        double[][] pb = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double s = 0.0;
                for (int k = 0; k < 3; k++) {
                    s += p[i][k] * b[k][j];
                }
                pb[i][j] = s;
            }
        }
        double[][] rt = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rt[i][j] = pb[j][i];
            }
        }
        return rt;
    }

    /** sidereon-core mat3_vec3_mul_unchecked. */
    static double[] mat3Vec3Mul(double[][] r, double[] p) {
        double[] out = new double[3];
        for (int i = 0; i < 3; i++) {
            double s = 0.0;
            for (int j = 0; j < 3; j++) {
                s += r[i][j] * p[j];
            }
            out[i] = s;
        }
        return out;
    }

    /** sidereon-core sun_moon_eci_unchecked, same operations and order. */
    static double[] series(double t, boolean moon) {
        double[] f = astArgs(t);
        double eps = 23.439291 - 0.0130042 * t;
        double sine = StrictMath.sin(eps * DEG_TO_RAD);
        double cose = StrictMath.cos(eps * DEG_TO_RAD);
        if (!moon) {
            double ms = 357.5277233 + 35999.05034 * t;
            double ls = 280.460
                + 36000.770 * t
                + 1.914666471 * StrictMath.sin(ms * DEG_TO_RAD)
                + 0.019994643 * StrictMath.sin(2.0 * ms * DEG_TO_RAD);
            double rs = MONTENBRUCK_AU_M
                * (1.000140612
                    - 0.016708617 * StrictMath.cos(ms * DEG_TO_RAD)
                    - 0.000139589 * StrictMath.cos(2.0 * ms * DEG_TO_RAD));
            double sinl = StrictMath.sin(ls * DEG_TO_RAD);
            double cosl = StrictMath.cos(ls * DEG_TO_RAD);
            return new double[] {rs * cosl, rs * cose * sinl, rs * sine * sinl};
        }
        double lm = 218.32 + 481267.883 * t + 6.29 * StrictMath.sin(f[0])
            - 1.27 * StrictMath.sin(f[0] - 2.0 * f[3])
            + 0.66 * StrictMath.sin(2.0 * f[3])
            + 0.21 * StrictMath.sin(2.0 * f[0])
            - 0.19 * StrictMath.sin(f[1])
            - 0.11 * StrictMath.sin(2.0 * f[2]);
        double pm = 5.13 * StrictMath.sin(f[2]) + 0.28 * StrictMath.sin(f[0] + f[2])
            - 0.28 * StrictMath.sin(f[2] - f[0])
            - 0.17 * StrictMath.sin(f[2] - 2.0 * f[3]);
        double rm = WGS84_A_M
            / StrictMath.sin(
                (0.9508
                    + 0.0518 * StrictMath.cos(f[0])
                    + 0.0095 * StrictMath.cos(f[0] - 2.0 * f[3])
                    + 0.0078 * StrictMath.cos(2.0 * f[3])
                    + 0.0028 * StrictMath.cos(2.0 * f[0]))
                    * DEG_TO_RAD);
        double sinlm = StrictMath.sin(lm * DEG_TO_RAD);
        double coslm = StrictMath.cos(lm * DEG_TO_RAD);
        double sinp = StrictMath.sin(pm * DEG_TO_RAD);
        double cosp = StrictMath.cos(pm * DEG_TO_RAD);
        return new double[] {
            rm * cosp * coslm,
            rm * (cose * cosp * sinlm - sine * sinp),
            rm * (sine * cosp * sinlm + cose * sinp),
        };
    }

    /** Delaunay arguments, IAU 1980 coefficients, as sidereon-core ast_args. */
    static double[] astArgs(double t) {
        double[][] fc = {
            {134.96340251, 1717915923.2178, 31.8792, 0.051635, -0.00024470},
            {357.52910918, 129596581.0481, -0.5532, 0.000136, -0.00001149},
            {93.27209062, 1739527262.8478, -12.7512, -0.001037, 0.00000417},
            {297.85019547, 1602961601.2090, -6.3706, 0.006593, -0.00003169},
            {125.04455501, -6962890.2665, 7.4722, 0.007702, -0.00005939},
        };
        double[] tt = new double[4];
        tt[0] = t;
        for (int i = 1; i < 4; i++) {
            tt[i] = tt[i - 1] * t;
        }
        double[] f = new double[5];
        double twoPi = 2.0 * Math.PI;
        for (int i = 0; i < 5; i++) {
            double v = fc[i][0] * 3600.0;
            for (int j = 0; j < 4; j++) {
                v += fc[i][j + 1] * tt[j];
            }
            double r = (v * ARCSEC_TO_RAD) % twoPi;
            f[i] = r < 0.0 ? r + twoPi : r;
        }
        return f;
    }

    /** Derivative at the first sample of the Lagrange polynomial through the first count. */
    static Vector3D lagrangeVelocity(List<Sample> samples, int count) {
        AbsoluteDate t0 = samples.get(0).date;
        double[] t = new double[count];
        for (int i = 0; i < count; i++) {
            t[i] = samples.get(i).date.durationFrom(t0);
        }
        Vector3D velocity = Vector3D.ZERO;
        for (int j = 0; j < count; j++) {
            // d/dt of the j-th basis polynomial at t = t[0].
            double weight = 0.0;
            for (int k = 0; k < count; k++) {
                if (k == j) {
                    continue;
                }
                double term = 1.0 / (t[j] - t[k]);
                for (int l = 0; l < count; l++) {
                    if (l != j && l != k) {
                        term *= (t[0] - t[l]) / (t[j] - t[l]);
                    }
                }
                weight += term;
            }
            velocity = velocity.add(weight, samples.get(j).gcrf);
        }
        return velocity;
    }

    static final class Sample {
        final AbsoluteDate date;
        final Vector3D itrf;
        Vector3D gcrf;

        Sample(AbsoluteDate date, Vector3D itrf) {
            this.date = date;
            this.itrf = itrf;
        }
    }

    /** Positions of one satellite from an SP3-c/d file in GPS time, km to m. */
    static List<Sample> readSp3(String path, String satellite, TimeScale gps) throws IOException {
        List<Sample> samples = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            AbsoluteDate epoch = null;
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("* ")) {
                    String[] f = line.substring(1).trim().split("\\s+");
                    epoch = new AbsoluteDate(Integer.parseInt(f[0]), Integer.parseInt(f[1]),
                        Integer.parseInt(f[2]), Integer.parseInt(f[3]), Integer.parseInt(f[4]),
                        Double.parseDouble(f[5]), gps);
                } else if (line.startsWith("P" + satellite) && epoch != null) {
                    double x = Double.parseDouble(line.substring(4, 18).trim());
                    double y = Double.parseDouble(line.substring(18, 32).trim());
                    double z = Double.parseDouble(line.substring(32, 46).trim());
                    if (x == 0.0 && y == 0.0 && z == 0.0) {
                        continue;
                    }
                    samples.add(new Sample(epoch, new Vector3D(x * 1000.0, y * 1000.0, z * 1000.0)));
                }
            }
        }
        return samples;
    }

    static String num(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new IllegalStateException("non-finite value in fixture");
        }
        return Double.toString(value);
    }

    static String vector(Vector3D v) {
        return "[" + num(v.getX()) + ", " + num(v.getY()) + ", " + num(v.getZ()) + "]";
    }

    static String quote(String s) {
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    static String stringList(List<String> list) {
        StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < list.size(); i++) {
            b.append(i == 0 ? "" : ", ").append(quote(list.get(i)));
        }
        return b.append("]").toString();
    }
}
