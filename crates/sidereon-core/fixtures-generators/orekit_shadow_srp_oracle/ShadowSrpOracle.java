/* Bit-exact oracle for Orekit 13.1.10 SolarRadiationPressure lighting ratios,
 * cannonball SRP accelerations, and their FieldSpacecraftState<Gradient>
 * position and reflection-coefficient (C_R) partials.
 *
 * Evaluates Earth alone (spherical and WGS84), Moon alone, and Earth+Moon
 * (spherical and WGS84) with fixed-position Sun and Moon providers in GCRF
 * across sunlit, penumbra, umbra, annular, near-boundary (inside and outside
 * ANGULAR_MARGIN = 1e-10), near-axis, inside-Sun, and LEO/GNSS/GEO/distant
 * geometries. Every coordinate, value, and partial is recorded as raw IEEE 754
 * binary64 hex bits.
 *
 * For two-body Earth+Moon cases, records both Orekit's own getLightingRatio /
 * acceleration and a second Java evaluation that uses Orekit's exact
 * OccultationEngine angles and maskingRatio for each body while replacing only
 * Orekit's mutual-eclipse subtraction with the exact three-disc union.
 *
 * usage: java ShadowSrpOracle <orekit-data directory> <orekit-data commit>
 */

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.hipparchus.CalculusFieldElement;
import org.hipparchus.analysis.differentiation.Gradient;
import org.hipparchus.analysis.differentiation.GradientField;
import org.hipparchus.geometry.euclidean.threed.FieldRotation;
import org.hipparchus.geometry.euclidean.threed.FieldVector3D;
import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.util.FastMath;
import org.hipparchus.util.Precision;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient;
import org.orekit.forces.radiation.SolarRadiationPressure;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.propagation.FieldSpacecraftState;
import org.orekit.propagation.SpacecraftState;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.FieldAbsoluteDate;
import org.orekit.utils.AbsolutePVCoordinates;
import org.orekit.utils.Constants;
import org.orekit.utils.ExtendedPositionProvider;
import org.orekit.utils.ExtendedPositionProviderAdapter;
import org.orekit.utils.FieldAbsolutePVCoordinates;
import org.orekit.utils.FieldPVCoordinates;
import org.orekit.utils.FrameAdapter;
import org.orekit.utils.OccultationEngine;
import org.orekit.utils.PVCoordinates;

public final class ShadowSrpOracle {

    private static final double M_PER_KM = 1000.0;
    private static final double EARTH_RADIUS = 6371000.0;
    private static final double WGS84_FLATTENING = 1.0 / 298.257223563;
    private static final double MOON_RADIUS = 1737400.0;
    private static final double D_REF_KM = 149597870.7;
    private static final double D_REF = D_REF_KM * M_PER_KM;
    private static final double P_REF = 4.56e-6;
    private static final double CR = 1.3;
    private static final double AREA_TO_MASS = 0.02;
    private static final double MASS = 1.0;
    private static final double ANGULAR_MARGIN = 1.0e-10;

    private static final GradientField FIELD = GradientField.getField(3);
    private static final Gradient ZERO = FIELD.getZero();
    private static final Gradient ONE = FIELD.getOne();
    private static final GradientField FIELD_CR = GradientField.getField(4);
    private static final Gradient ONE_CR = FIELD_CR.getOne();

    private static Frame gcrf;

    private ShadowSrpOracle() {
    }

    private static final class FixedBody implements ExtendedPositionProvider {

        private final Vector3D position;

        FixedBody(final Vector3D position) {
            this.position = position;
        }

        @Override
        public Vector3D getPosition(final AbsoluteDate date, final Frame frame) {
            if (frame == gcrf) {
                return position;
            }
            return gcrf.getStaticTransformTo(frame, date).transformPosition(position);
        }

        @Override
        public <T extends CalculusFieldElement<T>> FieldVector3D<T> getPosition(
                final FieldAbsoluteDate<T> date, final Frame frame) {
            final FieldVector3D<T> p = new FieldVector3D<>(date.getField(), position);
            if (frame == gcrf) {
                return p;
            }
            return gcrf.getStaticTransformTo(frame, date).transformPosition(p);
        }
    }

    public static void main(final String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException(
                "usage: ShadowSrpOracle <orekit-data directory> <orekit-data commit>");
        }
        DataContext.getDefault().getDataProvidersManager()
            .addProvider(new DirectoryCrawler(new File(args[0])));
        gcrf = FramesFactory.getGCRF();

        final List<String> earthCases = buildEarthCases();
        final List<String> moonCases = buildMoonCases();
        final List<String> earthMoonCases = buildEarthMoonCases();

        final StringBuilder out = new StringBuilder();
        out.append("{\n");
        out.append("  \"_source\": {\n");
        out.append("    \"generator\": \"fixtures-generators/orekit_shadow_srp_oracle/generate.sh\",\n");
        out.append("    \"orekit\": \"").append(mavenVersion("org.orekit", "orekit")).append("\",\n");
        out.append("    \"hipparchus\": \"").append(mavenVersion("org.hipparchus", "hipparchus-core")).append("\",\n");
        out.append("    \"orekit_data_commit\": \"").append(args[1]).append("\",\n");
        out.append("    \"java\": \"").append(System.getProperty("java.runtime.version")).append("\"\n");
        out.append("  },\n");
        out.append("  \"sun_radius_m_bits\": ").append(hex(Constants.SUN_RADIUS)).append(",\n");
        out.append("  \"earth_radius_m_bits\": ").append(hex(EARTH_RADIUS)).append(",\n");
        out.append("  \"moon_radius_m_bits\": ").append(hex(MOON_RADIUS)).append(",\n");
        out.append("  \"wgs84_flattening_bits\": ").append(hex(WGS84_FLATTENING)).append(",\n");
        out.append("  \"d_ref_km_bits\": ").append(hex(D_REF_KM)).append(",\n");
        out.append("  \"p_ref_n_m2_bits\": ").append(hex(P_REF)).append(",\n");
        out.append("  \"cr_bits\": ").append(hex(CR)).append(",\n");
        out.append("  \"area_to_mass_m2_kg_bits\": ").append(hex(AREA_TO_MASS)).append(",\n");
        appendArray(out, "earth_cases", earthCases);
        out.append(",\n");
        appendArray(out, "moon_cases", moonCases);
        out.append(",\n");
        appendArray(out, "earth_moon_cases", earthMoonCases);
        out.append("\n}\n");
        System.out.print(out);
    }

    private static void appendArray(final StringBuilder out, final String key, final List<String> items) {
        out.append("  \"").append(key).append("\": [\n");
        for (int i = 0; i < items.size(); ++i) {
            out.append("    ").append(items.get(i)).append(i + 1 < items.size() ? ",\n" : "\n");
        }
        out.append("  ]");
    }

    private static String mavenVersion(final String group, final String artifact) throws IOException {
        final String resource = "/META-INF/maven/" + group + "/" + artifact + "/pom.properties";
        try (InputStream in = ShadowSrpOracle.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("no " + resource + " on the class path");
            }
            final Properties properties = new Properties();
            properties.load(in);
            return properties.getProperty("version");
        }
    }

    private static String hex(final double v) {
        return String.format("\"0x%016x\"", Double.doubleToRawLongBits(v));
    }

    private static String hexVec3(final Vector3D v) {
        return "[" + hex(v.getX()) + ", " + hex(v.getY()) + ", " + hex(v.getZ()) + "]";
    }

    private static String hexArr3(final double[] a) {
        return "[" + hex(a[0]) + ", " + hex(a[1]) + ", " + hex(a[2]) + "]";
    }

    private static String hexFieldVec3Val(final FieldVector3D<Gradient> v) {
        return "[" + hex(v.getX().getValue()) + ", " + hex(v.getY().getValue()) + ", " + hex(v.getZ().getValue()) + "]";
    }

    private static String hexFieldVec3Partials(final FieldVector3D<Gradient> v) {
        return "["
            + hexArr3(v.getX().getGradient()) + ", "
            + hexArr3(v.getY().getGradient()) + ", "
            + hexArr3(v.getZ().getGradient())
            + "]";
    }

    private static String hexFieldVec3CrPartial(final FieldVector3D<Gradient> v) {
        return "["
            + hex(v.getX().getPartialDerivative(3)) + ", "
            + hex(v.getY().getPartialDerivative(3)) + ", "
            + hex(v.getZ().getPartialDerivative(3))
            + "]";
    }

    private static Vector3D toMetres(final Vector3D km) {
        return new Vector3D(km.getX() * M_PER_KM, km.getY() * M_PER_KM, km.getZ() * M_PER_KM);
    }

    private static Vector3D perpendicular(final Vector3D axis, final Vector3D v) {
        return v.subtract(axis.scalarMultiply(Vector3D.dotProduct(v, axis))).normalize();
    }

    /** Place Sun at distance sunDistKm from satKm at separation sepRad from toEarth along side. */
    private static Vector3D sunAtSeparationKm(final Vector3D satKm, final Vector3D sideDir,
                                              final double sepRad, final double sunDistKm) {
        final Vector3D toEarth = satKm.negate().normalize();
        final Vector3D side = perpendicular(toEarth, sideDir);
        final Vector3D toSun = new Vector3D(FastMath.cos(sepRad), toEarth, FastMath.sin(sepRad), side);
        return satKm.add(toSun.scalarMultiply(sunDistKm));
    }

    /** Compute Earth limb radius from satKm toward sideDir under given flattening. */
    private static double earthLimbRadius(final Vector3D satKm, final Vector3D sideDir, final double flattening) {
        final Vector3D satM = toMetres(satKm);
        final Vector3D toEarth = satM.negate().normalize();
        final Vector3D side = perpendicular(toEarth, sideDir);
        final OneAxisEllipsoid earth = new OneAxisEllipsoid(EARTH_RADIUS, flattening, gcrf);
        final Vector3D limb = earth.pointOnLimb(satM, satM.add(toEarth.add(side).scalarMultiply(1.0e9)));
        return Vector3D.angle(limb.subtract(satM), toEarth);
    }

    /** Place Moon at moonDistKm from satKm at separation sepRad from toSun and azimuth azRad around toSun. */
    private static Vector3D moonAtSeparationKm(final Vector3D satKm, final Vector3D sunKm,
                                               final double sepRad, final double azRad, final double moonDistKm) {
        final Vector3D toSun = sunKm.subtract(satKm).normalize();
        final Vector3D toEarth = satKm.negate().normalize();
        final Vector3D ref = FastMath.abs(Vector3D.dotProduct(toSun, toEarth)) < 0.999
            ? toEarth
            : (FastMath.abs(toSun.getZ()) < 0.9 ? Vector3D.PLUS_K : Vector3D.PLUS_I);
        final Vector3D e1 = perpendicular(toSun, ref);
        final Vector3D e2 = Vector3D.crossProduct(toSun, e1);
        final Vector3D across = new Vector3D(FastMath.cos(azRad), e1, FastMath.sin(azRad), e2);
        final Vector3D toMoon = new Vector3D(FastMath.cos(sepRad), toSun, FastMath.sin(sepRad), across);
        return satKm.add(toMoon.scalarMultiply(moonDistKm));
    }

    // --------------------------------------------------------- Case builders

    private static List<String> buildEarthCases() {
        final List<String> cases = new ArrayList<>();
        final double sunDistKm = 1.495978707e8;
        final double aSun = FastMath.asin((Constants.SUN_RADIUS / M_PER_KM) / sunDistKm);
        final double[] flattenings = {0.0, WGS84_FLATTENING};
        final String[] flatNames = {"spherical", "wgs84"};

        final String[] regimes = {"leo", "gnss", "geo"};
        final double[] radiiKm = {6878.0, 26560.0, 42164.0};
        final Vector3D[] dirs = {
            new Vector3D(0.36, -0.48, 0.80).normalize(),
            new Vector3D(-0.42, 0.55, 0.72).normalize(),
            new Vector3D(0.31, 0.22, 0.925).normalize(),
        };
        final Vector3D sideDir = new Vector3D(0.53, 0.29, -0.61);

        for (int f = 0; f < flattenings.length; ++f) {
            final double flat = flattenings[f];
            final String fname = flatNames[f];

            for (int r = 0; r < regimes.length; ++r) {
                final Vector3D satKm = dirs[r].scalarMultiply(radiiKm[r]);
                final double bLimb = earthLimbRadius(satKm, sideDir, flat);
                final String prefix = regimes[r] + "_" + fname + "_";

                // Sunlit (dayside)
                cases.add(evalEarthCase(prefix + "sunlit", flat, satKm,
                    sunAtSeparationKm(satKm, sideDir, 2.1, sunDistKm)));
                // Deep umbra (off-axis)
                cases.add(evalEarthCase(prefix + "umbra", flat, satKm,
                    sunAtSeparationKm(satKm, sideDir, 0.35 * (bLimb - aSun), sunDistKm)));
                // Near-axis umbra
                cases.add(evalEarthCase(prefix + "near_axis_umbra", flat, satKm,
                    sunAtSeparationKm(satKm, sideDir, 1.0e-6, sunDistKm)));
                // Mid penumbra
                cases.add(evalEarthCase(prefix + "penumbra_mid", flat, satKm,
                    sunAtSeparationKm(satKm, sideDir, bLimb, sunDistKm)));
                // Deep penumbra
                cases.add(evalEarthCase(prefix + "penumbra_deep", flat, satKm,
                    sunAtSeparationKm(satKm, sideDir, bLimb - 0.72 * aSun, sunDistKm)));
                // Shallow penumbra
                cases.add(evalEarthCase(prefix + "penumbra_shallow", flat, satKm,
                    sunAtSeparationKm(satKm, sideDir, bLimb + 0.72 * aSun, sunDistKm)));
                // Near umbra boundary: inside and outside ANGULAR_MARGIN (1e-10)
                cases.add(evalEarthCase(prefix + "umbra_inside_margin", flat, satKm,
                    sunAtSeparationKm(satKm, sideDir, (bLimb - aSun) + 0.4e-10, sunDistKm)));
                cases.add(evalEarthCase(prefix + "umbra_outside_margin", flat, satKm,
                    sunAtSeparationKm(satKm, sideDir, (bLimb - aSun) + 2.5e-10, sunDistKm)));
                // Near penumbra boundary: inside and outside ANGULAR_MARGIN (1e-10)
                cases.add(evalEarthCase(prefix + "penumbra_inside_margin", flat, satKm,
                    sunAtSeparationKm(satKm, sideDir, (bLimb + aSun) - 2.5e-10, sunDistKm)));
                cases.add(evalEarthCase(prefix + "penumbra_outside_margin", flat, satKm,
                    sunAtSeparationKm(satKm, sideDir, (bLimb + aSun) - 0.4e-10, sunDistKm)));
            }

            // Distant satellite past umbra tip (annular regime: b < a)
            final Vector3D distSatKm = new Vector3D(-0.28, -0.53, 0.80).normalize().scalarMultiply(2.0e6);
            final double bDist = earthLimbRadius(distSatKm, sideDir, flat);
            final String dprefix = "distant_" + fname + "_";
            cases.add(evalEarthCase(dprefix + "sunlit", flat, distSatKm,
                sunAtSeparationKm(distSatKm, sideDir, 1.5, sunDistKm)));
            cases.add(evalEarthCase(dprefix + "annular_off_axis", flat, distSatKm,
                sunAtSeparationKm(distSatKm, sideDir, 0.35 * (aSun - bDist), sunDistKm)));
            cases.add(evalEarthCase(dprefix + "annular_near_axis", flat, distSatKm,
                sunAtSeparationKm(distSatKm, sideDir, 1.0e-6, sunDistKm)));
            cases.add(evalEarthCase(dprefix + "annular_partial", flat, distSatKm,
                sunAtSeparationKm(distSatKm, sideDir, aSun, sunDistKm)));
            cases.add(evalEarthCase(dprefix + "penumbra_inside_margin", flat, distSatKm,
                sunAtSeparationKm(distSatKm, sideDir, (bDist + aSun) - 2.5e-10, sunDistKm)));
            cases.add(evalEarthCase(dprefix + "penumbra_outside_margin", flat, distSatKm,
                sunAtSeparationKm(distSatKm, sideDir, (bDist + aSun) - 0.4e-10, sunDistKm)));
            // Very near annular axis (1e-11 rad off Earth-Sun line)
            cases.add(evalEarthCase(dprefix + "annular_very_near_axis", flat, distSatKm,
                sunAtSeparationKm(distSatKm, sideDir, 1.0e-11, sunDistKm)));

            // Inside the Sun dummy angles (|sat - sun| < R_sun)
            final Vector3D sunForInside = new Vector3D(D_REF_KM, 0.0, 0.0);
            final Vector3D satInsideSun = new Vector3D(D_REF_KM, 400000.0, 100000.0);
            cases.add(evalEarthCase("inside_sun_" + fname, flat, satInsideSun, sunForInside));
        }
        return cases;
    }

    private static List<String> buildMoonCases() {
        final List<String> cases = new ArrayList<>();
        final Vector3D sunKm = new Vector3D(1.2e8, -8.0e7, -3.5e7);

        // LEO satellite on the dayside of Earth
        final Vector3D satLeo = sunKm.normalize().scalarMultiply(6878.0);
        final double aLeo = FastMath.asin((Constants.SUN_RADIUS / M_PER_KM) / sunKm.subtract(satLeo).getNorm());
        final double dMoonTotal = 356000.0;
        final double bTotal = FastMath.asin((MOON_RADIUS / M_PER_KM) / dMoonTotal);
        final double dMoonAnn = 405000.0;
        final double bAnn = FastMath.asin((MOON_RADIUS / M_PER_KM) / dMoonAnn);

        cases.add(evalMoonCase("leo_moon_sunlit", satLeo, sunKm,
            moonAtSeparationKm(satLeo, sunKm, 0.4, 0.7, dMoonTotal)));
        cases.add(evalMoonCase("leo_moon_total", satLeo, sunKm,
            moonAtSeparationKm(satLeo, sunKm, 0.3 * (bTotal - aLeo), 0.7, dMoonTotal)));
        cases.add(evalMoonCase("leo_moon_total_near_axis", satLeo, sunKm,
            moonAtSeparationKm(satLeo, sunKm, 1.0e-6, 0.7, dMoonTotal)));
        cases.add(evalMoonCase("leo_moon_partial", satLeo, sunKm,
            moonAtSeparationKm(satLeo, sunKm, aLeo, 1.1, dMoonTotal)));
        cases.add(evalMoonCase("leo_moon_annular", satLeo, sunKm,
            moonAtSeparationKm(satLeo, sunKm, 0.35 * (aLeo - bAnn), 1.9, dMoonAnn)));
        cases.add(evalMoonCase("leo_moon_annular_near_axis", satLeo, sunKm,
            moonAtSeparationKm(satLeo, sunKm, 1.0e-6, 1.9, dMoonAnn)));
        cases.add(evalMoonCase("leo_moon_umbra_inside_margin", satLeo, sunKm,
            moonAtSeparationKm(satLeo, sunKm, (bTotal - aLeo) + 0.4e-10, 0.7, dMoonTotal)));
        cases.add(evalMoonCase("leo_moon_umbra_outside_margin", satLeo, sunKm,
            moonAtSeparationKm(satLeo, sunKm, (bTotal - aLeo) + 2.5e-10, 0.7, dMoonTotal)));
        cases.add(evalMoonCase("leo_moon_penumbra_inside_margin", satLeo, sunKm,
            moonAtSeparationKm(satLeo, sunKm, (bTotal + aLeo) - 2.5e-10, 0.7, dMoonTotal)));
        cases.add(evalMoonCase("leo_moon_penumbra_outside_margin", satLeo, sunKm,
            moonAtSeparationKm(satLeo, sunKm, (bTotal + aLeo) - 0.4e-10, 0.7, dMoonTotal)));

        // GNSS, GEO, distant
        final Vector3D satGnss = sunKm.normalize().scalarMultiply(26560.0);
        final double aGnss = FastMath.asin((Constants.SUN_RADIUS / M_PER_KM) / sunKm.subtract(satGnss).getNorm());
        cases.add(evalMoonCase("gnss_moon_partial", satGnss, sunKm,
            moonAtSeparationKm(satGnss, sunKm, 0.8 * aGnss, 2.3, 384400.0)));

        final Vector3D satGeo = sunKm.normalize().scalarMultiply(42164.0);
        final double aGeo = FastMath.asin((Constants.SUN_RADIUS / M_PER_KM) / sunKm.subtract(satGeo).getNorm());
        cases.add(evalMoonCase("geo_moon_annular", satGeo, sunKm,
            moonAtSeparationKm(satGeo, sunKm, 0.4 * (aGeo - bAnn), 0.4, dMoonAnn)));

        final Vector3D satDist = sunKm.normalize().scalarMultiply(2.0e6);
        final double aDist = FastMath.asin((Constants.SUN_RADIUS / M_PER_KM) / sunKm.subtract(satDist).getNorm());
        cases.add(evalMoonCase("distant_moon_partial", satDist, sunKm,
            moonAtSeparationKm(satDist, sunKm, 0.9 * aDist, 1.4, 384400.0)));

        return cases;
    }

    private static List<String> buildEarthMoonCases() {
        final List<String> cases = new ArrayList<>();
        final double sunDistKm = 1.495978707e8;
        final double aSun = FastMath.asin((Constants.SUN_RADIUS / M_PER_KM) / sunDistKm);
        final Vector3D sideDir = new Vector3D(0.53, 0.29, -0.61);
        final double[] flattenings = {0.0, WGS84_FLATTENING};
        final String[] flatNames = {"spherical", "wgs84"};

        final double bMoon356 = FastMath.asin((MOON_RADIUS / M_PER_KM) / 356000.0);
        final double bMoon384 = FastMath.asin((MOON_RADIUS / M_PER_KM) / 384400.0);
        final double bMoon405 = FastMath.asin((MOON_RADIUS / M_PER_KM) / 405000.0);
        final double dMoonClose = 1875.66;
        final double bMoonClose = FastMath.asin((MOON_RADIUS / M_PER_KM) / dMoonClose);

        for (int f = 0; f < flattenings.length; ++f) {
            final double flat = flattenings[f];
            final String fname = flatNames[f];

            // LEO
            final Vector3D satLeo = new Vector3D(0.36, -0.48, 0.80).normalize().scalarMultiply(6878.0);
            final double bLeo = earthLimbRadius(satLeo, sideDir, flat);
            final Vector3D sunLeoClear = sunAtSeparationKm(satLeo, sideDir, 2.0, sunDistKm);
            final Vector3D sunLeoPart = sunAtSeparationKm(satLeo, sideDir, bLeo + 0.15 * aSun, sunDistKm);
            final Vector3D sunLeoDeep = sunAtSeparationKm(satLeo, sideDir, bLeo - 0.55 * aSun, sunDistKm);
            final Vector3D sunLeoUmbra = sunAtSeparationKm(satLeo, sideDir, 0.3 * (bLeo - aSun), sunDistKm);
            final Vector3D sunLeoGrazingUmbra = sunAtSeparationKm(satLeo, sideDir, (bLeo - aSun) + 2.0e-9, sunDistKm);
            final Vector3D sunLeoGrazingPen = sunAtSeparationKm(satLeo, sideDir, (bLeo + aSun) - 2.5e-10, sunDistKm);
            // Equatorial LEO satellite so bLimb == bEq under both spherical and WGS84
            // (ensuring opposite-side grazing Moon is mutually disjoint from Orekit's
            // equatorial mutual disc as well as on the Sun).
            final Vector3D satLeoEq = new Vector3D(0.80, -0.60, 0.0).normalize().scalarMultiply(6878.0);
            final Vector3D sideEq = new Vector3D(0.60, 0.80, 0.0);
            final double bLeoEq = earthLimbRadius(satLeoEq, sideEq, flat);
            final Vector3D sunLeoEqGrazingUmbra = sunAtSeparationKm(satLeoEq, sideEq, (bLeoEq - aSun) + 2.0e-9, sunDistKm);
            final Vector3D sunLeoEqGrazingPen = sunAtSeparationKm(satLeoEq, sideEq, (bLeoEq + aSun) - 2.5e-10, sunDistKm);
            final String lp = "leo_" + fname + "_";

            cases.add(evalEarthMoonCase(lp + "sunlit_clear", flat, satLeo, sunLeoClear,
                moonAtSeparationKm(satLeo, sunLeoClear, 0.6, 1.0, 384400.0), "sunlit_clear"));
            cases.add(evalEarthMoonCase(lp + "earth_partial_moon_clear", flat, satLeo, sunLeoPart,
                moonAtSeparationKm(satLeo, sunLeoPart, 0.6, 1.0, 384400.0), "earth_partial_moon_clear"));
            cases.add(evalEarthMoonCase(lp + "earth_clear_moon_partial", flat, satLeo, sunLeoClear,
                moonAtSeparationKm(satLeo, sunLeoClear, 0.9 * aSun, 1.2, 360000.0), "earth_clear_moon_partial"));
            // True earth_clear + moon_total: sep = 0.3 * (bMoon356 - aSun) < bMoon356 - aSun - ANGULAR_MARGIN
            cases.add(evalEarthMoonCase(lp + "earth_clear_moon_total", flat, satLeo, sunLeoClear,
                moonAtSeparationKm(satLeo, sunLeoClear, 0.3 * (bMoon356 - aSun), 0.4, 356000.0), "earth_clear_moon_total"));
            cases.add(evalEarthMoonCase(lp + "earth_clear_moon_annular", flat, satLeo, sunLeoClear,
                moonAtSeparationKm(satLeo, sunLeoClear, 0.05 * aSun, 0.4, 405000.0), "earth_clear_moon_annular"));
            cases.add(evalEarthMoonCase(lp + "earth_umbra_moon_partial", flat, satLeo, sunLeoUmbra,
                moonAtSeparationKm(satLeo, sunLeoUmbra, 0.8 * aSun, 1.2, 360000.0), "earth_umbra_moon_partial"));
            // Both bodies total umbra simultaneously
            cases.add(evalEarthMoonCase(lp + "earth_total_moon_total", flat, satLeo, sunLeoUmbra,
                moonAtSeparationKm(satLeo, sunLeoUmbra, 0.3 * (bMoon356 - aSun), 0.4, 356000.0), "earth_total_moon_total"));
            // Truly disjoint simultaneous partial+partial (neither overlapping on Sun nor mutually)
            cases.add(evalEarthMoonCase(lp + "disjoint_partial", flat, satLeo, sunLeoPart,
                moonAtSeparationKm(satLeo, sunLeoPart, flat == 0.0 ? 1.35 * aSun : 1.85 * aSun, FastMath.PI, 384400.0), "disjoint_partial"));
            if (flat != 0.0) {
                // Disjoint on the Sun, yet overlapping Orekit's equatorial mutual disc outside the Sun
                cases.add(evalEarthMoonCase(lp + "disjoint_on_sun_mutual_overlap", flat, satLeo, sunLeoPart,
                    moonAtSeparationKm(satLeo, sunLeoPart, 1.35 * aSun, FastMath.PI, 384400.0), "mutual_overlap"));
            }
            // Overlapping partial+partial (triple intersection > 0)
            cases.add(evalEarthMoonCase(lp + "overlap_partial", flat, satLeo, sunLeoPart,
                moonAtSeparationKm(satLeo, sunLeoPart, 0.7 * aSun, 0.85, 360000.0), "mutual_overlap"));
            // Coplanar overlap: psi = 0 (Moon toward Earth) and psi = pi (Moon opposite Earth, overlapping)
            cases.add(evalEarthMoonCase(lp + "overlap_coplanar_psi0", flat, satLeo, sunLeoPart,
                moonAtSeparationKm(satLeo, sunLeoPart, 0.6 * aSun, 0.0, 360000.0), "mutual_overlap"));
            cases.add(evalEarthMoonCase(lp + "overlap_coplanar_psipi", flat, satLeo, sunLeoDeep,
                moonAtSeparationKm(satLeo, sunLeoDeep, 0.65 * aSun, FastMath.PI, 360000.0), "mutual_overlap"));
            // Combined total coverage where neither body alone covers the Sun
            cases.add(evalEarthMoonCase(lp + "overlap_combined_total", flat, satLeo, sunLeoDeep,
                moonAtSeparationKm(satLeo, sunLeoDeep, 0.4 * aSun, FastMath.PI, 350000.0), "mutual_overlap"));
            // Grazing umbra overshoot (l_earth < 0) + disjoint Moon (matches raw Orekit, negative lighting)
            cases.add(evalEarthMoonCase(lp + "grazing_umbra_overshoot_moon_disjoint", flat, satLeoEq, sunLeoEqGrazingUmbra,
                moonAtSeparationKm(satLeoEq, sunLeoEqGrazingUmbra, (aSun + bMoon384) - 1.0e-9, FastMath.PI, 384400.0), "grazing_umbra_disjoint"));
            // Grazing penumbra overshoot (l_earth > 1) + disjoint Moon (matches raw Orekit, lighting > 1)
            cases.add(evalEarthMoonCase(lp + "grazing_penumbra_overshoot_moon_disjoint", flat, satLeoEq, sunLeoEqGrazingPen,
                moonAtSeparationKm(satLeoEq, sunLeoEqGrazingPen, (aSun + bMoon384) - 2.0e-7, FastMath.PI, 384400.0), "grazing_pen_disjoint"));
            // Grazing umbra overshoot (l_earth < 0, != 0) + overlapping Moon (Orekit subtracts mutual)
            cases.add(evalEarthMoonCase(lp + "grazing_umbra_overshoot_moon_overlap", flat, satLeo, sunLeoGrazingUmbra,
                moonAtSeparationKm(satLeo, sunLeoGrazingUmbra, 0.5 * aSun, 0.0, 384400.0), "grazing_umbra_overlap"));
            // Grazing penumbra overshoot (l_earth > 1) + overlapping Moon (Orekit subtracts mutual)
            cases.add(evalEarthMoonCase(lp + "grazing_penumbra_overshoot_moon_overlap", flat, satLeoEq, sunLeoEqGrazingPen,
                moonAtSeparationKm(satLeoEq, sunLeoEqGrazingPen, 0.5 * aSun, 0.0, 384400.0), "grazing_pen_overlap"));

            // GNSS
            final Vector3D satGnss = new Vector3D(-0.42, 0.55, 0.72).normalize().scalarMultiply(26560.0);
            final double bGnss = earthLimbRadius(satGnss, sideDir, flat);
            final Vector3D sunGnssPart = sunAtSeparationKm(satGnss, sideDir, bGnss + 0.1 * aSun, sunDistKm);
            final String np = "gnss_" + fname + "_";
            cases.add(evalEarthMoonCase(np + "disjoint_partial", flat, satGnss, sunGnssPart,
                moonAtSeparationKm(satGnss, sunGnssPart, 1.35 * aSun, FastMath.PI - 0.1, 384400.0), "disjoint_partial"));
            cases.add(evalEarthMoonCase(np + "overlap_partial", flat, satGnss, sunGnssPart,
                moonAtSeparationKm(satGnss, sunGnssPart, 0.65 * aSun, 1.1, 360000.0), "mutual_overlap"));
            // Earth partial + annular Moon (c_m + b_m < a_sun, b_m < a_sun) overlapping Earth's disc
            cases.add(evalEarthMoonCase(np + "overlap_annular_moon", flat, satGnss, sunGnssPart,
                moonAtSeparationKm(satGnss, sunGnssPart, 0.5 * (aSun - bMoon405), 0.7, 405000.0),
                "earth_partial_moon_annular_overlap"));
            // Close Moon with grazing penumbra overshoot (l_moon > 1, != 1) overlapping Earth (Orekit subtracts mutual)
            cases.add(evalEarthMoonCase(np + "earth_partial_moon_grazing_penumbra_overshoot_overlap", flat, satGnss, sunGnssPart,
                moonAtSeparationKm(satGnss, sunGnssPart, (bMoonClose + aSun) - 1.5e-10, 0.0, dMoonClose), "moon_grazing_pen_overlap"));

            // GEO
            final Vector3D satGeo = new Vector3D(0.31, 0.22, 0.925).normalize().scalarMultiply(42164.0);
            final double bGeo = earthLimbRadius(satGeo, sideDir, flat);
            final Vector3D sunGeoPart = sunAtSeparationKm(satGeo, sideDir, bGeo + 0.2 * aSun, sunDistKm);
            final Vector3D sunGeoMargin = sunAtSeparationKm(satGeo, sideDir, (bGeo + aSun) - 2.5e-10, sunDistKm);
            final String gp = "geo_" + fname + "_";
            cases.add(evalEarthMoonCase(gp + "disjoint_partial", flat, satGeo, sunGeoPart,
                moonAtSeparationKm(satGeo, sunGeoPart, 1.4 * aSun, FastMath.PI + 0.15, 384400.0), "disjoint_partial"));
            cases.add(evalEarthMoonCase(gp + "overlap_partial", flat, satGeo, sunGeoPart,
                moonAtSeparationKm(satGeo, sunGeoPart, 0.75 * aSun, 0.9, 365000.0), "mutual_overlap"));
            cases.add(evalEarthMoonCase(gp + "near_margin_earth_moon_partial", flat, satGeo, sunGeoMargin,
                moonAtSeparationKm(satGeo, sunGeoMargin, 0.8 * aSun, 1.5, 370000.0), "mutual_overlap"));

            // Distant (2e6 km, Earth annular + Moon partial)
            final Vector3D satDist = new Vector3D(-0.28, -0.53, 0.80).normalize().scalarMultiply(2.0e6);
            final double bDist = earthLimbRadius(satDist, sideDir, flat);
            final Vector3D sunDistAnn = sunAtSeparationKm(satDist, sideDir, 0.3 * (aSun - bDist), sunDistKm);
            final Vector3D sunDistNearAxis = sunAtSeparationKm(satDist, sideDir, 1.0e-6, sunDistKm);
            final String dp = "distant_" + fname + "_";
            // Truly disjoint annular Earth + opposite-limb Moon (neither overlapping on Sun nor mutually)
            cases.add(evalEarthMoonCase(dp + "annular_moon_disjoint", flat, satDist, sunDistAnn,
                moonAtSeparationKm(satDist, sunDistAnn, 1.8 * aSun, FastMath.PI, 405000.0), "disjoint_partial"));
            // Opposite-limb annular Earth + Moon whose discs overlap on the Sun
            cases.add(evalEarthMoonCase(dp + "annular_moon_opposite_overlap", flat, satDist, sunDistAnn,
                moonAtSeparationKm(satDist, sunDistAnn, 1.25 * aSun, FastMath.PI, 405000.0), "mutual_overlap"));
            cases.add(evalEarthMoonCase(dp + "annular_moon_overlap", flat, satDist, sunDistAnn,
                moonAtSeparationKm(satDist, sunDistAnn, 0.6 * aSun, 0.6, 360000.0), "mutual_overlap"));
            cases.add(evalEarthMoonCase(dp + "annular_near_axis_moon_overlap", flat, satDist, sunDistNearAxis,
                moonAtSeparationKm(satDist, sunDistNearAxis, 0.55 * aSun, 0.9, 360000.0), "mutual_overlap"));
        }
        return cases;
    }

    // --------------------------------------------------------- Evaluators

    private static SpacecraftState makeState(final AbsoluteDate date, final Vector3D satM) {
        final Vector3D vel = Vector3D.ZERO;
        return new SpacecraftState(
            new AbsolutePVCoordinates(gcrf, date, new PVCoordinates(satM, vel))).withMass(MASS);
    }

    private static FieldSpacecraftState<Gradient> makeFieldState(final AbsoluteDate date, final Vector3D satM) {
        final FieldAbsoluteDate<Gradient> fDate = new FieldAbsoluteDate<>(FIELD, date);
        final FieldVector3D<Gradient> fPos = new FieldVector3D<>(
            Gradient.variable(3, 0, satM.getX()),
            Gradient.variable(3, 1, satM.getY()),
            Gradient.variable(3, 2, satM.getZ()));
        final FieldVector3D<Gradient> fVel = new FieldVector3D<>(FIELD, Vector3D.ZERO);
        final FieldAbsolutePVCoordinates<Gradient> fApv =
            new FieldAbsolutePVCoordinates<>(gcrf, fDate, new FieldPVCoordinates<>(fPos, fVel));
        return new FieldSpacecraftState<>(fApv).withMass(ONE);
    }

    private static FieldSpacecraftState<Gradient> makeFieldStateWithCr(final AbsoluteDate date, final Vector3D satM) {
        final FieldAbsoluteDate<Gradient> fDate = new FieldAbsoluteDate<>(FIELD_CR, date);
        final FieldVector3D<Gradient> fPos = new FieldVector3D<>(
            Gradient.variable(4, 0, satM.getX()),
            Gradient.variable(4, 1, satM.getY()),
            Gradient.variable(4, 2, satM.getZ()));
        final FieldVector3D<Gradient> fVel = new FieldVector3D<>(FIELD_CR, Vector3D.ZERO);
        final FieldAbsolutePVCoordinates<Gradient> fApv =
            new FieldAbsolutePVCoordinates<>(gcrf, fDate, new FieldPVCoordinates<>(fPos, fVel));
        return new FieldSpacecraftState<>(fApv).withMass(ONE_CR);
    }

    private static Gradient[] srpParamsWithCr(final SolarRadiationPressure srp,
                                              final FieldAbsoluteDate<Gradient> fDate) {
        final Gradient[] params = srp.getParameters(FIELD_CR, fDate);
        if (params.length != 2 || params[0].getValue() != 1.0 || params[1].getValue() != CR) {
            throw new IllegalStateException("unexpected SRP parameter layout");
        }
        params[1] = Gradient.variable(4, 3, CR);
        return params;
    }

    private static void assertFieldPosMatchesCr(final String label,
                                                final FieldVector3D<Gradient> a3,
                                                final FieldVector3D<Gradient> a4) {
        assertBitsEq(label + ".x.v", a3.getX().getValue(), a4.getX().getValue());
        assertBitsEq(label + ".y.v", a3.getY().getValue(), a4.getY().getValue());
        assertBitsEq(label + ".z.v", a3.getZ().getValue(), a4.getZ().getValue());
        for (int j = 0; j < 3; ++j) {
            assertBitsEq(label + ".x.g[" + j + "]", a3.getX().getPartialDerivative(j), a4.getX().getPartialDerivative(j));
            assertBitsEq(label + ".y.g[" + j + "]", a3.getY().getPartialDerivative(j), a4.getY().getPartialDerivative(j));
            assertBitsEq(label + ".z.g[" + j + "]", a3.getZ().getPartialDerivative(j), a4.getZ().getPartialDerivative(j));
        }
    }

    private static String evalEarthCase(final String name, final double flattening,
                                        final Vector3D satKm, final Vector3D sunKm) {
        final AbsoluteDate date = AbsoluteDate.J2000_EPOCH;
        final Vector3D satM = toMetres(satKm);
        final Vector3D sunM = toMetres(sunKm);
        final double dSunM = satM.subtract(sunM).getNorm();
        final boolean insideSun = dSunM < Constants.SUN_RADIUS;
        if (!insideSun && satM.getNorm() >= dSunM) {
            throw new IllegalStateException(name + ": d_earth >= d_sun (departure would fire)");
        }
        if (sunM.getNorm() < 2.0 * Constants.SUN_RADIUS) {
            throw new IllegalStateException(name + ": inside Orekit Sun-centered frame guard");
        }
        final ExtendedPositionProvider sunProv = new FixedBody(sunM);
        final OneAxisEllipsoid earth = new OneAxisEllipsoid(EARTH_RADIUS, flattening, gcrf);
        final IsotropicRadiationSingleCoefficient sc = new IsotropicRadiationSingleCoefficient(AREA_TO_MASS, CR);
        final SolarRadiationPressure srp = new SolarRadiationPressure(D_REF, P_REF, sunProv, earth, sc);

        final SpacecraftState state = makeState(date, satM);
        final FieldSpacecraftState<Gradient> fState = makeFieldState(date, satM);
        final FieldAbsoluteDate<Gradient> fDate = fState.getDate();
        final FieldSpacecraftState<Gradient> fStateCr = makeFieldStateWithCr(date, satM);
        final FieldAbsoluteDate<Gradient> fDateCr = fStateCr.getDate();

        final double lightingD = srp.getLightingRatio(state);
        final Vector3D accelD = srp.acceleration(state, srp.getParameters(date));
        final Gradient lightingF = srp.getLightingRatio(fState);
        final FieldVector3D<Gradient> accelF = srp.acceleration(fState, srp.getParameters(FIELD, fDate));
        final FieldVector3D<Gradient> accelFCr = srp.acceleration(fStateCr, srpParamsWithCr(srp, fDateCr));
        assertFieldPosMatchesCr(name, accelF, accelFCr);

        if (name.contains("annular")) {
            final OccultationEngine.OccultationAngles earthAnglesD =
                srp.getOccultingBodies().get(0).angles(state);
            final double aS = earthAnglesD.getOccultedApparentRadius();
            final double bE = earthAnglesD.getLimbRadius();
            final double cE = earthAnglesD.getSeparation();
            if (!(bE < aS)) {
                throw new IllegalStateException(name + ": expected annular Earth (b_e < a_sun), got b=" + bE + " a=" + aS);
            }
            if (!name.contains("annular_partial") && !(cE + bE < aS)) {
                throw new IllegalStateException(name + ": expected annular Earth (c_e + b_e < a_sun), got c+b=" + (cE + bE) + " a=" + aS);
            }
        }

        // Also evaluate Orekit's 2-body (Earth + clear Moon) path so public two-body
        // APIs can verify every earth_case against raw Orekit at 0 ULP without
        // triggering any d_body >= d_sun departure.
        final Vector3D clearMoonKm = moonAtSeparationKm(satKm, sunKm, 0.5 * FastMath.PI, 0.7, 384400.0);
        final Vector3D clearMoonM = toMetres(clearMoonKm);
        if (!insideSun && satM.subtract(clearMoonM).getNorm() >= dSunM) {
            throw new IllegalStateException(name + ": clearMoon d_moon >= d_sun");
        }
        final SolarRadiationPressure srpClearMoon = new SolarRadiationPressure(D_REF, P_REF, sunProv, earth, sc);
        srpClearMoon.addOccultingBody(new FixedBody(clearMoonM), MOON_RADIUS);
        final double clearMoonLightingD = srpClearMoon.getLightingRatio(state);
        final Vector3D clearMoonAccelD = srpClearMoon.acceleration(state, srpClearMoon.getParameters(date));
        final Gradient clearMoonLightingF = srpClearMoon.getLightingRatio(fState);
        final FieldVector3D<Gradient> clearMoonAccelF =
            srpClearMoon.acceleration(fState, srpClearMoon.getParameters(FIELD, fDate));
        final FieldVector3D<Gradient> clearMoonAccelFCr =
            srpClearMoon.acceleration(fStateCr, srpParamsWithCr(srpClearMoon, fDateCr));
        assertFieldPosMatchesCr(name + " clear_moon", clearMoonAccelF, clearMoonAccelFCr);

        final StringBuilder sb = new StringBuilder();
        sb.append("{\"name\": \"").append(name).append("\"");
        sb.append(", \"earth_flattening\": ").append(flattening);
        sb.append(", \"sat_km_bits\": ").append(hexVec3(satKm));
        sb.append(", \"sun_km_bits\": ").append(hexVec3(sunKm));
        sb.append(", \"lighting_double_bits\": ").append(hex(lightingD));
        sb.append(", \"accel_double_bits\": ").append(hexVec3(accelD));
        sb.append(", \"lighting_field_val_bits\": ").append(hex(lightingF.getValue()));
        sb.append(", \"lighting_field_grad_bits\": ").append(hexArr3(lightingF.getGradient()));
        sb.append(", \"accel_field_val_bits\": ").append(hexFieldVec3Val(accelF));
        sb.append(", \"accel_field_partials_bits\": ").append(hexFieldVec3Partials(accelF));
        sb.append(", \"clear_moon_km_bits\": ").append(hexVec3(clearMoonKm));
        sb.append(", \"clear_moon_lighting_double_bits\": ").append(hex(clearMoonLightingD));
        sb.append(", \"clear_moon_accel_double_bits\": ").append(hexVec3(clearMoonAccelD));
        sb.append(", \"clear_moon_lighting_field_val_bits\": ").append(hex(clearMoonLightingF.getValue()));
        sb.append(", \"clear_moon_lighting_field_grad_bits\": ").append(hexArr3(clearMoonLightingF.getGradient()));
        sb.append(", \"clear_moon_accel_field_val_bits\": ").append(hexFieldVec3Val(clearMoonAccelF));
        sb.append(", \"clear_moon_accel_field_partials_bits\": ").append(hexFieldVec3Partials(clearMoonAccelF));
        sb.append(", \"accel_field_wrt_cr_bits\": ").append(hexFieldVec3CrPartial(accelFCr));
        sb.append(", \"clear_moon_accel_field_wrt_cr_bits\": ").append(hexFieldVec3CrPartial(clearMoonAccelFCr));
        sb.append("}");
        return sb.toString();
    }

    private static String evalMoonCase(final String name, final Vector3D satKm,
                                       final Vector3D sunKm, final Vector3D moonKm) {
        final AbsoluteDate date = AbsoluteDate.J2000_EPOCH;
        final Vector3D satM = toMetres(satKm);
        final Vector3D sunM = toMetres(sunKm);
        final Vector3D moonM = toMetres(moonKm);
        if (satM.subtract(moonM).getNorm() >= satM.subtract(sunM).getNorm()) {
            throw new IllegalStateException(name + ": d_moon >= d_sun (departure would fire)");
        }
        if (sunM.getNorm() < 2.0 * Constants.SUN_RADIUS) {
            throw new IllegalStateException(name + ": inside Orekit Sun-centered frame guard");
        }
        final ExtendedPositionProvider sunProv = new FixedBody(sunM);
        final ExtendedPositionProvider moonProv = new FixedBody(moonM);
        final Frame moonFrame = new ExtendedPositionProviderAdapter(gcrf, moonProv, "moon-centred");
        final OneAxisEllipsoid moon = new OneAxisEllipsoid(MOON_RADIUS, 0.0, moonFrame);
        final IsotropicRadiationSingleCoefficient sc = new IsotropicRadiationSingleCoefficient(AREA_TO_MASS, CR);
        final SolarRadiationPressure srp = new SolarRadiationPressure(D_REF, P_REF, sunProv, moon, sc);

        final SpacecraftState state = makeState(date, satM);
        final FieldSpacecraftState<Gradient> fState = makeFieldState(date, satM);
        final FieldAbsoluteDate<Gradient> fDate = fState.getDate();
        final FieldSpacecraftState<Gradient> fStateCr = makeFieldStateWithCr(date, satM);
        final FieldAbsoluteDate<Gradient> fDateCr = fStateCr.getDate();

        final double lightingD = srp.getLightingRatio(state);
        final Vector3D accelD = srp.acceleration(state, srp.getParameters(date));
        final Gradient lightingF = srp.getLightingRatio(fState);
        final FieldVector3D<Gradient> accelF = srp.acceleration(fState, srp.getParameters(FIELD, fDate));
        final FieldVector3D<Gradient> accelFCr = srp.acceleration(fStateCr, srpParamsWithCr(srp, fDateCr));
        assertFieldPosMatchesCr(name, accelF, accelFCr);

        if (name.contains("annular")) {
            final OccultationEngine.OccultationAngles moonAnglesD =
                srp.getOccultingBodies().get(0).angles(state);
            final double aS = moonAnglesD.getOccultedApparentRadius();
            final double bM = moonAnglesD.getLimbRadius();
            final double cM = moonAnglesD.getSeparation();
            if (!(bM < aS) || !(cM + bM < aS)) {
                throw new IllegalStateException(name + ": expected annular Moon (b_m < a_sun && c_m + b_m < a_sun), got c=" + cM + " b=" + bM + " a=" + aS);
            }
        }

        final StringBuilder sb = new StringBuilder();
        sb.append("{\"name\": \"").append(name).append("\"");
        sb.append(", \"sat_km_bits\": ").append(hexVec3(satKm));
        sb.append(", \"sun_km_bits\": ").append(hexVec3(sunKm));
        sb.append(", \"moon_km_bits\": ").append(hexVec3(moonKm));
        sb.append(", \"lighting_double_bits\": ").append(hex(lightingD));
        sb.append(", \"accel_double_bits\": ").append(hexVec3(accelD));
        sb.append(", \"lighting_field_val_bits\": ").append(hex(lightingF.getValue()));
        sb.append(", \"lighting_field_grad_bits\": ").append(hexArr3(lightingF.getGradient()));
        sb.append(", \"accel_field_val_bits\": ").append(hexFieldVec3Val(accelF));
        sb.append(", \"accel_field_partials_bits\": ").append(hexFieldVec3Partials(accelF));
        sb.append(", \"accel_field_wrt_cr_bits\": ").append(hexFieldVec3CrPartial(accelFCr));
        sb.append("}");
        return sb.toString();
    }

    private static void assertBitsEq(final String label, final double a, final double b) {
        if (Double.doubleToRawLongBits(a) != Double.doubleToRawLongBits(b)) {
            throw new IllegalStateException(String.format(
                "%s: 0x%016x (%.17e) != 0x%016x (%.17e)",
                label, Double.doubleToRawLongBits(a), a, Double.doubleToRawLongBits(b), b));
        }
    }

    private static String evalEarthMoonCase(final String name, final double flattening,
                                            final Vector3D satKm, final Vector3D sunKm, final Vector3D moonKm,
                                            final String expectedKind) {
        final AbsoluteDate date = AbsoluteDate.J2000_EPOCH;
        final Vector3D satM = toMetres(satKm);
        final Vector3D sunM = toMetres(sunKm);
        final Vector3D moonM = toMetres(moonKm);
        final double dSunM = satM.subtract(sunM).getNorm();
        if (satM.getNorm() >= dSunM || satM.subtract(moonM).getNorm() >= dSunM) {
            throw new IllegalStateException(name + ": d_body >= d_sun (departure would fire)");
        }
        if (sunM.getNorm() < 2.0 * Constants.SUN_RADIUS) {
            throw new IllegalStateException(name + ": inside Orekit Sun-centered frame guard");
        }
        final ExtendedPositionProvider sunProv = new FixedBody(sunM);
        final ExtendedPositionProvider moonProv = new FixedBody(moonM);
        final OneAxisEllipsoid earth = new OneAxisEllipsoid(EARTH_RADIUS, flattening, gcrf);
        final IsotropicRadiationSingleCoefficient sc = new IsotropicRadiationSingleCoefficient(AREA_TO_MASS, CR);
        final SolarRadiationPressure srp = new SolarRadiationPressure(D_REF, P_REF, sunProv, earth, sc);
        srp.addOccultingBody(moonProv, MOON_RADIUS);

        final SpacecraftState state = makeState(date, satM);
        final FieldSpacecraftState<Gradient> fState = makeFieldState(date, satM);
        final FieldAbsoluteDate<Gradient> fDate = fState.getDate();
        final FieldSpacecraftState<Gradient> fStateCr = makeFieldStateWithCr(date, satM);
        final FieldAbsoluteDate<Gradient> fDateCr = fStateCr.getDate();

        // 1. Raw Orekit evaluation
        final double[] paramsD = srp.getParameters(date);
        final Gradient[] paramsF = srp.getParameters(FIELD, fDate);
        final Gradient[] paramsFCr = srpParamsWithCr(srp, fDateCr);
        final double orekitLightingD = srp.getLightingRatio(state);
        final Vector3D orekitAccelD = srp.acceleration(state, paramsD);
        final Gradient orekitLightingF = srp.getLightingRatio(fState);
        final FieldVector3D<Gradient> orekitAccelF = srp.acceleration(fState, paramsF);
        final FieldVector3D<Gradient> orekitAccelFCr = srp.acceleration(fStateCr, paramsFCr);
        assertFieldPosMatchesCr(name + " orekit", orekitAccelF, orekitAccelFCr);

        // Detect whether Orekit's mutual subtraction fired using Orekit's exact tests:
        // lightingRatioI != 0.0, lightingRatioJ != 0.0, lightingRatioJ != 1.0,
        // and mutual maskingRatioIJ != 1.0.
        final List<OccultationEngine> engines = srp.getOccultingBodies();
        final OccultationEngine.OccultationAngles earthAnglesD = engines.get(0).angles(state);
        final OccultationEngine.OccultationAngles moonAnglesD = engines.get(1).angles(state);
        final OccultationEngine.OccultationAngles mutualAnglesD =
            new OccultationEngine(new FrameAdapter(earth.getBodyFrame()), earth.getEquatorialRadius(),
                                  engines.get(1).getOcculting()).angles(state);
        final double leRaw = maskingRatioDouble(earthAnglesD);
        final double lmRaw = maskingRatioDouble(moonAnglesD);
        final boolean orekitSubtractsMutual = leRaw != 0.0 && lmRaw != 0.0 && lmRaw != 1.0
            && (mutualAnglesD.getSeparation() - mutualAnglesD.getLimbRadius()
                - mutualAnglesD.getOccultedApparentRadius() < -ANGULAR_MARGIN);

        // Verify case labels match actual geometry.
        switch (expectedKind) {
            case "sunlit_clear":
                if (leRaw != 1.0 || lmRaw != 1.0 || orekitSubtractsMutual) {
                    throw new IllegalStateException(name + ": expected sunlit_clear, got le=" + leRaw + " lm=" + lmRaw);
                }
                break;
            case "earth_partial_moon_clear":
                if (!(leRaw > 0.0 && leRaw < 1.0) || lmRaw != 1.0 || orekitSubtractsMutual) {
                    throw new IllegalStateException(name + ": expected earth_partial_moon_clear, got le=" + leRaw + " lm=" + lmRaw);
                }
                break;
            case "earth_clear_moon_partial":
                if (leRaw != 1.0 || !(lmRaw > 0.0 && lmRaw < 1.0) || orekitSubtractsMutual) {
                    throw new IllegalStateException(name + ": expected earth_clear_moon_partial, got le=" + leRaw + " lm=" + lmRaw + " mut=" + orekitSubtractsMutual);
                }
                break;
            case "earth_clear_moon_annular":
                if (leRaw != 1.0 || !(lmRaw > 0.0 && lmRaw < 1.0) || orekitSubtractsMutual
                    || !(moonAnglesD.getLimbRadius() < moonAnglesD.getOccultedApparentRadius())
                    || !(moonAnglesD.getSeparation() + moonAnglesD.getLimbRadius()
                         < moonAnglesD.getOccultedApparentRadius())) {
                    throw new IllegalStateException(name + ": expected earth_clear_moon_annular, got le=" + leRaw
                        + " lm=" + lmRaw + " mut=" + orekitSubtractsMutual
                        + " c_m=" + moonAnglesD.getSeparation() + " b_m=" + moonAnglesD.getLimbRadius()
                        + " a_s=" + moonAnglesD.getOccultedApparentRadius());
                }
                break;
            case "earth_partial_moon_annular_overlap":
                if (!(leRaw > 0.0 && leRaw < 1.0) || !(lmRaw > 0.0 && lmRaw < 1.0) || !orekitSubtractsMutual
                    || !(moonAnglesD.getLimbRadius() < moonAnglesD.getOccultedApparentRadius())
                    || !(moonAnglesD.getSeparation() + moonAnglesD.getLimbRadius()
                         < moonAnglesD.getOccultedApparentRadius())) {
                    throw new IllegalStateException(name + ": expected earth_partial_moon_annular_overlap, got le=" + leRaw
                        + " lm=" + lmRaw + " mut=" + orekitSubtractsMutual
                        + " c_m=" + moonAnglesD.getSeparation() + " b_m=" + moonAnglesD.getLimbRadius()
                        + " a_s=" + moonAnglesD.getOccultedApparentRadius());
                }
                break;
            case "earth_clear_moon_total":
                if (leRaw != 1.0 || lmRaw != 0.0 || orekitLightingD != 0.0 || orekitSubtractsMutual) {
                    throw new IllegalStateException(name + ": expected earth_clear_moon_total, got le=" + leRaw + " lm=" + lmRaw);
                }
                break;
            case "earth_umbra_moon_partial":
                if (leRaw != 0.0 || !(lmRaw > 0.0 && lmRaw < 1.0) || orekitLightingD != 0.0 || orekitSubtractsMutual) {
                    throw new IllegalStateException(name + ": expected earth_umbra_moon_partial, got le=" + leRaw + " lm=" + lmRaw);
                }
                break;
            case "earth_total_moon_total":
                if (leRaw != 0.0 || lmRaw != 0.0 || orekitLightingD != 0.0 || orekitSubtractsMutual) {
                    throw new IllegalStateException(name + ": expected earth_total_moon_total, got le=" + leRaw + " lm=" + lmRaw);
                }
                break;
            case "disjoint_partial":
                if (!(leRaw > 0.0 && leRaw < 1.0) || !(lmRaw > 0.0 && lmRaw < 1.0) || orekitSubtractsMutual) {
                    throw new IllegalStateException(name + ": expected disjoint_partial, got le=" + leRaw + " lm=" + lmRaw + " mut=" + orekitSubtractsMutual);
                }
                break;
            case "mutual_overlap":
                if (!orekitSubtractsMutual) {
                    throw new IllegalStateException(name + ": expected mutual_overlap, got mut=false le=" + leRaw + " lm=" + lmRaw);
                }
                break;
            case "grazing_umbra_disjoint":
                if (!(leRaw < 0.0) || !(lmRaw > 0.0 && lmRaw < 1.0) || orekitSubtractsMutual || !(orekitLightingD < 0.0)) {
                    throw new IllegalStateException(name + ": expected grazing_umbra_disjoint, got le=" + leRaw + " lm=" + lmRaw + " mut=" + orekitSubtractsMutual + " out=" + orekitLightingD);
                }
                break;
            case "grazing_pen_disjoint":
                if (!(leRaw > 1.0) || !(lmRaw > 0.0 && lmRaw < 1.0) || orekitSubtractsMutual || !(orekitLightingD > 1.0)) {
                    throw new IllegalStateException(name + ": expected grazing_pen_disjoint, got le=" + leRaw + " lm=" + lmRaw + " mut=" + orekitSubtractsMutual + " out=" + orekitLightingD);
                }
                break;
            case "grazing_umbra_overlap":
                if (!(leRaw < 0.0) || !(lmRaw > 0.0 && lmRaw < 1.0) || !orekitSubtractsMutual) {
                    throw new IllegalStateException(name + ": expected grazing_umbra_overlap, got le=" + leRaw + " lm=" + lmRaw + " mut=" + orekitSubtractsMutual);
                }
                break;
            case "grazing_pen_overlap":
                if (!(leRaw > 1.0) || !(lmRaw > 0.0 && lmRaw < 1.0) || !orekitSubtractsMutual) {
                    throw new IllegalStateException(name + ": expected grazing_pen_overlap, got le=" + leRaw + " lm=" + lmRaw + " mut=" + orekitSubtractsMutual);
                }
                break;
            case "moon_grazing_pen_overlap":
                if (!(leRaw > 0.0 && leRaw < 1.0) || !(lmRaw > 1.0) || !orekitSubtractsMutual) {
                    throw new IllegalStateException(name + ": expected moon_grazing_pen_overlap, got le=" + leRaw + " lm=" + lmRaw + " mut=" + orekitSubtractsMutual);
                }
                break;
            default:
                throw new IllegalArgumentException("unknown expectedKind: " + expectedKind);
        }

        if (name.startsWith("distant_") && name.contains("_annular")) {
            final double aS = earthAnglesD.getOccultedApparentRadius();
            final double bE = earthAnglesD.getLimbRadius();
            final double cE = earthAnglesD.getSeparation();
            if (!(bE < aS) || !(cE + bE < aS)) {
                throw new IllegalStateException(name + ": expected annular Earth (b_e < a_sun && c_e + b_e < a_sun), got c=" + cE + " b=" + bE + " a=" + aS);
            }
        }

        // 2. Patched two-body evaluation: verbatim copy of Orekit's getLightingRatio
        //    and acceleration with only the mutual term replaced by the triple-disc area.
        final OccultationEngine.FieldOccultationAngles<Gradient> earthAnglesF = engines.get(0).angles(fState);
        final OccultationEngine.FieldOccultationAngles<Gradient> moonAnglesF = engines.get(1).angles(fState);

        final double unionLightingD = patchedGetLightingRatioDouble(satM, sunM, moonM, earthAnglesD, moonAnglesD);
        if (expectedKind.equals("earth_partial_moon_annular_overlap")
            && !(unionLightingD > (leRaw + lmRaw) - 1.0)) {
            throw new IllegalStateException(name + ": expected positive Earth-Moon overlap on the Sun, got union="
                + unionLightingD + " additive=" + ((leRaw + lmRaw) - 1.0));
        }
        final Vector3D unionAccelD = orekitSrpAccelDouble(state, sunM, unionLightingD, sc, paramsD);
        final Gradient unionLightingF = patchedGetLightingRatioField(fState.getPosition(), sunM, moonM,
                                                                     earthAnglesF, moonAnglesF);
        final FieldVector3D<Gradient> unionAccelF =
            orekitSrpAccelField(fState, sunM, unionLightingF, sc, paramsF);
        final Gradient unionLightingFCr = new Gradient(
            unionLightingF.getValue(),
            unionLightingF.getPartialDerivative(0),
            unionLightingF.getPartialDerivative(1),
            unionLightingF.getPartialDerivative(2),
            0.0);
        final FieldVector3D<Gradient> unionAccelFCr =
            orekitSrpAccelField(fStateCr, sunM, unionLightingFCr, sc, paramsFCr);
        assertFieldPosMatchesCr(name + " union", unionAccelF, unionAccelFCr);

        if (!orekitSubtractsMutual) {
            assertBitsEq(name + " ld", unionLightingD, orekitLightingD);
            assertBitsEq(name + " lf.v", unionLightingF.getValue(), orekitLightingF.getValue());
            final double[] ulg = unionLightingF.getGradient();
            final double[] olg = orekitLightingF.getGradient();
            for (int i = 0; i < 3; ++i) {
                assertBitsEq(name + " lf.g[" + i + "]", ulg[i], olg[i]);
            }
            assertBitsEq(name + " ad.x", unionAccelD.getX(), orekitAccelD.getX());
            assertBitsEq(name + " ad.y", unionAccelD.getY(), orekitAccelD.getY());
            assertBitsEq(name + " ad.z", unionAccelD.getZ(), orekitAccelD.getZ());
            assertBitsEq(name + " af.x.v", unionAccelF.getX().getValue(), orekitAccelF.getX().getValue());
            assertBitsEq(name + " af.y.v", unionAccelF.getY().getValue(), orekitAccelF.getY().getValue());
            assertBitsEq(name + " af.z.v", unionAccelF.getZ().getValue(), orekitAccelF.getZ().getValue());
            for (int j = 0; j < 3; ++j) {
                assertBitsEq(name + " af.x.g[" + j + "]", unionAccelF.getX().getGradient()[j], orekitAccelF.getX().getGradient()[j]);
                assertBitsEq(name + " af.y.g[" + j + "]", unionAccelF.getY().getGradient()[j], orekitAccelF.getY().getGradient()[j]);
                assertBitsEq(name + " af.z.g[" + j + "]", unionAccelF.getZ().getGradient()[j], orekitAccelF.getZ().getGradient()[j]);
            }
            assertBitsEq(name + " af.x.cr", unionAccelFCr.getX().getPartialDerivative(3), orekitAccelFCr.getX().getPartialDerivative(3));
            assertBitsEq(name + " af.y.cr", unionAccelFCr.getY().getPartialDerivative(3), orekitAccelFCr.getY().getPartialDerivative(3));
            assertBitsEq(name + " af.z.cr", unionAccelFCr.getZ().getPartialDerivative(3), orekitAccelFCr.getZ().getPartialDerivative(3));
        }

        final StringBuilder sb = new StringBuilder();
        sb.append("{\"name\": \"").append(name).append("\"");
        sb.append(", \"earth_flattening\": ").append(flattening);
        sb.append(", \"sat_km_bits\": ").append(hexVec3(satKm));
        sb.append(", \"sun_km_bits\": ").append(hexVec3(sunKm));
        sb.append(", \"moon_km_bits\": ").append(hexVec3(moonKm));
        sb.append(", \"orekit_subtracts_mutual\": ").append(orekitSubtractsMutual);
        sb.append(", \"orekit_lighting_double_bits\": ").append(hex(orekitLightingD));
        sb.append(", \"orekit_accel_double_bits\": ").append(hexVec3(orekitAccelD));
        sb.append(", \"orekit_lighting_field_val_bits\": ").append(hex(orekitLightingF.getValue()));
        sb.append(", \"orekit_lighting_field_grad_bits\": ").append(hexArr3(orekitLightingF.getGradient()));
        sb.append(", \"orekit_accel_field_val_bits\": ").append(hexFieldVec3Val(orekitAccelF));
        sb.append(", \"orekit_accel_field_partials_bits\": ").append(hexFieldVec3Partials(orekitAccelF));
        sb.append(", \"union_lighting_double_bits\": ").append(hex(unionLightingD));
        sb.append(", \"union_accel_double_bits\": ").append(hexVec3(unionAccelD));
        sb.append(", \"union_lighting_field_val_bits\": ").append(hex(unionLightingF.getValue()));
        sb.append(", \"union_lighting_field_grad_bits\": ").append(hexArr3(unionLightingF.getGradient()));
        sb.append(", \"union_accel_field_val_bits\": ").append(hexFieldVec3Val(unionAccelF));
        sb.append(", \"union_accel_field_partials_bits\": ").append(hexFieldVec3Partials(unionAccelF));
        sb.append(", \"orekit_accel_field_wrt_cr_bits\": ").append(hexFieldVec3CrPartial(orekitAccelFCr));
        sb.append(", \"union_accel_field_wrt_cr_bits\": ").append(hexFieldVec3CrPartial(unionAccelFCr));
        sb.append("}");
        return sb.toString();
    }

    // ----------------------------- Verbatim Orekit private copies + mutual patch

    /** Verbatim copy of Orekit SolarRadiationPressure.acceleration (double). */
    private static Vector3D orekitSrpAccelDouble(final SpacecraftState s, final Vector3D sunPosition,
                                                 final double ratio,
                                                 final IsotropicRadiationSingleCoefficient spacecraft,
                                                 final double[] parameters) {
        final double kRef = P_REF * D_REF * D_REF;
        final Vector3D position = s.getPosition();
        final Vector3D sunSatVector = position.subtract(sunPosition);
        final double r2 = sunSatVector.getNormSq();
        final double rawP = ratio * kRef / r2;
        final Vector3D flux = new Vector3D(rawP / FastMath.sqrt(r2), sunSatVector);
        return spacecraft.radiationPressureAcceleration(s, flux, parameters);
    }

    /** Verbatim copy of Orekit SolarRadiationPressure.acceleration (Field). */
    private static FieldVector3D<Gradient> orekitSrpAccelField(final FieldSpacecraftState<Gradient> s,
                                                               final Vector3D sunM, final Gradient ratio,
                                                               final IsotropicRadiationSingleCoefficient spacecraft,
                                                               final Gradient[] parameters) {
        final double kRef = P_REF * D_REF * D_REF;
        final FieldVector3D<Gradient> position = s.getPosition();
        final FieldVector3D<Gradient> sunPosition = new FieldVector3D<>(s.getDate().getField(), sunM);
        final FieldVector3D<Gradient> sunSatVector = position.subtract(sunPosition);
        final Gradient r2 = sunSatVector.getNormSq();
        final Gradient rawP = ratio.multiply(kRef).divide(r2);
        final FieldVector3D<Gradient> flux = new FieldVector3D<>(rawP.divide(r2.sqrt()), sunSatVector);
        return spacecraft.radiationPressureAcceleration(s, flux, parameters);
    }

    /** Verbatim copy of Orekit SolarRadiationPressure.maskingRatio (double). */
    private static double maskingRatioDouble(final OccultationEngine.OccultationAngles angles) {
        final double sunSatCentralBodyAngle = angles.getSeparation();
        final double alphaCentral = angles.getLimbRadius();
        final double alphaSun = angles.getOccultedApparentRadius();
        if (sunSatCentralBodyAngle - alphaCentral + alphaSun <= ANGULAR_MARGIN) {
            return 0.0;
        } else if (sunSatCentralBodyAngle - alphaCentral - alphaSun < -ANGULAR_MARGIN) {
            final double sEA2 = sunSatCentralBodyAngle * sunSatCentralBodyAngle;
            final double oo2sEA = 1.0 / (2. * sunSatCentralBodyAngle);
            final double aS2 = alphaSun * alphaSun;
            final double aE2 = alphaCentral * alphaCentral;
            final double aE2maS2 = aE2 - aS2;
            final double alpha1 = (sEA2 - aE2maS2) * oo2sEA;
            final double alpha2 = (sEA2 + aE2maS2) * oo2sEA;
            final double almost0 = Precision.SAFE_MIN;
            final double almost1 = FastMath.nextDown(1.0);
            final double a1oaS = FastMath.min(almost1, FastMath.max(-almost1, alpha1 / alphaSun));
            final double aS2ma12 = FastMath.max(almost0, aS2 - alpha1 * alpha1);
            final double a2oaE = FastMath.min(almost1, FastMath.max(-almost1, alpha2 / alphaCentral));
            final double aE2ma22 = FastMath.max(almost0, aE2 - alpha2 * alpha2);
            final double p1 = aS2 * FastMath.acos(a1oaS) - alpha1 * FastMath.sqrt(aS2ma12);
            final double p2 = aE2 * FastMath.acos(a2oaE) - alpha2 * FastMath.sqrt(aE2ma22);
            return 1. - (p1 + p2) / (FastMath.PI * aS2);
        } else {
            return 1.0;
        }
    }

    private static Gradient orekitMin(final double d, final Gradient f) {
        return f.getReal() > d ? f.newInstance(d) : f;
    }

    private static Gradient orekitMax(final double d, final Gradient f) {
        return f.getReal() <= d ? f.newInstance(d) : f;
    }

    /** Verbatim copy of Orekit SolarRadiationPressure.maskingRatio (Field). */
    private static Gradient maskingRatioField(final OccultationEngine.FieldOccultationAngles<Gradient> angles) {
        final Gradient occultedSatOcculting = angles.getSeparation();
        final Gradient alphaOcculting = angles.getLimbRadius();
        final Gradient alphaOcculted = angles.getOccultedApparentRadius();
        if (occultedSatOcculting.getReal() - alphaOcculting.getReal() + alphaOcculted.getReal() <= ANGULAR_MARGIN) {
            return ZERO;
        } else if (occultedSatOcculting.getReal() - alphaOcculting.getReal() - alphaOcculted.getReal() < -ANGULAR_MARGIN) {
            final Gradient sEA2 = occultedSatOcculting.multiply(occultedSatOcculting);
            final Gradient oo2sEA = occultedSatOcculting.multiply(2).reciprocal();
            final Gradient aS2 = alphaOcculted.multiply(alphaOcculted);
            final Gradient aE2 = alphaOcculting.multiply(alphaOcculting);
            final Gradient aE2maS2 = aE2.subtract(aS2);
            final Gradient alpha1 = sEA2.subtract(aE2maS2).multiply(oo2sEA);
            final Gradient alpha2 = sEA2.add(aE2maS2).multiply(oo2sEA);
            final double almost0 = Precision.SAFE_MIN;
            final double almost1 = FastMath.nextDown(1.0);
            final Gradient a1oaS = orekitMin(almost1, orekitMax(-almost1, alpha1.divide(alphaOcculted)));
            final Gradient aS2ma12 = orekitMax(almost0, aS2.subtract(alpha1.multiply(alpha1)));
            final Gradient a2oaE = orekitMin(almost1, orekitMax(-almost1, alpha2.divide(alphaOcculting)));
            final Gradient aE2ma22 = orekitMax(almost0, aE2.subtract(alpha2.multiply(alpha2)));
            final Gradient p1 = aS2.multiply(a1oaS.acos()).subtract(alpha1.multiply(aS2ma12.sqrt()));
            final Gradient p2 = aE2.multiply(a2oaE.acos()).subtract(alpha2.multiply(aE2ma22.sqrt()));
            return ONE.subtract(p1.add(p2).divide(aS2.multiply(aS2.getPi())));
        } else {
            return ONE;
        }
    }

    /** Verbatim copy of Orekit SolarRadiationPressure.getLightingRatio (double, n=2)
     *  with only the mutual-eclipse term replaced by adding tripleArea / (PI * alphaSunSq). */
    private static double patchedGetLightingRatioDouble(final Vector3D satM, final Vector3D sunM, final Vector3D moonM,
                                                        final OccultationEngine.OccultationAngles anglesEarth,
                                                        final OccultationEngine.OccultationAngles anglesMoon) {
        final double alphaSunSq = anglesEarth.getOccultedApparentRadius() * anglesEarth.getOccultedApparentRadius();

        double result = 0.0;

        // i = 0 (Earth)
        final double lightingRatio0 = maskingRatioDouble(anglesEarth);
        if (lightingRatio0 == 0.0) {
            return 0.0;
        }
        result += lightingRatio0;

        // j = 1 (Moon)
        final double lightingRatio1 = maskingRatioDouble(anglesMoon);
        if (lightingRatio1 == 0.0) {
            return 0.0;
        } else if (lightingRatio1 != 1) {
            if (lightingRatio0 != 1) {
                final double a = anglesEarth.getOccultedApparentRadius();
                final double bE = anglesEarth.getLimbRadius();
                final double cE = anglesEarth.getSeparation();
                final double bM = anglesMoon.getLimbRadius();
                final double cM = anglesMoon.getSeparation();
                final Vector3D psEarth = satM.subtract(sunM);
                final Vector3D psatMoon = Rotation.IDENTITY.applyTo(moonM.negate().add(satM));
                final double[] psi = positionAngleDouble(psEarth, satM, psatMoon);
                final double tripleArea = tripleDiscAreaDouble(new double[][] {
                    {0.0, 0.0, a},
                    {cE, 0.0, bE},
                    {cM * psi[0], cM * psi[1], bM},
                });
                final double correction = tripleArea / (FastMath.PI * alphaSunSq);
                result += correction;
            }
        }

        // i = 1 (Moon)
        result += lightingRatio1;

        // Final term (n - 1 = 1)
        result -= 1;

        return result;
    }

    /** Verbatim copy of Orekit SolarRadiationPressure.getLightingRatio (Field, n=2)
     *  with only the mutual-eclipse term replaced by adding tripleArea / (PI * alphaSunSq). */
    private static Gradient patchedGetLightingRatioField(final FieldVector3D<Gradient> satF, final Vector3D sunM,
                                                         final Vector3D moonM,
                                                         final OccultationEngine.FieldOccultationAngles<Gradient> anglesEarth,
                                                         final OccultationEngine.FieldOccultationAngles<Gradient> anglesMoon) {
        final Gradient alphaSunSq =
            anglesEarth.getOccultedApparentRadius().multiply(anglesEarth.getOccultedApparentRadius());

        Gradient result = FIELD.getZero();

        // i = 0 (Earth)
        final Gradient lightingRatio0 = maskingRatioField(anglesEarth);
        if (lightingRatio0.isZero()) {
            return ZERO;
        }
        result = result.add(lightingRatio0);

        // j = 1 (Moon)
        final Gradient lightingRatio1 = maskingRatioField(anglesMoon);
        if (lightingRatio1.isZero()) {
            return ZERO;
        } else if (lightingRatio1.getReal() != 1) {
            if (lightingRatio0.getReal() != 1) {
                final Gradient a = anglesEarth.getOccultedApparentRadius();
                final Gradient bE = anglesEarth.getLimbRadius();
                final Gradient cE = anglesEarth.getSeparation();
                final Gradient bM = anglesMoon.getLimbRadius();
                final Gradient cM = anglesMoon.getSeparation();
                final FieldVector3D<Gradient> sunF = new FieldVector3D<>(FIELD, sunM);
                final FieldVector3D<Gradient> moonF = new FieldVector3D<>(FIELD, moonM);
                final FieldVector3D<Gradient> psEarth = satF.subtract(sunF);
                final FieldVector3D<Gradient> psatMoon =
                    FieldRotation.getIdentity(FIELD).applyTo(moonF.negate().add(satF));
                final Gradient[] psi = positionAngleField(psEarth, satF, psatMoon);
                final Gradient tripleArea = tripleDiscAreaField(new Gradient[][] {
                    {ZERO, ZERO, a},
                    {cE, ZERO, bE},
                    {cM.multiply(psi[0]), cM.multiply(psi[1]), bM},
                });
                final Gradient correction = tripleArea.divide(alphaSunSq.multiply(alphaSunSq.getPi()));
                result = result.add(correction);
            }
        }

        // i = 1 (Moon)
        result = result.add(lightingRatio1);

        // Final term (n - 1 = 1)
        result = result.subtract(1);

        return result;
    }

    private static double[] positionAngleDouble(final Vector3D ps, final Vector3D psatE, final Vector3D psatM) {
        final Vector3D wE = Vector3D.crossProduct(ps, psatE);
        final Vector3D wM = Vector3D.crossProduct(ps, psatM);
        final double nE = wE.getNorm();
        final double nM = wM.getNorm();
        final double nS = ps.negate().getNorm();
        if (nE == 0.0 || nM == 0.0 || nS == 0.0) {
            return new double[] {1.0, 0.0};
        }
        final Vector3D we = wE.normalize();
        final Vector3D wm = wM.normalize();
        final Vector3D uSun = ps.negate().normalize();
        return new double[] {
            Vector3D.dotProduct(we, wm),
            Vector3D.dotProduct(Vector3D.crossProduct(we, wm), uSun),
        };
    }

    private static Gradient[] positionAngleField(final FieldVector3D<Gradient> ps,
                                                 final FieldVector3D<Gradient> psatE,
                                                 final FieldVector3D<Gradient> psatM) {
        final FieldVector3D<Gradient> wE = FieldVector3D.crossProduct(ps, psatE);
        final FieldVector3D<Gradient> wM = FieldVector3D.crossProduct(ps, psatM);
        final double nE = wE.getNorm().getReal();
        final double nM = wM.getNorm().getReal();
        final double nS = ps.negate().getNorm().getReal();
        if (nE == 0.0 || nM == 0.0 || nS == 0.0) {
            return new Gradient[] {ONE, ZERO};
        }
        final FieldVector3D<Gradient> we = wE.normalize();
        final FieldVector3D<Gradient> wm = wM.normalize();
        final FieldVector3D<Gradient> uSun = ps.negate().normalize();
        return new Gradient[] {
            FieldVector3D.dotProduct(we, wm),
            FieldVector3D.dotProduct(FieldVector3D.crossProduct(we, wm), uSun),
        };
    }

    // --- Three-disc intersection area (double) ---

    private static double[] discPairDistDeltaDouble(final double[] d1, final double[] d2) {
        final double d = FastMath.hypot(d2[0] - d1[0], d2[1] - d1[1]);
        final double[] small = d2[2] >= d1[2] ? d1 : d2;
        final double[] big = d2[2] >= d1[2] ? d2 : d1;
        final double xs = small[0], ys = small[1];
        final double xb = big[0], yb = big[1], rb = big[2];
        final double cs = FastMath.hypot(xs, ys);
        if (cs > 0.0 && rb > 0.0 && cs < 0.25 * rb) {
            final double cb = FastMath.hypot(xb, yb);
            if (cb >= 0.5 * rb && cb <= 2.0 * rb) {
                final double deltaB = cb - rb;
                final double ux = xb / cb;
                final double uy = yb / cb;
                final double sPar = xs * ux + ys * uy;
                final double sPerp = -xs * uy + ys * ux;
                final double q = deltaB - sPar;
                final double dMinusRb = (2.0 * rb * q + (q * q + sPerp * sPerp)) / (d + rb);
                return new double[] {d, dMinusRb};
            }
        }
        return new double[] {d, d - rb};
    }

    private static double[] heronFactorsDouble(final double r1, final double r2,
                                               final double d, final double dMinusRmax) {
        if (r2 >= r1) {
            final double p1 = r1 - dMinusRmax;
            final double p2 = r1 >= 0.5 * r2 ? d - (r2 - r1) : r1 + dMinusRmax;
            final double p3 = d + (r2 - r1);
            final double p4 = (d + r2) + r1;
            return new double[] {p1, p2, p3, p4};
        } else {
            final double p1 = r2 - dMinusRmax;
            final double p2 = d + (r1 - r2);
            final double p3 = r2 >= 0.5 * r1 ? d - (r1 - r2) : r2 + dMinusRmax;
            final double p4 = (d + r1) + r2;
            return new double[] {p1, p2, p3, p4};
        }
    }

    private static double thetaMinusSinDouble(final double theta) {
        if (FastMath.abs(theta) < 0.1) {
            final double t2 = theta * theta;
            return theta * t2 * (1.0 / 6.0
                + t2 * (-1.0 / 120.0
                    + t2 * (1.0 / 5040.0 + t2 * (-1.0 / 362880.0 + t2 / 39916800.0))));
        }
        return theta - FastMath.sin(theta);
    }

    private static double twoDiscAreaDouble(final double r1, final double r2,
                                            final double d, final double dMinusRmax) {
        final double rMin = FastMath.min(r1, r2);
        if (dMinusRmax >= rMin) {
            return 0.0;
        }
        if (r2 >= r1 && (r1 >= 0.5 * r2 && d <= r2 - r1 || r1 < 0.5 * r2 && dMinusRmax <= -r1)) {
            return FastMath.PI * r1 * r1;
        }
        if (r1 > r2 && (r2 >= 0.5 * r1 && d <= r1 - r2 || r2 < 0.5 * r1 && dMinusRmax <= -r2)) {
            return FastMath.PI * r2 * r2;
        }
        final double[] p = heronFactorsDouble(r1, r2, d, dMinusRmax);
        if (p[0] <= 0.0) {
            return 0.0;
        }
        if (p[1] <= 0.0) {
            return FastMath.PI * r1 * r1;
        }
        if (p[2] <= 0.0) {
            return FastMath.PI * r2 * r2;
        }
        final double s1 = FastMath.sqrt(p[0]);
        final double s2 = FastMath.sqrt(p[1]);
        final double s3 = FastMath.sqrt(p[2]);
        final double s4 = FastMath.sqrt(p[3]);
        final double alpha = 2.0 * FastMath.atan2(s1 * s3, s2 * s4);
        final double beta = 2.0 * FastMath.atan2(s1 * s2, s3 * s4);
        return 0.5 * r1 * r1 * thetaMinusSinDouble(2.0 * alpha)
             + 0.5 * r2 * r2 * thetaMinusSinDouble(2.0 * beta);
    }

    private static int splitArcDouble(final double center, final double halfWidth, final double[][] out) {
        final double pi = FastMath.PI;
        final double tau = 2.0 * pi;
        if (halfWidth <= 0.0) {
            return 0;
        }
        if (halfWidth >= pi) {
            out[0][0] = -pi; out[0][1] = pi;
            return 1;
        }
        final double lo = center - halfWidth;
        final double hi = center + halfWidth;
        if (lo >= hi) {
            return 0;
        }
        if (lo >= -pi && hi <= pi) {
            out[0][0] = lo; out[0][1] = hi;
            return 1;
        }
        if (lo < -pi) {
            final double loWrap = lo + tau;
            if (loWrap <= hi) {
                out[0][0] = -pi; out[0][1] = pi;
                return 1;
            }
            int count = 0;
            if (hi > -pi) {
                out[count][0] = -pi; out[count][1] = hi;
                ++count;
            }
            if (loWrap < pi) {
                out[count][0] = loWrap; out[count][1] = pi;
                ++count;
            }
            return count;
        } else {
            final double hiWrap = hi - tau;
            if (hiWrap >= lo) {
                out[0][0] = -pi; out[0][1] = pi;
                return 1;
            }
            int count = 0;
            if (hiWrap > -pi) {
                out[count][0] = -pi; out[count][1] = hiWrap;
                ++count;
            }
            if (lo < pi) {
                out[count][0] = lo; out[count][1] = pi;
                ++count;
            }
            return count;
        }
    }

    private static int intersectArcsDouble(final double[][] a, final int aLen,
                                           final double[][] b, final int bLen, final double[][] out) {
        int count = 0;
        for (int i = 0; i < aLen; ++i) {
            for (int j = 0; j < bLen; ++j) {
                final double lo = FastMath.max(a[i][0], b[j][0]);
                final double hi = FastMath.min(a[i][1], b[j][1]);
                if (hi > lo && count < out.length) {
                    out[count][0] = lo;
                    out[count][1] = hi;
                    ++count;
                }
            }
        }
        return count;
    }

    private static double[] circlePointLocalDouble(final double r, final double c,
                                                   final double rMinusC, final double s) {
        if (c == 0.0) {
            final org.hipparchus.util.SinCos sc = FastMath.sinCos(s);
            return new double[] {r * sc.cos(), r * sc.sin()};
        }
        final double sHalf = FastMath.sin(0.5 * s);
        return new double[] {
            rMinusC - 2.0 * r * sHalf * sHalf,
            r * FastMath.sin(s),
        };
    }

    private static double tripleDiscAreaDouble(final double[][] discs) {
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                final double[] dd = discPairDistDeltaDouble(discs[i], discs[j]);
                if (dd[1] >= FastMath.min(discs[i][2], discs[j][2])) {
                    return 0.0;
                }
            }
        }
        final double[][] kept = new double[3][];
        int keptLen = 0;
        for (int i = 0; i < 3; ++i) {
            boolean redundant = false;
            for (int j = 0; j < 3; ++j) {
                if (i == j) {
                    continue;
                }
                final double[] dd = discPairDistDeltaDouble(discs[i], discs[j]);
                final double ri = discs[i][2], rj = discs[j][2];
                final boolean contains = ri >= rj && (rj >= 0.5 * ri ? dd[0] <= ri - rj : dd[1] <= -rj);
                if (contains && (ri > rj || i > j)) {
                    redundant = true;
                    break;
                }
            }
            if (!redundant) {
                kept[keptLen++] = discs[i];
            }
        }
        if (keptLen == 1) {
            return FastMath.PI * kept[0][2] * kept[0][2];
        }
        if (keptLen == 2) {
            final double[] dd = discPairDistDeltaDouble(kept[0], kept[1]);
            return twoDiscAreaDouble(kept[0][2], kept[1][2], dd[0], dd[1]);
        }
        int minIdx = 0;
        for (int k = 1; k < 3; ++k) {
            if (kept[k][2] < kept[minIdx][2]) {
                minIdx = k;
            }
        }
        final double[] originDisc = kept[minIdx];
        final double ox = originDisc[0], oy = originDisc[1];
        final double pi = FastMath.PI;
        double total = 0.0;
        for (int i = 0; i < 3; ++i) {
            final double xi = kept[i][0], yi = kept[i][1], ri = kept[i][2];
            final double vx = ox - xi, vy = oy - yi;
            final double ci, ciMinusRi;
            if (i == minIdx) {
                ci = 0.0;
                ciMinusRi = -ri;
            } else {
                final double[] dd = discPairDistDeltaDouble(kept[i], originDisc);
                ci = dd[0];
                ciMinusRi = dd[1];
            }
            final double riMinusCi = -ciMinusRi;
            final double ePar0 = ci > 0.0 ? vx / ci : 1.0;
            final double ePar1 = ci > 0.0 ? vy / ci : 0.0;
            final double ePerp0 = ci > 0.0 ? -ePar1 : 0.0;
            final double ePerp1 = ci > 0.0 ? ePar0 : 1.0;

            double[][] arcs = {{-pi, pi}, {0, 0}, {0, 0}, {0, 0}};
            int arcsLen = 1;
            for (int j = 0; j < 3; ++j) {
                if (i == j) {
                    continue;
                }
                final double xj = kept[j][0], yj = kept[j][1], rj = kept[j][2];
                final double[] dd = discPairDistDeltaDouble(kept[i], kept[j]);
                final double[] p = heronFactorsDouble(ri, rj, dd[0], dd[1]);
                if (p[0] <= 0.0 || p[2] <= 0.0) {
                    arcsLen = 0;
                    break;
                }
                if (p[1] <= 0.0) {
                    continue;
                }
                final double alpha = 2.0 * FastMath.atan2(
                    FastMath.sqrt(p[0]) * FastMath.sqrt(p[2]),
                    FastMath.sqrt(p[1]) * FastMath.sqrt(p[3]));
                final double sPhi;
                if (j == minIdx && ci > 0.0) {
                    sPhi = 0.0;
                } else {
                    final double wx = xj - ox, wy = yj - oy;
                    final double uPar, uPerp;
                    if (FastMath.hypot(wx, wy) <= dd[0]) {
                        uPar = ci + (wx * ePar0 + wy * ePar1);
                        uPerp = wx * ePerp0 + wy * ePerp1;
                    } else {
                        final double dx = xj - xi, dy = yj - yi;
                        uPar = dx * ePar0 + dy * ePar1;
                        uPerp = dx * ePerp0 + dy * ePerp1;
                    }
                    sPhi = FastMath.atan2(uPerp, uPar);
                }
                final double[][] other = new double[2][2];
                final int otherLen = splitArcDouble(sPhi, alpha, other);
                final double[][] next = new double[4][2];
                arcsLen = intersectArcsDouble(arcs, arcsLen, other, otherLen, next);
                arcs = next;
                if (arcsLen == 0) {
                    break;
                }
            }
            for (int k = 0; k < arcsLen; ++k) {
                final double s1 = arcs[k][0], s2 = arcs[k][1];
                final double dt = s2 - s1;
                final double[] p1 = circlePointLocalDouble(ri, ci, riMinusCi, s1);
                final double[] p2 = circlePointLocalDouble(ri, ci, riMinusCi, s2);
                total += 0.5 * ri * ri * thetaMinusSinDouble(dt) + 0.5 * (p1[0] * p2[1] - p1[1] * p2[0]);
            }
        }
        return FastMath.max(0.0, total);
    }

    // --- Three-disc intersection area (Field<Gradient>) ---

    private static Gradient[] discPairDistDeltaField(final Gradient[] d1, final Gradient[] d2) {
        final Gradient d = d2[0].subtract(d1[0]).hypot(d2[1].subtract(d1[1]));
        final Gradient[] small = d2[2].getReal() >= d1[2].getReal() ? d1 : d2;
        final Gradient[] big = d2[2].getReal() >= d1[2].getReal() ? d2 : d1;
        final Gradient xs = small[0], ys = small[1];
        final Gradient xb = big[0], yb = big[1], rb = big[2];
        final Gradient cs = xs.hypot(ys);
        if (cs.getReal() > 0.0 && rb.getReal() > 0.0 && cs.getReal() < 0.25 * rb.getReal()) {
            final Gradient cb = xb.hypot(yb);
            if (cb.getReal() >= 0.5 * rb.getReal() && cb.getReal() <= 2.0 * rb.getReal()) {
                final Gradient deltaB = cb.subtract(rb);
                final Gradient ux = xb.divide(cb);
                final Gradient uy = yb.divide(cb);
                final Gradient sPar = xs.multiply(ux).add(ys.multiply(uy));
                final Gradient sPerp = xs.negate().multiply(uy).add(ys.multiply(ux));
                final Gradient q = deltaB.subtract(sPar);
                final Gradient dMinusRb = rb.multiply(2.0).multiply(q)
                    .add(q.multiply(q).add(sPerp.multiply(sPerp)))
                    .divide(d.add(rb));
                return new Gradient[] {d, dMinusRb};
            }
        }
        return new Gradient[] {d, d.subtract(rb)};
    }

    private static Gradient[] heronFactorsField(final Gradient r1, final Gradient r2,
                                                final Gradient d, final Gradient dMinusRmax) {
        if (r2.getReal() >= r1.getReal()) {
            final Gradient p1 = r1.subtract(dMinusRmax);
            final Gradient p2 = r1.getReal() >= 0.5 * r2.getReal()
                ? d.subtract(r2.subtract(r1)) : r1.add(dMinusRmax);
            final Gradient p3 = d.add(r2.subtract(r1));
            final Gradient p4 = d.add(r2).add(r1);
            return new Gradient[] {p1, p2, p3, p4};
        } else {
            final Gradient p1 = r2.subtract(dMinusRmax);
            final Gradient p2 = d.add(r1.subtract(r2));
            final Gradient p3 = r2.getReal() >= 0.5 * r1.getReal()
                ? d.subtract(r1.subtract(r2)) : r2.add(dMinusRmax);
            final Gradient p4 = d.add(r1).add(r2);
            return new Gradient[] {p1, p2, p3, p4};
        }
    }

    private static Gradient thetaMinusSinField(final Gradient theta) {
        if (FastMath.abs(theta.getReal()) < 0.1) {
            final Gradient t2 = theta.multiply(theta);
            final Gradient poly = t2.divide(39916800.0)
                .add(-1.0 / 362880.0).multiply(t2)
                .add(1.0 / 5040.0).multiply(t2)
                .add(-1.0 / 120.0).multiply(t2)
                .add(1.0 / 6.0);
            return theta.multiply(t2).multiply(poly);
        }
        return theta.subtract(theta.sin());
    }

    private static Gradient twoDiscAreaField(final Gradient r1, final Gradient r2,
                                             final Gradient d, final Gradient dMinusRmax) {
        final double rMin = FastMath.min(r1.getReal(), r2.getReal());
        if (dMinusRmax.getReal() >= rMin) {
            return ZERO;
        }
        if (r2.getReal() >= r1.getReal()
            && (r1.getReal() >= 0.5 * r2.getReal() && d.getReal() <= r2.getReal() - r1.getReal()
                || r1.getReal() < 0.5 * r2.getReal() && dMinusRmax.getReal() <= -r1.getReal())) {
            return ZERO.getPi().multiply(r1).multiply(r1);
        }
        if (r1.getReal() > r2.getReal()
            && (r2.getReal() >= 0.5 * r1.getReal() && d.getReal() <= r1.getReal() - r2.getReal()
                || r2.getReal() < 0.5 * r1.getReal() && dMinusRmax.getReal() <= -r2.getReal())) {
            return ZERO.getPi().multiply(r2).multiply(r2);
        }
        final Gradient[] p = heronFactorsField(r1, r2, d, dMinusRmax);
        if (p[0].getReal() <= 0.0) {
            return ZERO;
        }
        if (p[1].getReal() <= 0.0) {
            return ZERO.getPi().multiply(r1).multiply(r1);
        }
        if (p[2].getReal() <= 0.0) {
            return ZERO.getPi().multiply(r2).multiply(r2);
        }
        final Gradient s1 = p[0].sqrt();
        final Gradient s2 = p[1].sqrt();
        final Gradient s3 = p[2].sqrt();
        final Gradient s4 = p[3].sqrt();
        final Gradient alpha = s1.multiply(s3).atan2(s2.multiply(s4)).multiply(2.0);
        final Gradient beta = s1.multiply(s2).atan2(s3.multiply(s4)).multiply(2.0);
        return r1.multiply(r1).multiply(0.5).multiply(thetaMinusSinField(alpha.multiply(2.0)))
            .add(r2.multiply(r2).multiply(0.5).multiply(thetaMinusSinField(beta.multiply(2.0))));
    }

    private static int splitArcField(final Gradient center, final Gradient halfWidth, final Gradient[][] out) {
        final double pi = FastMath.PI;
        final double tau = 2.0 * pi;
        final Gradient negPi = ZERO.newInstance(-pi);
        final Gradient posPi = ZERO.newInstance(pi);
        if (halfWidth.getReal() <= 0.0) {
            return 0;
        }
        if (halfWidth.getReal() >= pi) {
            out[0][0] = negPi; out[0][1] = posPi;
            return 1;
        }
        final Gradient lo = center.subtract(halfWidth);
        final Gradient hi = center.add(halfWidth);
        if (lo.getReal() >= hi.getReal()) {
            return 0;
        }
        if (lo.getReal() >= -pi && hi.getReal() <= pi) {
            out[0][0] = lo; out[0][1] = hi;
            return 1;
        }
        if (lo.getReal() < -pi) {
            final Gradient loWrap = lo.add(tau);
            if (loWrap.getReal() <= hi.getReal()) {
                out[0][0] = negPi; out[0][1] = posPi;
                return 1;
            }
            int count = 0;
            if (hi.getReal() > -pi) {
                out[count][0] = negPi; out[count][1] = hi;
                ++count;
            }
            if (loWrap.getReal() < pi) {
                out[count][0] = loWrap; out[count][1] = posPi;
                ++count;
            }
            return count;
        } else {
            final Gradient hiWrap = hi.subtract(tau);
            if (hiWrap.getReal() >= lo.getReal()) {
                out[0][0] = negPi; out[0][1] = posPi;
                return 1;
            }
            int count = 0;
            if (hiWrap.getReal() > -pi) {
                out[count][0] = negPi; out[count][1] = hiWrap;
                ++count;
            }
            if (lo.getReal() < pi) {
                out[count][0] = lo; out[count][1] = posPi;
                ++count;
            }
            return count;
        }
    }

    private static int intersectArcsField(final Gradient[][] a, final int aLen,
                                          final Gradient[][] b, final int bLen, final Gradient[][] out) {
        int count = 0;
        for (int i = 0; i < aLen; ++i) {
            for (int j = 0; j < bLen; ++j) {
                final Gradient lo = a[i][0].getReal() >= b[j][0].getReal() ? a[i][0] : b[j][0];
                final Gradient hi = a[i][1].getReal() <= b[j][1].getReal() ? a[i][1] : b[j][1];
                if (hi.getReal() > lo.getReal() && count < out.length) {
                    out[count][0] = lo;
                    out[count][1] = hi;
                    ++count;
                }
            }
        }
        return count;
    }

    private static Gradient[] circlePointLocalField(final Gradient r, final Gradient c,
                                                    final Gradient rMinusC, final Gradient s) {
        if (c.getReal() == 0.0) {
            final org.hipparchus.util.FieldSinCos<Gradient> sc = s.sinCos();
            return new Gradient[] {r.multiply(sc.cos()), r.multiply(sc.sin())};
        }
        final Gradient sHalf = s.multiply(0.5).sin();
        return new Gradient[] {
            rMinusC.subtract(r.multiply(2.0).multiply(sHalf).multiply(sHalf)),
            r.multiply(s.sin()),
        };
    }

    private static Gradient tripleDiscAreaField(final Gradient[][] discs) {
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                final Gradient[] dd = discPairDistDeltaField(discs[i], discs[j]);
                if (dd[1].getReal() >= FastMath.min(discs[i][2].getReal(), discs[j][2].getReal())) {
                    return ZERO;
                }
            }
        }
        final Gradient[][] kept = new Gradient[3][];
        int keptLen = 0;
        for (int i = 0; i < 3; ++i) {
            boolean redundant = false;
            for (int j = 0; j < 3; ++j) {
                if (i == j) {
                    continue;
                }
                final Gradient[] dd = discPairDistDeltaField(discs[i], discs[j]);
                final double ri = discs[i][2].getReal(), rj = discs[j][2].getReal();
                final boolean contains = ri >= rj && (rj >= 0.5 * ri ? dd[0].getReal() <= ri - rj : dd[1].getReal() <= -rj);
                if (contains && (ri > rj || i > j)) {
                    redundant = true;
                    break;
                }
            }
            if (!redundant) {
                kept[keptLen++] = discs[i];
            }
        }
        if (keptLen == 1) {
            return ZERO.getPi().multiply(kept[0][2]).multiply(kept[0][2]);
        }
        if (keptLen == 2) {
            final Gradient[] dd = discPairDistDeltaField(kept[0], kept[1]);
            return twoDiscAreaField(kept[0][2], kept[1][2], dd[0], dd[1]);
        }
        int minIdx = 0;
        for (int k = 1; k < 3; ++k) {
            if (kept[k][2].getReal() < kept[minIdx][2].getReal()) {
                minIdx = k;
            }
        }
        final Gradient[] originDisc = kept[minIdx];
        final Gradient ox = originDisc[0], oy = originDisc[1];
        final Gradient negPi = ZERO.newInstance(-FastMath.PI);
        final Gradient posPi = ZERO.newInstance(FastMath.PI);
        Gradient total = ZERO;
        for (int i = 0; i < 3; ++i) {
            final Gradient xi = kept[i][0], yi = kept[i][1], ri = kept[i][2];
            final Gradient vx = ox.subtract(xi), vy = oy.subtract(yi);
            final Gradient ci, ciMinusRi;
            if (i == minIdx) {
                ci = ZERO;
                ciMinusRi = ri.negate();
            } else {
                final Gradient[] dd = discPairDistDeltaField(kept[i], originDisc);
                ci = dd[0];
                ciMinusRi = dd[1];
            }
            final Gradient riMinusCi = ciMinusRi.negate();
            final Gradient ePar0, ePar1, ePerp0, ePerp1;
            if (ci.getReal() > 0.0) {
                ePar0 = vx.divide(ci);
                ePar1 = vy.divide(ci);
                ePerp0 = ePar1.negate();
                ePerp1 = ePar0;
            } else {
                ePar0 = ONE; ePar1 = ZERO; ePerp0 = ZERO; ePerp1 = ONE;
            }

            Gradient[][] arcs = {{negPi, posPi}, {ZERO, ZERO}, {ZERO, ZERO}, {ZERO, ZERO}};
            int arcsLen = 1;
            for (int j = 0; j < 3; ++j) {
                if (i == j) {
                    continue;
                }
                final Gradient xj = kept[j][0], yj = kept[j][1], rj = kept[j][2];
                final Gradient[] dd = discPairDistDeltaField(kept[i], kept[j]);
                final Gradient[] p = heronFactorsField(ri, rj, dd[0], dd[1]);
                if (p[0].getReal() <= 0.0 || p[2].getReal() <= 0.0) {
                    arcsLen = 0;
                    break;
                }
                if (p[1].getReal() <= 0.0) {
                    continue;
                }
                final Gradient alpha = p[0].sqrt().multiply(p[2].sqrt())
                    .atan2(p[1].sqrt().multiply(p[3].sqrt())).multiply(2.0);
                final Gradient sPhi;
                if (j == minIdx && ci.getReal() > 0.0) {
                    sPhi = ZERO;
                } else {
                    final Gradient wx = xj.subtract(ox), wy = yj.subtract(oy);
                    final Gradient uPar, uPerp;
                    if (wx.hypot(wy).getReal() <= dd[0].getReal()) {
                        uPar = ci.add(wx.multiply(ePar0).add(wy.multiply(ePar1)));
                        uPerp = wx.multiply(ePerp0).add(wy.multiply(ePerp1));
                    } else {
                        final Gradient dx = xj.subtract(xi), dy = yj.subtract(yi);
                        uPar = dx.multiply(ePar0).add(dy.multiply(ePar1));
                        uPerp = dx.multiply(ePerp0).add(dy.multiply(ePerp1));
                    }
                    sPhi = uPerp.atan2(uPar);
                }
                final Gradient[][] other = new Gradient[2][2];
                final int otherLen = splitArcField(sPhi, alpha, other);
                final Gradient[][] next = new Gradient[4][2];
                arcsLen = intersectArcsField(arcs, arcsLen, other, otherLen, next);
                arcs = next;
                if (arcsLen == 0) {
                    break;
                }
            }
            for (int k = 0; k < arcsLen; ++k) {
                final Gradient s1 = arcs[k][0], s2 = arcs[k][1];
                final Gradient dt = s2.subtract(s1);
                final Gradient[] p1 = circlePointLocalField(ri, ci, riMinusCi, s1);
                final Gradient[] p2 = circlePointLocalField(ri, ci, riMinusCi, s2);
                final Gradient term = ri.multiply(ri).multiply(0.5).multiply(thetaMinusSinField(dt))
                    .add(p1[0].multiply(p2[1]).subtract(p1[1].multiply(p2[0])).multiply(0.5));
                total = total.add(term);
            }
        }
        return orekitMax(0.0, total);
    }
}
