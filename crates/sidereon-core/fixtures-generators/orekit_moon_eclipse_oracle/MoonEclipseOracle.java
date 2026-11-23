/* Oracle for the Earth and Moon shadow function of sidereon-core.
 *
 * Prints, as JSON, the lighting ratio of Orekit's SolarRadiationPressure for
 * satellite, Sun and Moon positions that the Rust tests read back and feed to
 * astro::events::eclipse. For each case it records:
 *
 * - the positions Orekit was given, in the inertial frame, in metres;
 * - the lighting ratio with Earth alone, with the Moon alone (a model whose
 *   central body is the Moon), and with Earth plus the Moon added through
 *   addOccultingBody;
 * - the occultation angles of Orekit's OccultationEngine for Earth, for the
 *   Moon, and for the Earth-behind-Moon engine that getLightingRatio builds for
 *   its two-body term;
 * - the acceleration of the Earth-plus-Moon model;
 * - the single-body shadow function of Montenbruck & Gill, "Satellite Orbits"
 *   (2000), section 3.4.2, evaluated on Orekit's angles with 40 decimal digits.
 *
 * Two families of cases are written. "constructed" places the Sun and the Moon
 * at chosen apparent offsets from Earth's limb as seen from the satellite, so
 * that every arrangement of the three discs occurs. "de440" samples circular
 * low orbits through the solar eclipse of 8 April 2024 with Orekit's own Sun
 * and Moon from the JPL DE 440 ephemeris.
 *
 * usage: java MoonEclipseOracle <orekit-data directory> <orekit-data commit>
 */

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.hipparchus.CalculusFieldElement;
import org.hipparchus.dfp.Dfp;
import org.hipparchus.dfp.DfpField;
import org.hipparchus.dfp.DfpMath;
import org.hipparchus.geometry.euclidean.threed.FieldVector3D;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.util.FastMath;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient;
import org.orekit.forces.radiation.SolarRadiationPressure;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.FieldAbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.AbsolutePVCoordinates;
import org.orekit.utils.Constants;
import org.orekit.utils.ExtendedPositionProvider;
import org.orekit.utils.ExtendedPositionProviderAdapter;
import org.orekit.utils.FrameAdapter;
import org.orekit.utils.OccultationEngine;
import org.orekit.utils.PVCoordinates;

public final class MoonEclipseOracle {

    /** Spherical Earth radius of sidereon-core's shadow functions (m). */
    private static final double EARTH_RADIUS = 6371000.0;

    /** Flattening of sidereon-core's oblate shadow model. */
    private static final double WGS84_FLATTENING = 1.0 / 298.257223563;

    /** Mean lunar radius of sidereon-core (m). */
    private static final double MOON_RADIUS = 1737400.0;

    /** Reference distance of the radiation pressure, one astronomical unit (m). */
    private static final double D_REF = 149597870700.0;

    /** Radiation pressure at the reference distance (N/m^2). */
    private static final double P_REF = 4.56e-6;

    /** Cannonball cross-section (m^2), reflectivity coefficient and mass (kg). */
    private static final double AREA = 20.0;
    private static final double CR = 1.3;
    private static final double MASS = 1000.0;

    /** Working precision of the reference evaluation, decimal digits. */
    private static final int DIGITS = 40;

    private static Frame gcrf;

    private MoonEclipseOracle() {
    }

    /** A body at a fixed position of the inertial frame. */
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
        public <T extends CalculusFieldElement<T>> FieldVector3D<T> getPosition(final FieldAbsoluteDate<T> date,
                                                                                final Frame frame) {
            final FieldVector3D<T> p = new FieldVector3D<>(date.getField(), position);
            if (frame == gcrf) {
                return p;
            }
            return gcrf.getStaticTransformTo(frame, date).transformPosition(p);
        }
    }

    public static void main(final String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("usage: MoonEclipseOracle <orekit-data directory> <orekit-data commit>");
        }
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(new File(args[0])));
        gcrf = FramesFactory.getGCRF();

        final List<String> constructed = constructedCases();
        final List<String> de440 = de440Cases();

        final StringBuilder out = new StringBuilder();
        out.append("{\n");
        out.append("  \"_source\": {\n");
        out.append("    \"generator\": \"fixtures-generators/orekit_moon_eclipse_oracle/generate.sh\",\n");
        out.append("    \"oracle\": \"Orekit org.orekit.forces.radiation.SolarRadiationPressure.getLightingRatio and acceleration, with the Moon added through addOccultingBody; org.orekit.utils.OccultationEngine.angles\",\n");
        out.append("    \"reference\": \"Montenbruck & Gill, Satellite Orbits (2000), section 3.4.2, evaluated on Orekit's angles with ").append(DIGITS).append(" decimal digits (org.hipparchus.dfp)\",\n");
        out.append("    \"orekit\": \"").append(mavenVersion("org.orekit", "orekit")).append("\",\n");
        out.append("    \"hipparchus\": \"").append(mavenVersion("org.hipparchus", "hipparchus-core")).append("\",\n");
        out.append("    \"orekit_data_commit\": \"").append(args[1]).append("\",\n");
        out.append("    \"java\": \"").append(System.getProperty("java.runtime.version")).append("\",\n");
        out.append("    \"frame\": \"GCRF, Earth-centred; positions in metres; the ellipsoid's polar axis is the frame's z axis\",\n");
        out.append("    \"de440\": \"Sun and Moon of CelestialBodyFactory (JPL DE 440) at UTC epochs of the solar eclipse of 2024-04-08; satellites on circular orbits of 6878 km radius\"\n");
        out.append("  },\n");
        out.append("  \"sun_radius_m\": ").append(Constants.SUN_RADIUS).append(",\n");
        out.append("  \"earth_radius_m\": ").append(EARTH_RADIUS).append(",\n");
        out.append("  \"moon_radius_m\": ").append(MOON_RADIUS).append(",\n");
        out.append("  \"wgs84_flattening\": ").append(WGS84_FLATTENING).append(",\n");
        out.append("  \"d_ref_m\": ").append(D_REF).append(",\n");
        out.append("  \"p_ref_n_m2\": ").append(P_REF).append(",\n");
        out.append("  \"cr\": ").append(CR).append(",\n");
        out.append("  \"area_to_mass_m2_kg\": ").append(AREA / MASS).append(",\n");
        appendCases(out, "constructed", constructed);
        out.append(",\n");
        appendCases(out, "de440", de440);
        out.append("\n}\n");
        System.out.print(out);
    }

    private static void appendCases(final StringBuilder out, final String key, final List<String> cases) {
        out.append("  \"").append(key).append("\": [\n");
        for (int i = 0; i < cases.size(); ++i) {
            out.append("    ").append(cases.get(i)).append(i + 1 < cases.size() ? ",\n" : "\n");
        }
        out.append("  ]");
    }

    /** Version of a Maven artifact on the class path, from its pom.properties. */
    private static String mavenVersion(final String group, final String artifact) throws IOException {
        final String resource = "/META-INF/maven/" + group + "/" + artifact + "/pom.properties";
        try (InputStream in = MoonEclipseOracle.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("no " + resource + " on the class path");
            }
            final Properties properties = new Properties();
            properties.load(in);
            return properties.getProperty("version");
        }
    }

    // ---------------------------------------------------------------- cases

    /** Satellite regime of the constructed cases. */
    private static final class Regime {

        final String name;
        final Vector3D satellite;
        final double flattening;
        final double[] moonDistances;
        final boolean reduced;

        Regime(final String name, final double radius, final Vector3D direction, final double flattening,
               final double[] moonDistances, final boolean reduced) {
            this.name = name;
            this.satellite = direction.normalize().scalarMultiply(radius);
            this.flattening = flattening;
            this.moonDistances = moonDistances;
            this.reduced = reduced;
        }
    }

    /** Sun, Moon and satellite arrangements built from apparent offsets.
     * <p>
     * The Sun is placed so that its centre is {@code earthOffset} solar radii
     * outside Earth's limb as seen from the satellite (negative values are
     * behind Earth; infinity puts the Sun 2 rad from Earth's centre). The Moon
     * is placed at position angle {@code moonAngle}, measured around the Sun
     * from the direction of Earth's centre, at a separation set by
     * {@code moonDepth}: -1 is concentric with the Sun, 0 internally tangent,
     * 1 externally tangent, values in between scale linearly, and values above
     * 1 multiply the sum of the two apparent radii.
     * </p>
     */
    private static List<String> constructedCases() {
        final double[] nearEarth = {3.58e8, 4.05e8};
        final double[] beyondMoon = {6.45e7, 7.2e7};
        final Regime[] regimes = {
            new Regime("leo", 6.878e6, new Vector3D(0.36, -0.48, 0.80), 0.0, nearEarth, false),
            new Regime("geo", 4.2164e7, new Vector3D(0.70, 0.71, -0.07), 0.0, nearEarth, false),
            new Regime("beyond_moon", 4.49e8, new Vector3D(-0.28, -0.53, 0.80), 0.0, beyondMoon, false),
            new Regime("leo_oblate", 6.878e6, new Vector3D(0.36, -0.48, 0.80), WGS84_FLATTENING, nearEarth, true),
            new Regime("geo_oblate", 4.2164e7, new Vector3D(0.31, 0.22, 0.925), WGS84_FLATTENING, nearEarth, true),
        };
        final double[] sunDistances = {1.47098e11, 1.52098e11};
        // Sun centre relative to Earth's limb, in solar radii
        final double[] partialOffsets = {-0.62, 0.0, 0.55};
        final double[] otherOffsets = {-4.0, -1.0, 1.0, 3.5, Double.POSITIVE_INFINITY};
        // {depth, position angle in degrees}
        final double[][] allMoons = {
            {-1.0, 0.0}, {-0.4, 215.0}, {0.0, 40.0}, {0.3, 0.0}, {0.3, 180.0}, {0.5, 90.0}, {0.6, 300.0},
            {0.85, 160.0}, {1.0, 310.0}, {1.3, 0.0},
        };
        final double[][] fewMoons = {{-1.0, 0.0}, {0.5, 130.0}, {30.0, 75.0}};
        final double[][] reducedMoons = {{-1.0, 0.0}, {0.3, 0.0}, {0.5, 90.0}, {30.0, 75.0}};
        final double[][] farMoon = {{30.0, 75.0}};

        final List<String> cases = new ArrayList<>();
        final AbsoluteDate date = AbsoluteDate.J2000_EPOCH;
        int counter = 0;
        for (final Regime regime : regimes) {
            for (int pass = 0; pass < 2; ++pass) {
                final double[] offsets = pass == 0 ? partialOffsets : otherOffsets;
                final double[][] moons = pass == 0 ? (regime.reduced ? reducedMoons : allMoons) :
                                                     (regime.reduced ? farMoon : fewMoons);
                for (final double earthOffset : offsets) {
                    for (final double[] moon : moons) {
                        final double sunDistance = sunDistances[counter % 2];
                        final double moonDistance = regime.moonDistances[(counter / 2) % 2];
                        ++counter;
                        cases.add(constructed(regime, date, sunDistance, earthOffset, moonDistance, moon[0], moon[1]));
                    }
                }
            }
        }
        return cases;
    }

    private static String constructed(final Regime regime, final AbsoluteDate date, final double sunDistance,
                                      final double earthOffset, final double moonDistance,
                                      final double moonDepth, final double moonAngle) {
        final Vector3D sat = regime.satellite;
        final Vector3D toEarth = sat.negate().normalize();
        final Vector3D side = perpendicular(toEarth, new Vector3D(0.53, 0.29, -0.61));
        final OneAxisEllipsoid earth = new OneAxisEllipsoid(EARTH_RADIUS, regime.flattening, gcrf);

        final double sunRadius = FastMath.asin(Constants.SUN_RADIUS / sunDistance);
        final Vector3D limb = earth.pointOnLimb(sat, sat.add(toEarth.add(side).scalarMultiply(1.0e9)));
        final double limbRadius = Vector3D.angle(limb.subtract(sat), toEarth);
        final double earthSeparation = Double.isInfinite(earthOffset) ? 2.0 : limbRadius + earthOffset * sunRadius;
        final Vector3D toSun = new Vector3D(FastMath.cos(earthSeparation), toEarth, FastMath.sin(earthSeparation), side);
        final Vector3D sun = sat.add(toSun.scalarMultiply(sunDistance));

        final double moonRadius = FastMath.asin(MOON_RADIUS / moonDistance);
        final double inner = FastMath.abs(moonRadius - sunRadius);
        final double separation;
        if (moonDepth <= 0.0) {
            separation = (1.0 + moonDepth) * inner;
        } else if (moonDepth <= 1.0) {
            separation = inner + moonDepth * 2.0 * FastMath.min(sunRadius, moonRadius);
        } else {
            separation = moonDepth * (sunRadius + moonRadius);
        }
        final Vector3D e1 = perpendicular(toSun, toEarth);
        final Vector3D e2 = Vector3D.crossProduct(toSun, e1);
        final double angle = FastMath.toRadians(moonAngle);
        final Vector3D across = new Vector3D(FastMath.cos(angle), e1, FastMath.sin(angle), e2);
        final Vector3D toMoon = new Vector3D(FastMath.cos(separation), toSun, FastMath.sin(separation), across);
        final Vector3D moon = sat.add(toMoon.scalarMultiply(moonDistance));

        final Vector3D velocity = Vector3D.crossProduct(new Vector3D(0.1, 0.2, 0.97), sat).normalize().
                                  scalarMultiply(FastMath.sqrt(Constants.EIGEN5C_EARTH_MU / sat.getNorm()));
        return evaluate(regime.name, date, sat, velocity, new FixedBody(sun), new FixedBody(moon), regime.flattening);
    }

    /** Unit vector of the part of {@code v} perpendicular to the unit vector {@code axis}. */
    private static Vector3D perpendicular(final Vector3D axis, final Vector3D v) {
        return v.subtract(axis.scalarMultiply(Vector3D.dotProduct(v, axis))).normalize();
    }

    /** Low circular orbits through the solar eclipse of 8 April 2024. */
    private static List<String> de440Cases() {
        final ExtendedPositionProvider sun = CelestialBodyFactory.getSun();
        final ExtendedPositionProvider moon = CelestialBodyFactory.getMoon();
        final AbsoluteDate start = new AbsoluteDate(2024, 4, 8, 16, 0, 0.0, TimeScalesFactory.getUTC());
        final double duration = 4.5 * 3600.0;
        final double step = 20.0;
        final int perKind = 4;

        final List<String> cases = new ArrayList<>();
        for (int k = 0; k < 6; ++k) {
            final KeplerianOrbit orbit = new KeplerianOrbit(6.878e6, 0.0, FastMath.toRadians(51.6 + 9.0 * k), 0.0,
                                                           FastMath.toRadians(60.0 * k), FastMath.toRadians(25.0 * k),
                                                           PositionAngleType.MEAN, gcrf, start,
                                                           Constants.EIGEN5C_EARTH_MU);
            final KeplerianPropagator propagator = new KeplerianPropagator(orbit);
            final List<String> moonOnly = new ArrayList<>();
            final List<String> earthOnly = new ArrayList<>();
            final List<String> both = new ArrayList<>();
            final List<String> neither = new ArrayList<>();
            for (double t = 0.0; t <= duration; t += step) {
                final SpacecraftState s = propagator.propagate(start.shiftedBy(t));
                final Vector3D sat = s.getPosition(gcrf);
                final Vector3D velocity = s.getPVCoordinates(gcrf).getVelocity();
                final String json = evaluate("de440_orbit_" + k, s.getDate(), sat, velocity, sun, moon, 0.0);
                final boolean moonPartial = json.contains("moon_partial") || json.contains("moon_annular") ||
                                            json.contains("moon_total");
                final boolean earthPartial = json.contains("earth_partial");
                if (moonPartial && earthPartial) {
                    both.add(json);
                } else if (moonPartial && json.contains("earth_clear")) {
                    moonOnly.add(json);
                } else if (earthPartial) {
                    earthOnly.add(json);
                } else if (json.contains("earth_clear")) {
                    neither.add(json);
                }
            }
            cases.addAll(thin(moonOnly, perKind));
            cases.addAll(thin(both, perKind));
            cases.addAll(thin(earthOnly, 2));
            cases.addAll(thin(neither, 1));
        }
        return cases;
    }

    /** At most {@code count} evenly spaced elements. */
    private static List<String> thin(final List<String> all, final int count) {
        if (all.size() <= count) {
            return all;
        }
        final List<String> kept = new ArrayList<>();
        for (int i = 0; i < count; ++i) {
            kept.add(all.get((int) ((long) i * (all.size() - 1) / (count - 1 == 0 ? 1 : count - 1))));
        }
        return kept;
    }

    // ------------------------------------------------------------ evaluation

    private static String evaluate(final String name, final AbsoluteDate date, final Vector3D sat,
                                   final Vector3D velocity, final ExtendedPositionProvider sun,
                                   final ExtendedPositionProvider moon, final double flattening) {
        final OneAxisEllipsoid earth = new OneAxisEllipsoid(EARTH_RADIUS, flattening, gcrf);
        final IsotropicRadiationSingleCoefficient spacecraft = new IsotropicRadiationSingleCoefficient(AREA, CR);

        final SolarRadiationPressure earthOnly = new SolarRadiationPressure(D_REF, P_REF, sun, earth, spacecraft);
        final SolarRadiationPressure earthAndMoon = new SolarRadiationPressure(D_REF, P_REF, sun, earth, spacecraft);
        earthAndMoon.addOccultingBody(moon, MOON_RADIUS);
        final Frame moonFrame = new ExtendedPositionProviderAdapter(gcrf, moon, "moon-centred");
        final SolarRadiationPressure moonOnly =
                        new SolarRadiationPressure(D_REF, P_REF, sun, new OneAxisEllipsoid(MOON_RADIUS, 0.0, moonFrame),
                                                   spacecraft);

        final SpacecraftState state =
                        new SpacecraftState(new AbsolutePVCoordinates(gcrf, date, new PVCoordinates(sat, velocity)), MASS);

        final double ratioEarth = earthOnly.getLightingRatio(state);
        final double ratioMoon = moonOnly.getLightingRatio(state);
        final double ratioBoth = earthAndMoon.getLightingRatio(state);
        final Vector3D acceleration = earthAndMoon.acceleration(state, earthAndMoon.getParameters(date));

        final List<OccultationEngine> engines = earthAndMoon.getOccultingBodies();
        final OccultationEngine.OccultationAngles earthAngles = engines.get(0).angles(state);
        final OccultationEngine.OccultationAngles moonAngles = engines.get(1).angles(state);
        // the engine getLightingRatio builds for its two-body term: Earth's disc behind the Moon
        final OccultationEngine.OccultationAngles mutualAngles =
                        new OccultationEngine(new FrameAdapter(earth.getBodyFrame()), earth.getEquatorialRadius(),
                                              engines.get(1).getOcculting()).angles(state);

        final String earthKind = ratioEarth == 1.0 ? "earth_clear" : ratioEarth == 0.0 ? "earth_total" : "earth_partial";
        final String moonKind;
        if (ratioMoon == 1.0) {
            moonKind = "moon_clear";
        } else if (ratioMoon == 0.0) {
            moonKind = "moon_total";
        } else if (moonAngles.getSeparation() + moonAngles.getLimbRadius() <= moonAngles.getOccultedApparentRadius()) {
            moonKind = "moon_annular";
        } else {
            moonKind = "moon_partial";
        }

        final StringBuilder json = new StringBuilder();
        json.append("{\"name\": \"").append(name).append("\"");
        json.append(", \"kind\": \"").append(earthKind).append('+').append(moonKind).append("\"");
        json.append(", \"earth_flattening\": ").append(flattening);
        json.append(", \"sat_m\": ").append(vector(sat));
        json.append(", \"sun_m\": ").append(vector(sun.getPosition(date, gcrf)));
        json.append(", \"moon_m\": ").append(vector(moon.getPosition(date, gcrf)));
        json.append(", \"lighting_earth\": ").append(ratioEarth);
        json.append(", \"lighting_moon\": ").append(ratioMoon);
        json.append(", \"lighting_earth_and_moon\": ").append(ratioBoth);
        json.append(", \"acceleration_m_s2\": ").append(vector(acceleration));
        json.append(", \"earth_angles\": ").append(angles(earthAngles));
        json.append(", \"moon_angles\": ").append(angles(moonAngles));
        json.append(", \"mutual_angles\": ").append(angles(mutualAngles));
        json.append(", \"reference_lighting_earth\": \"").append(reference(earthAngles)).append("\"");
        json.append(", \"reference_lighting_moon\": \"").append(reference(moonAngles)).append("\"");
        json.append("}");
        return json.toString();
    }

    private static String vector(final Vector3D v) {
        return "[" + v.getX() + ", " + v.getY() + ", " + v.getZ() + "]";
    }

    /** {separation, limb radius of the occulting body, apparent radius of the occulted body}, radians. */
    private static String angles(final OccultationEngine.OccultationAngles a) {
        return "[" + a.getSeparation() + ", " + a.getLimbRadius() + ", " + a.getOccultedApparentRadius() + "]";
    }

    /** Lighting ratio 1 - A / (pi a^2) of Montenbruck & Gill section 3.4.2, in extended precision.
     * <p>
     * a is the apparent radius of the occulted body, b that of the occulting
     * body and c their separation, taken as the exact values of Orekit's
     * doubles. The occulted area is zero for c >= a + b, the whole disc for
     * c <= b - a, the occulting disc for c <= a - b, and otherwise
     * A = a^2 acos(x / a) + b^2 acos((c - x) / b) - c y with
     * x = (c^2 + a^2 - b^2) / (2 c) and y = sqrt(a^2 - x^2).
     * </p>
     */
    private static String reference(final OccultationEngine.OccultationAngles angles) {
        final DfpField field = new DfpField(DIGITS);
        final Dfp a = exact(field, angles.getOccultedApparentRadius());
        final Dfp b = exact(field, angles.getLimbRadius());
        final Dfp c = exact(field, angles.getSeparation());
        final Dfp one = field.getOne();
        final Dfp lighting;
        if (!c.lessThan(a.add(b))) {
            lighting = one;
        } else if (!c.greaterThan(b.subtract(a))) {
            lighting = field.getZero();
        } else if (!c.greaterThan(a.subtract(b))) {
            lighting = one.subtract(b.multiply(b).divide(a.multiply(a)));
        } else {
            final Dfp a2 = a.multiply(a);
            final Dfp b2 = b.multiply(b);
            final Dfp x = c.multiply(c).add(a2).subtract(b2).divide(c.multiply(2));
            final Dfp y = a2.subtract(x.multiply(x)).sqrt();
            final Dfp area = a2.multiply(DfpMath.acos(x.divide(a))).
                             add(b2.multiply(DfpMath.acos(c.subtract(x).divide(b)))).
                             subtract(c.multiply(y));
            lighting = one.subtract(area.divide(field.getPi().multiply(a2)));
        }
        return lighting.toString();
    }

    private static Dfp exact(final DfpField field, final double value) {
        return field.newDfp(new BigDecimal(value).toPlainString());
    }

}
