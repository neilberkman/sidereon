/* Oracle for the mean-of-date to GCRS rotation and the GCRS Sun and Moon of
 * sidereon-core (astro::frames::transforms::mean_of_date_to_gcrs_matrix and
 * astro::bodies::sun_moon::sun_moon_gcrs).
 *
 * Prints, as JSON, for TT epochs from 1990 to 2050:
 *
 * - fixed vectors transformed from Orekit's mean-of-date frame to GCRF:
 *   FramesFactory.getMOD(IERSConventions.IERS_2010), whose parent is EME2000
 *   (GCRF through the IAU 2006 frame bias) and whose transform from it is the
 *   IAU 2006 precession R3(chi_A) R1(-omega_A) R3(-psi_A) R1(eps_0)
 *   (IERS Conventions 2010, equations 5.39 and 5.40; Capitaine et al. 2003);
 *   the same vectors through the two factors separately, MOD to EME2000 (the
 *   precession) at each epoch and EME2000 to GCRF (the frame bias) once;
 * - the geocentric Sun and Moon of CelestialBodyFactory (JPL DE 440 from
 *   orekit-data), geometric (no light time or aberration), in GCRF and in that
 *   mean-of-date frame.
 *
 * Orekit evaluates the precession angles at TT (IERSConventions.evaluateTC,
 * Julian centuries of TT since J2000.0); each epoch therefore also records the
 * TT and the TDB in seconds since J2000.0.
 *
 * usage: java SunMoonGcrsOracle <orekit-data directory> <orekit-data commit>
 */

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.bodies.CelestialBody;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.Transform;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.IERSConventions;

public final class SunMoonGcrsOracle {

    /** Vectors transformed from the mean-of-date frame to GCRF: the three axes and one off-axis unit vector. */
    private static final Vector3D[] VECTORS = {
        Vector3D.PLUS_I, Vector3D.PLUS_J, Vector3D.PLUS_K, new Vector3D(0.36, -0.48, 0.8)
    };

    private SunMoonGcrsOracle() {
    }

    public static void main(final String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("usage: SunMoonGcrsOracle <orekit-data directory> <orekit-data commit>");
        }
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(new File(args[0])));
        final Frame gcrf = FramesFactory.getGCRF();
        final Frame mod = FramesFactory.getMOD(IERSConventions.IERS_2010);
        final Frame eme2000 = FramesFactory.getEME2000();
        if (mod.getParent() != eme2000 || eme2000.getParent() != gcrf) {
            throw new IllegalStateException("MOD (IERS 2010) is expected to be a child of EME2000, a child of GCRF");
        }
        final TimeScale tt = TimeScalesFactory.getTT();
        final TimeScale tdb = TimeScalesFactory.getTDB();
        final CelestialBody sun = CelestialBodyFactory.getSun();
        final CelestialBody moon = CelestialBodyFactory.getMoon();
        final AbsoluteDate j2000Tdb = new AbsoluteDate(2000, 1, 1, 12, 0, 0.0, tdb);

        final List<String> epochs = new ArrayList<>();
        // One epoch a year, 1990 to 2050, spread over the months and hours.
        for (int k = 0; k <= 60; ++k) {
            final int year = 1990 + k;
            final int month = 1 + (7 * k) % 12;
            final int day = 1 + (11 * k) % 28;
            final int hour = (5 * k) % 24;
            final int minute = (13 * k) % 60;
            epochs.add(epoch("yearly", new AbsoluteDate(year, month, day, hour, minute, 0.0, tt),
                             new int[] {year, month, day, hour, minute}, tt, tdb, j2000Tdb, gcrf, eme2000, mod,
                             sun, moon));
        }
        // One lunar month of 2026, every 2 days, for the Moon's error over its orbit.
        for (int k = 0; k < 15; ++k) {
            final AbsoluteDate date = new AbsoluteDate(2026, 11, 1, 0, 0, 0.0, tt).shiftedBy(2.0 * 86400.0 * k);
            final int day = 1 + 2 * k;
            final int[] cal = day <= 30 ? new int[] {2026, 11, day, 0, 0} : new int[] {2026, 12, day - 30, 0, 0};
            epochs.add(epoch("month_2026", date, cal, tt, tdb, j2000Tdb, gcrf, eme2000, mod, sun, moon));
        }

        final StringBuilder out = new StringBuilder();
        out.append("{\n");
        out.append("  \"_source\": {\n");
        out.append("    \"generator\": \"fixtures-generators/orekit_sun_moon_gcrs_oracle/generate.sh\",\n");
        out.append("    \"oracle\": \"Orekit FramesFactory.getMOD(IERSConventions.IERS_2010).getTransformTo(FramesFactory.getGCRF(), date).transformVector; CelestialBodyFactory.getSun() and getMoon().getPosition(date, frame)\",\n");
        out.append("    \"orekit\": \"").append(mavenVersion("org.orekit", "orekit")).append("\",\n");
        out.append("    \"hipparchus\": \"").append(mavenVersion("org.hipparchus", "hipparchus-core")).append("\",\n");
        out.append("    \"orekit_data_commit\": \"").append(args[1]).append("\",\n");
        out.append("    \"java\": \"").append(System.getProperty("java.runtime.version")).append("\",\n");
        out.append("    \"ephemeris\": \"JPL DE 440 (orekit-data DE-440-ephemerides)\",\n");
        out.append("    \"mod_frame\": \"MOD, IERS 2010 conventions: parent EME2000 (GCRF with the IAU 2006 frame bias), IAU 2006 precession R3(chi_A) R1(-omega_A) R3(-psi_A) R1(eps_0) evaluated at TT\",\n");
        out.append("    \"units\": \"positions in metres, geocentric and geometric; times in seconds since J2000.0 of their scale\"\n");
        out.append("  },\n");
        out.append("  \"vectors\": [");
        for (int i = 0; i < VECTORS.length; ++i) {
            out.append(vector(VECTORS[i])).append(i + 1 < VECTORS.length ? ", " : "");
        }
        out.append("],\n");
        // The frame bias does not depend on the date.
        final Transform eme2000ToGcrf = eme2000.getTransformTo(gcrf, AbsoluteDate.J2000_EPOCH);
        out.append("  \"eme2000_to_gcrf\": [");
        for (int i = 0; i < VECTORS.length; ++i) {
            out.append(vector(eme2000ToGcrf.transformVector(VECTORS[i]))).append(i + 1 < VECTORS.length ? ", " : "");
        }
        out.append("],\n");
        out.append("  \"epochs\": [\n");
        for (int i = 0; i < epochs.size(); ++i) {
            out.append("    ").append(epochs.get(i)).append(i + 1 < epochs.size() ? ",\n" : "\n");
        }
        out.append("  ]\n");
        out.append("}\n");
        System.out.print(out);
    }

    private static String epoch(final String set, final AbsoluteDate date, final int[] calendar, final TimeScale tt,
                                final TimeScale tdb, final AbsoluteDate j2000Tdb, final Frame gcrf,
                                final Frame eme2000, final Frame mod,
                                final CelestialBody sun, final CelestialBody moon) {
        final Transform modToGcrf = mod.getTransformTo(gcrf, date);
        final Transform modToEme2000 = mod.getTransformTo(eme2000, date);
        final StringBuilder json = new StringBuilder();
        json.append("{\"set\": \"").append(set).append("\"");
        json.append(", \"tt\": [").append(calendar[0]).append(", ").append(calendar[1]).append(", ")
            .append(calendar[2]).append(", ").append(calendar[3]).append(", ").append(calendar[4]).append(", 0.0]");
        json.append(", \"tt_seconds\": ").append(date.durationFrom(AbsoluteDate.J2000_EPOCH));
        json.append(", \"tdb_seconds\": ").append(date.offsetFrom(j2000Tdb, tdb));
        json.append(", \"mod_to_gcrf\": [");
        for (int i = 0; i < VECTORS.length; ++i) {
            json.append(vector(modToGcrf.transformVector(VECTORS[i]))).append(i + 1 < VECTORS.length ? ", " : "");
        }
        json.append("]");
        json.append(", \"mod_to_eme2000\": [");
        for (int i = 0; i < VECTORS.length; ++i) {
            json.append(vector(modToEme2000.transformVector(VECTORS[i]))).append(i + 1 < VECTORS.length ? ", " : "");
        }
        json.append("]");
        json.append(", \"sun_gcrf_m\": ").append(vector(sun.getPosition(date, gcrf)));
        json.append(", \"moon_gcrf_m\": ").append(vector(moon.getPosition(date, gcrf)));
        json.append(", \"sun_mod_m\": ").append(vector(sun.getPosition(date, mod)));
        json.append(", \"moon_mod_m\": ").append(vector(moon.getPosition(date, mod)));
        json.append("}");
        // The calendar label must name the same TT instant.
        final AbsoluteDate fromLabel = new AbsoluteDate(calendar[0], calendar[1], calendar[2], calendar[3],
                                                        calendar[4], 0.0, tt);
        if (fromLabel.durationFrom(date) != 0.0) {
            throw new IllegalStateException("calendar label does not match the epoch " + date);
        }
        return json.toString();
    }

    private static String vector(final Vector3D v) {
        return "[" + v.getX() + ", " + v.getY() + ", " + v.getZ() + "]";
    }

    /** Version of a Maven artifact on the class path, from its pom.properties. */
    private static String mavenVersion(final String group, final String artifact) throws IOException {
        final String resource = "/META-INF/maven/" + group + "/" + artifact + "/pom.properties";
        try (InputStream in = SunMoonGcrsOracle.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("no " + resource + " on the class path");
            }
            final Properties properties = new Properties();
            properties.load(in);
            return properties.getProperty("version");
        }
    }
}
