import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.jar.JarFile;

import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.util.FastMath;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.LOFType;
import org.orekit.frames.Transform;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.CartesianDerivativesFilter;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;
import org.orekit.utils.TimeStampedPVCoordinates;
import org.orekit.utils.TimeStampedPVCoordinatesHermiteInterpolator;

/**
 * Orekit local-orbital-frame projections for the orbit scoring tests.
 *
 * <p>Writes one JSON document to standard output with two case sets:
 *
 * <ul>
 *   <li>{@code synthetic}: inertial states built by {@link KeplerianOrbit} (circular, eccentric,
 *       retrograde, polar, geostationary) and fixed difference vectors, with the difference
 *       resolved on {@link LOFType#QSW} and {@link LOFType#TNW} of the state.
 *   <li>{@code sp3}: for each predicted ultra-rapid product, the difference between its record
 *       and the final product's record of the same satellite at the same epoch. The final
 *       product's Earth-fixed velocity at the record is the derivative of an 11-record Hermite
 *       interpolation of its positions ({@link TimeStampedPVCoordinatesHermiteInterpolator},
 *       positions only). The Earth-fixed state and difference are taken to GCRF with the full
 *       IERS 2010 ITRF transformation (EOP from the data context), and the difference is resolved
 *       on QSW and TNW of the GCRF state. {@code qsw_itrf_m} resolves the Earth-fixed difference
 *       on QSW of the Earth-fixed state taken as it stands.
 * </ul>
 *
 * <p>SP3 records are read with a fixed-column reader of the SP3-c position records ({@code P}
 * lines), so the reference does not depend on either library's SP3 parser.
 *
 * <p>usage: {@code java LofOracle <orekit-data> <orekit-data-commit> <final.sp3> <ult.sp3>...}
 */
public final class LofOracle {

    /** Number of records in the Hermite interpolation of the final product. */
    private static final int HERMITE_POINTS = 11;

    /** Record indices of the final product at which states are written. */
    private static final int[] RECORD_INDICES = {0, 6, 12};

    private LofOracle() {
    }

    /** One SP3 epoch: its date and the position of every satellite with a valid record. */
    private static final class Sp3Epoch {
        final AbsoluteDate date;
        final String label;
        final Map<String, Vector3D> positions = new TreeMap<>();

        Sp3Epoch(AbsoluteDate date, String label) {
            this.date = date;
            this.label = label;
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 4) {
            throw new IllegalArgumentException(
                "usage: LofOracle <orekit-data> <orekit-data-commit> <final.sp3> <ult.sp3>...");
        }
        Locale.setDefault(Locale.ROOT);
        DataContext.getDefault().getDataProvidersManager()
            .addProvider(new DirectoryCrawler(new File(args[0])));
        String dataCommit = args[1];
        TimeScale gps = TimeScalesFactory.getGPS();
        Frame gcrf = FramesFactory.getGCRF();
        Frame itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, false);

        StringBuilder out = new StringBuilder();
        out.append("{\n");
        out.append("  \"_source\": {\n");
        out.append("    \"generator\": \"fixtures-generators/orekit_lof_oracle/generate.sh\",\n");
        out.append("    \"orekit\": \"").append(bundleVersion(LOFType.class)).append("\",\n");
        out.append("    \"hipparchus\": \"").append(bundleVersion(FastMath.class)).append("\",\n");
        out.append("    \"orekit_data_commit\": \"").append(dataCommit).append("\",\n");
        out.append("    \"java\": \"").append(System.getProperty("java.version")).append("\",\n");
        out.append("    \"oracle\": \"org.orekit.frames.LOFType.QSW and LOFType.TNW, rotationFromInertial(pv).applyTo(d)\",\n");
        out.append("    \"sp3_states\": \"Final-product record r and record difference d = predicted - final, meters; Earth-fixed velocity from an ")
            .append(HERMITE_POINTS)
            .append("-record Hermite interpolation of the final positions (positions only); ITRF to GCRF with IERSConventions.IERS_2010, full EOP\",\n");
        out.append("    \"mu_m3_s2\": ").append(num(Constants.WGS84_EARTH_MU)).append("\n");
        out.append("  },\n");

        writeSynthetic(out, gcrf, gps);
        out.append(",\n");

        Sp3Product reference = Sp3Product.read(new File(args[2]), gps);
        out.append("  \"sp3\": {\n");
        out.append("    \"reference\": ").append(str(new File(args[2]).getName())).append(",\n");
        out.append("    \"sources\": [\n");
        for (int s = 3; s < args.length; s++) {
            Sp3Product source = Sp3Product.read(new File(args[s]), gps);
            writeSp3Source(out, reference, source, new File(args[s]).getName(), itrf, gcrf);
            out.append(s + 1 < args.length ? ",\n" : "\n");
        }
        out.append("    ]\n");
        out.append("  }\n");
        out.append("}\n");
        System.out.print(out);
    }

    /** States and differences built from Keplerian elements in GCRF. */
    private static void writeSynthetic(StringBuilder out, Frame gcrf, TimeScale gps) {
        AbsoluteDate date = new AbsoluteDate(2026, 5, 13, 0, 0, 0.0, gps);
        double mu = Constants.WGS84_EARTH_MU;
        // name, a (m), e, i (deg), argument of perigee (deg), RAAN (deg), true anomaly (deg)
        Object[][] orbits = {
            {"gps_circular", 26_559_700.0, 0.0, 55.0, 0.0, 30.0, 40.0},
            {"galileo_near_circular", 29_600_000.0, 0.0003, 56.0, 10.0, 120.0, 250.0},
            {"eccentric_ascending", 26_559_700.0, 0.3, 63.4, 270.0, 15.0, 60.0},
            {"eccentric_descending", 26_559_700.0, 0.3, 63.4, 270.0, 15.0, 230.0},
            {"highly_eccentric", 24_000_000.0, 0.7, 28.5, 180.0, 75.0, 100.0},
            {"retrograde", 7_000_000.0, 0.01, 98.0, 90.0, 200.0, 135.0},
            {"polar", 26_559_700.0, 0.02, 90.0, 45.0, 300.0, 315.0},
            {"geostationary", 42_164_000.0, 0.0002, 0.05, 0.0, 0.0, 77.0},
            {"inclined_geosynchronous", 42_164_000.0, 0.075, 43.0, 270.0, 120.0, 10.0},
        };
        double[][] differences = {
            {0.031, -0.084, 0.052},
            {1.25, -2.5, 0.75},
            {-1500.0, 2200.0, -900.0},
        };
        out.append("  \"synthetic\": [\n");
        int count = 0;
        int total = orbits.length * differences.length;
        for (Object[] orbit : orbits) {
            KeplerianOrbit kepler = new KeplerianOrbit(
                (double) orbit[1], (double) orbit[2],
                FastMath.toRadians((double) orbit[3]), FastMath.toRadians((double) orbit[4]),
                FastMath.toRadians((double) orbit[5]), FastMath.toRadians((double) orbit[6]),
                PositionAngleType.TRUE, gcrf, date, mu);
            PVCoordinates pv = kepler.getPVCoordinates();
            for (int k = 0; k < differences.length; k++) {
                Vector3D d = new Vector3D(differences[k][0], differences[k][1], differences[k][2]);
                Vector3D qsw = LOFType.QSW.rotationFromInertial(pv).applyTo(d);
                Vector3D tnw = LOFType.TNW.rotationFromInertial(pv).applyTo(d);
                out.append("    {\"name\": ").append(str(orbit[0] + "_d" + k));
                out.append(", \"r_m\": ").append(vec(pv.getPosition()));
                out.append(", \"v_m_s\": ").append(vec(pv.getVelocity()));
                out.append(", \"d_m\": ").append(vec(d));
                out.append(", \"qsw_m\": ").append(vec(qsw));
                out.append(", \"tnw_m\": ").append(vec(tnw));
                out.append("}");
                count++;
                out.append(count < total ? ",\n" : "\n");
            }
        }
        out.append("  ]");
    }

    private static void writeSp3Source(StringBuilder out, Sp3Product reference, Sp3Product source,
                                       String name, Frame itrf, Frame gcrf) {
        out.append("      {\"file\": ").append(str(name)).append(", \"states\": [\n");
        List<String> lines = new ArrayList<>();
        for (int index : RECORD_INDICES) {
            Sp3Epoch epoch = reference.epochs.get(index);
            Sp3Epoch predicted = source.at(epoch.date);
            if (predicted == null) {
                throw new IllegalStateException(name + " has no record epoch " + epoch.label);
            }
            Transform toGcrf = itrf.getTransformTo(gcrf, epoch.date);
            for (Map.Entry<String, Vector3D> entry : epoch.positions.entrySet()) {
                String sat = entry.getKey();
                Vector3D predictedPosition = predicted.positions.get(sat);
                if (predictedPosition == null) {
                    continue;
                }
                Vector3D r = entry.getValue();
                Vector3D d = predictedPosition.subtract(r);
                Vector3D v = reference.velocity(sat, index);
                PVCoordinates earthFixed = new PVCoordinates(r, v);
                PVCoordinates inertial = toGcrf.transformPVCoordinates(earthFixed);
                Vector3D dInertial = toGcrf.transformVector(d);
                Rotation qswInertial = LOFType.QSW.rotationFromInertial(inertial);
                Rotation tnwInertial = LOFType.TNW.rotationFromInertial(inertial);
                Rotation qswEarthFixed = LOFType.QSW.rotationFromInertial(earthFixed);
                StringBuilder line = new StringBuilder();
                line.append("        {\"satellite\": ").append(str(sat));
                line.append(", \"epoch\": ").append(str(epoch.label));
                line.append(", \"reference_record\": ").append(index);
                line.append(", \"r_itrf_m\": ").append(vec(r));
                line.append(", \"v_itrf_m_s\": ").append(vec(v));
                line.append(", \"d_itrf_m\": ").append(vec(d));
                line.append(", \"r_gcrf_m\": ").append(vec(inertial.getPosition()));
                line.append(", \"v_gcrf_m_s\": ").append(vec(inertial.getVelocity()));
                line.append(", \"d_gcrf_m\": ").append(vec(dInertial));
                line.append(", \"qsw_gcrf_m\": ").append(vec(qswInertial.applyTo(dInertial)));
                line.append(", \"tnw_gcrf_m\": ").append(vec(tnwInertial.applyTo(dInertial)));
                line.append(", \"qsw_itrf_m\": ").append(vec(qswEarthFixed.applyTo(d)));
                line.append("}");
                lines.add(line.toString());
            }
        }
        out.append(String.join(",\n", lines)).append("\n      ]}");
    }

    /** The position records of an SP3-c file, in GPS time. */
    private static final class Sp3Product {
        final List<Sp3Epoch> epochs = new ArrayList<>();

        static Sp3Product read(File file, TimeScale gps) throws IOException {
            Sp3Product product = new Sp3Product();
            Sp3Epoch current = null;
            boolean timeSystemSeen = false;
            for (String line : Files.readAllLines(file.toPath(), StandardCharsets.US_ASCII)) {
                if (line.startsWith("%c") && !timeSystemSeen) {
                    timeSystemSeen = true;
                    String system = line.substring(9, 12);
                    if (!system.equals("GPS")) {
                        throw new IllegalStateException(file + ": time system " + system);
                    }
                } else if (line.startsWith("* ")) {
                    int year = Integer.parseInt(line.substring(3, 7).trim());
                    int month = Integer.parseInt(line.substring(8, 10).trim());
                    int day = Integer.parseInt(line.substring(11, 13).trim());
                    int hour = Integer.parseInt(line.substring(14, 16).trim());
                    int minute = Integer.parseInt(line.substring(17, 19).trim());
                    double second = Double.parseDouble(line.substring(20, 31).trim());
                    AbsoluteDate date = new AbsoluteDate(year, month, day, hour, minute, second, gps);
                    String label = String.format("%04d-%02d-%02dT%02d:%02d:%02d",
                        year, month, day, hour, minute, (int) second);
                    current = new Sp3Epoch(date, label);
                    product.epochs.add(current);
                } else if (line.startsWith("P") && current != null) {
                    String sat = line.substring(1, 4);
                    double x = Double.parseDouble(line.substring(4, 18).trim());
                    double y = Double.parseDouble(line.substring(18, 32).trim());
                    double z = Double.parseDouble(line.substring(32, 46).trim());
                    if (x == 0.0 && y == 0.0 && z == 0.0) {
                        continue;
                    }
                    current.positions.put(sat, new Vector3D(x * 1000.0, y * 1000.0, z * 1000.0));
                }
            }
            if (!timeSystemSeen) {
                throw new IllegalStateException(file + ": no time system line");
            }
            return product;
        }

        Sp3Epoch at(AbsoluteDate date) {
            for (Sp3Epoch epoch : epochs) {
                if (epoch.date.durationFrom(date) == 0.0) {
                    return epoch;
                }
            }
            return null;
        }

        /**
         * Earth-fixed velocity of {@code sat} at record {@code index}: the derivative of the
         * Hermite interpolation, positions only, through the {@link #HERMITE_POINTS} records
         * centred on the record where the product allows, shifted inward at its ends.
         */
        Vector3D velocity(String sat, int index) {
            int first = Math.max(0, Math.min(index - HERMITE_POINTS / 2, epochs.size() - HERMITE_POINTS));
            List<TimeStampedPVCoordinates> sample = new ArrayList<>();
            for (int k = first; k < first + HERMITE_POINTS; k++) {
                Vector3D position = epochs.get(k).positions.get(sat);
                if (position == null) {
                    throw new IllegalStateException("no record of " + sat + " at " + epochs.get(k).label);
                }
                sample.add(new TimeStampedPVCoordinates(epochs.get(k).date, position, Vector3D.ZERO));
            }
            TimeStampedPVCoordinatesHermiteInterpolator interpolator =
                new TimeStampedPVCoordinatesHermiteInterpolator(HERMITE_POINTS, CartesianDerivativesFilter.USE_P);
            return interpolator.interpolate(epochs.get(index).date, sample).getVelocity();
        }
    }

    private static String bundleVersion(Class<?> type) throws IOException {
        String path = type.getProtectionDomain().getCodeSource().getLocation().getPath();
        try (JarFile jar = new JarFile(path)) {
            return jar.getManifest().getMainAttributes().getValue("Bundle-Version");
        }
    }

    private static String num(double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalStateException("non-finite value");
        }
        return Double.toString(value);
    }

    private static String vec(Vector3D v) {
        return "[" + num(v.getX()) + ", " + num(v.getY()) + ", " + num(v.getZ()) + "]";
    }

    private static String str(String s) {
        return "\"" + s + "\"";
    }
}
