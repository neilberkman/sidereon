import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.ICGEMFormatReader;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider.NormalizedSphericalHarmonics;
import org.orekit.frames.FramesFactory;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;

/**
 * Orekit oracle for the ICGEM gravity field reader.
 *
 * <p>For each case, reads an ICGEM file with Orekit's ICGEMFormatReader,
 * builds a HolmesFeatherstoneAttractionModel truncated to the case's degree
 * and order, and prints, as JSON on standard output, the body-fixed gradient
 * (acceleration, m/s^2) and the non-central potential (m^2/s^2) at a spread
 * of body-fixed positions and epochs. For time-variable fields it also prints
 * the fully normalized coefficients Orekit evaluates at each epoch.</p>
 *
 * <p>usage: java IcgemOracle FIXTURE_DIR OREKIT_DATA SCRATCH_DIR</p>
 *
 * <p>SCRATCH_DIR receives synthetic_d2190.gfc, a degree 2190 field whose
 * coefficients come from the integer formula in {@link #syntheticValue}; the
 * Rust test regenerates the same text.</p>
 */
public class IcgemOracle {

    /** Seconds Orekit's ICGEM reader adds to a bare yyyymmdd epoch (it reads 12:00). */
    private static final double ICGEM1_NOON_SHIFT_S = 43200.0;

    /** Points as {radius km, latitude deg, longitude deg}. */
    private static final double[][] POINTS = {
        {6778.137, 0.0, 37.0},          // LEO, equator
        {6778.137, 51.6, -122.3},       // LEO, mid latitude
        {7178.137, -72.5, -170.0},      // 800 km, high southern latitude
        {7178.137, 89.99, 12.0},        // near the north pole
        {7000.0, -89.9999, 45.0},       // very near the south pole
        {6371.0, 30.0, 60.0},           // Earth surface, inside the reference sphere
        {26560.0, 55.0, 100.0},         // GNSS altitude
        {42164.0, 0.1, -75.0},          // geostationary
    };

    public static void main(final String[] args) throws IOException {
        final File fixtures = new File(args[0]);
        final File orekitData = new File(args[1]);
        final File scratch = new File(args[2]);
        writeSynthetic(scratch.toPath().resolve("synthetic_d2190.gfc"));

        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(orekitData));
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(fixtures));
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(scratch));

        final AbsoluteDate j2000 = AbsoluteDate.J2000_EPOCH;
        final List<AbsoluteDate> staticEpoch = List.of(j2000);
        final List<AbsoluteDate> icgem2Epochs = List.of(
            tt(2003, 3, 1, 0, 0, 0.0),
            tt(2005, 6, 15, 12, 29, 0.0),
            tt(2007, 9, 20, 12, 34, 56.0),
            tt(2012, 7, 4, 3, 0, 0.0),
            tt(2024, 11, 22, 0, 0, 0.0));
        final List<AbsoluteDate> icgem1Epochs = List.of(
            tt(2004, 7, 1, 0, 0, 0.0),
            tt(2005, 1, 1, 0, 0, 0.0),
            tt(2011, 3, 11, 5, 46, 24.0),
            tt(2020, 2, 29, 18, 0, 0.0));
        final int[] allPoints = {0, 1, 2, 3, 4, 5, 6, 7};
        final int[] fewPoints = {0, 3, 6};
        final int[][] icgem2Coefficients = {{2, 0}, {2, 1}, {2, 2}, {3, 0}, {4, 3}};
        final int[][] icgem1Coefficients = {{2, 0}, {3, 1}, {4, 4}};

        final StringBuilder out = new StringBuilder();
        out.append("  \"cases\": [\n");
        final List<String> cases = new ArrayList<>();
        cases.add(runCase("egm96_d36", "egm96_d36.gfc", 36, 36, staticEpoch, 0.0, allPoints, new int[0][]));
        cases.add(runCase("egm96_d36_o20", "egm96_d36.gfc", 36, 20, staticEpoch, 0.0, allPoints, new int[0][]));
        cases.add(runCase("egm96_d12", "egm96_d36.gfc", 12, 12, staticEpoch, 0.0, allPoints, new int[0][]));
        cases.add(runCase("egm96_zonal_d8", "egm96_d36.gfc", 8, 0, staticEpoch, 0.0, allPoints, new int[0][]));
        cases.add(runCase("unnormalized_d8", "unnormalized_d8.gfc", 8, 8, staticEpoch, 0.0, allPoints, new int[0][]));
        cases.add(runCase("synthetic_icgem2", "synthetic_icgem2.gfc", 6, 6, icgem2Epochs, 0.0, fewPoints,
                          icgem2Coefficients));
        cases.add(runCase("synthetic_icgem2_d4_o2", "synthetic_icgem2.gfc", 4, 2, icgem2Epochs, 0.0, fewPoints,
                          new int[][] {{2, 0}, {2, 1}, {2, 2}, {3, 0}}));
        cases.add(runCase("synthetic_icgem1", "synthetic_icgem1.gfc", 5, 5, icgem1Epochs, ICGEM1_NOON_SHIFT_S,
                          fewPoints, icgem1Coefficients));
        cases.add(runCase("synthetic_d2190", "synthetic_d2190.gfc", 2190, 2190, staticEpoch, 0.0, allPoints,
                          new int[0][]));
        cases.add(runCase("synthetic_d2190_o300", "synthetic_d2190.gfc", 2190, 300, staticEpoch, 0.0, fewPoints,
                          new int[0][]));
        cases.add(runCase("synthetic_d360", "synthetic_d2190.gfc", 360, 360, staticEpoch, 0.0, allPoints,
                          new int[0][]));
        out.append(String.join(",\n", cases)).append("\n  ]\n");
        System.out.print(out);
    }

    private static AbsoluteDate tt(final int y, final int mo, final int d, final int h, final int mi, final double s) {
        return new AbsoluteDate(y, mo, d, h, mi, s, TimeScalesFactory.getTT());
    }

    private static String runCase(final String name, final String file, final int degree, final int order,
                                  final List<AbsoluteDate> epochs, final double orekitShift,
                                  final int[] points, final int[][] coefficients) {
        GravityFieldFactory.clearPotentialCoefficientsReaders();
        GravityFieldFactory.addPotentialCoefficientsReader(new ICGEMFormatReader(Pattern.quote(file), false));
        final NormalizedSphericalHarmonicsProvider provider = GravityFieldFactory.getNormalizedProvider(degree, order);
        final HolmesFeatherstoneAttractionModel model =
            new HolmesFeatherstoneAttractionModel(FramesFactory.getGCRF(), provider);
        final double mu = provider.getMu();

        final StringBuilder sb = new StringBuilder();
        sb.append("    {\n");
        sb.append("      \"name\": \"").append(name).append("\",\n");
        sb.append("      \"file\": \"").append(file).append("\",\n");
        sb.append("      \"degree\": ").append(degree).append(",\n");
        sb.append("      \"order\": ").append(order).append(",\n");
        sb.append("      \"orekit_epoch_shift_s\": ").append(orekitShift).append(",\n");
        sb.append("      \"samples\": [\n");
        final List<String> samples = new ArrayList<>();
        for (final AbsoluteDate epoch : epochs) {
            final AbsoluteDate orekitDate = epoch.shiftedBy(orekitShift);
            final double epochTt = epoch.durationFrom(AbsoluteDate.J2000_EPOCH);
            for (final int p : points) {
                final double[] point = POINTS[p];
                final double lat = Math.toRadians(point[1]);
                final double lon = Math.toRadians(point[2]);
                final double rM = point[0] * 1000.0;
                final Vector3D position = new Vector3D(rM * Math.cos(lat) * Math.cos(lon),
                                                       rM * Math.cos(lat) * Math.sin(lon),
                                                       rM * Math.sin(lat));
                final double[] gradient = model.gradient(orekitDate, position, mu);
                final double potential = model.nonCentralPart(orekitDate, position, mu);
                samples.add(String.format(Locale.ROOT,
                    "        {\"epoch_tt_s\": %s, \"point\": %d, \"position_m\": [%s, %s, %s], "
                        + "\"acceleration_m_s2\": [%s, %s, %s], \"potential_m2_s2\": %s}",
                    d(epochTt), p, d(position.getX()), d(position.getY()), d(position.getZ()),
                    d(gradient[0]), d(gradient[1]), d(gradient[2]), d(potential)));
            }
        }
        sb.append(String.join(",\n", samples)).append("\n      ],\n");
        sb.append("      \"coefficients\": [\n");
        final List<String> rows = new ArrayList<>();
        for (final AbsoluteDate epoch : epochs) {
            if (coefficients.length == 0) {
                break;
            }
            final NormalizedSphericalHarmonics harmonics = provider.onDate(epoch.shiftedBy(orekitShift));
            final double epochTt = epoch.durationFrom(AbsoluteDate.J2000_EPOCH);
            for (final int[] nm : coefficients) {
                rows.add(String.format(Locale.ROOT,
                    "        {\"epoch_tt_s\": %s, \"degree\": %d, \"order\": %d, \"c\": %s, \"s\": %s}",
                    d(epochTt), nm[0], nm[1],
                    d(harmonics.getNormalizedCnm(nm[0], nm[1])), d(harmonics.getNormalizedSnm(nm[0], nm[1]))));
            }
        }
        sb.append(String.join(",\n", rows));
        sb.append(rows.isEmpty() ? "" : "\n").append("      ]\n");
        sb.append("    }");
        return sb.toString();
    }

    /** Shortest decimal that reads back as the same double. */
    private static String d(final double value) {
        return Double.toString(value);
    }

    /** SplitMix64 finalizer. */
    private static long mix(final long seed) {
        long z = seed + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /**
     * Text of a synthetic fully normalized coefficient: an integer mantissa in
     * [-1e6, 1e6] from SplitMix64 of (n, m, which) times a power of ten near
     * 1e-5 / n^2 (Kaula's rule), with C20 set to EGM96's value.
     */
    static String syntheticValue(final int n, final int m, final int which) {
        if (which == 1 && m == 0) {
            return "0.0";
        }
        if (n == 2 && m == 0) {
            return "-0.484165371736E-03";
        }
        final long key = ((long) n << 32) | ((long) m << 1) | which;
        final long mantissa = Long.remainderUnsigned(mix(key), 2000001L) - 1000000L;
        final int digits = Long.toString((long) n * n).length();
        return mantissa + "e" + (-(10 + digits));
    }

    private static void writeSynthetic(final Path path) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.US_ASCII)) {
            w.write("product_type gravity_field\n");
            w.write("modelname SYNTHETIC_D2190\n");
            w.write("earth_gravity_constant 0.3986004415E+15\n");
            w.write("radius 0.6378136300E+07\n");
            w.write("max_degree 2190\n");
            w.write("errors no\n");
            w.write("tide_system tide_free\n");
            w.write("end_of_head\n");
            w.write("gfc 0 0 1.0 0.0\n");
            w.write("gfc 1 0 0.0 0.0\n");
            w.write("gfc 1 1 0.0 0.0\n");
            for (int n = 2; n <= 2190; ++n) {
                for (int m = 0; m <= n; ++m) {
                    w.write("gfc " + n + " " + m + " " + syntheticValue(n, m, 0) + " "
                            + syntheticValue(n, m, 1) + "\n");
                }
            }
        }
    }
}
