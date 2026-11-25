import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.util.FastMath;
import org.orekit.bodies.CelestialBody;
import org.orekit.bodies.JPLEphemeridesLoader;
import org.orekit.bodies.JPLEphemeridesLoader.EphemerisType;
import org.orekit.data.DirectoryCrawler;
import org.orekit.data.LazyLoadedDataContext;
import org.orekit.frames.Frame;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeOffset;
import org.orekit.time.TimeScale;
import org.orekit.utils.TimeStampedPVCoordinates;

public final class DeOracle {
    private static final String NAME = "lnxp1990.440";
    private static final double J2000 = 2451545.0;
    private static final double DAY = 86400.0;

    private DeOracle() {}

    private static LazyLoadedDataContext context(Path directory) {
        LazyLoadedDataContext context = new LazyLoadedDataContext();
        context.getDataProvidersManager().addProvider(new DirectoryCrawler(directory.toFile()));
        return context;
    }

    private static JPLEphemeridesLoader loader(LazyLoadedDataContext context) {
        return new JPLEphemeridesLoader(NAME, EphemerisType.SUN,
            context.getDataProvidersManager(), context.getTimeScales(), context.getFrames().getGCRF());
    }

    private static final class Bodies {
        private final TimeScale tdb;
        private final AbsoluteDate origin;
        private final Frame gcrf;
        private final CelestialBody sun;
        private final CelestialBody moon;

        private Bodies(LazyLoadedDataContext context) {
            tdb = context.getTimeScales().getTDB();
            origin = new AbsoluteDate(2000, 1, 1, 12, 0, 0.0, tdb);
            gcrf = context.getFrames().getGCRF();
            sun = context.getCelestialBodies().getSun();
            moon = context.getCelestialBodies().getMoon();
        }

        private String row(double seconds) {
            AbsoluteDate date = new AbsoluteDate(origin, new TimeOffset(seconds), tdb);
            List<String> values = new ArrayList<>();
            for (CelestialBody body : new CelestialBody[] {sun, moon}) {
                Vector3D p = body.getPosition(date, gcrf);
                TimeStampedPVCoordinates pv = body.getPVCoordinates(date, gcrf);
                values.add(vector(p));
                values.add(vector(pv.getPosition()));
                values.add(vector(pv.getVelocity()));
            }
            return "    [" + bits(seconds) + ", " + String.join(", ", values) + "]";
        }
    }

    private static String bits(double value) {
        return String.format("\"%016x\"", Double.doubleToRawLongBits(value));
    }

    private static String vector(Vector3D value) {
        return "[" + bits(value.getX()) + ", " + bits(value.getY()) + ", " + bits(value.getZ()) + "]";
    }

    private static String object(Map<String, Double> values) {
        List<String> pairs = new ArrayList<>();
        values.forEach((key, value) -> pairs.add("    \"" + key + "\": " + bits(value)));
        return "{\n" + String.join(",\n", pairs) + "\n  }";
    }

    private static String version(Class<?> type, String artifact) {
        String jar = type.getProtectionDomain().getCodeSource().getLocation().getPath();
        String prefix = "/" + artifact + "-";
        return jar.substring(jar.lastIndexOf(prefix) + prefix.length(), jar.length() - 4);
    }

    public static void main(String[] args) throws IOException {
        Path directory = Path.of(args[0], "DE-440-ephemerides");
        Path output = Path.of(args[1]);
        byte[] bytes = Files.readAllBytes(directory.resolve(NAME));
        ByteBuffer data = ByteBuffer.wrap(bytes);
        if (Integer.toUnsignedLong(data.getInt(2840)) > 1 << 15) {
            data.order(ByteOrder.LITTLE_ENDIAN);
        }
        int words = 2;
        for (int slot = 0; slot < 13; slot++) {
            int offset = slot == 12 ? 2844 : 2696 + 12 * slot;
            words += (slot == 11 ? 2 : 3) * data.getInt(offset + 4) * data.getInt(offset + 8);
        }
        int recordSize = 8 * words;
        int count = bytes.length / recordSize - 2;
        TreeSet<Integer> records = new TreeSet<>();
        TreeSet<Double> epochs = new TreeSet<>();
        double coverageStart = (data.getDouble(2652) - J2000) * DAY;
        double coverageEnd = (data.getDouble(2660) - J2000) * DAY;
        for (int index : new int[] {0, count / 5, 2 * count / 5, 3 * count / 5, 4 * count / 5, count - 2}) {
            for (int neighbor = Math.max(0, index - 2); neighbor < Math.min(count, index + 4); neighbor++) {
                records.add(neighbor);
            }
            double start = (data.getDouble((index + 2) * recordSize) - J2000) * DAY;
            for (double days : new double[] {0, 4, 8, 12, 16, 28, 32, 36, 48, 60, 64}) {
                double boundary = start + days * DAY;
                for (double offset : new double[] {-0.000125, 0, 0.000125, 73457.12345678}) {
                    double epoch = boundary + offset;
                    if (epoch >= coverageStart && epoch <= coverageEnd &&
                        epoch >= start && epoch <= start + 64 * DAY) {
                        epochs.add(epoch);
                    }
                }
                if (boundary >= coverageStart && boundary <= coverageEnd &&
                    boundary >= start && boundary <= start + 64 * DAY) {
                    if (boundary > start) {
                        epochs.add(Math.nextDown(boundary));
                    }
                    if (boundary < start + 64 * DAY) {
                        epochs.add(Math.nextUp(boundary));
                    }
                }
            }
        }

        // The orbit fit uses the GPS G03 arc on 2020-06-25.
        for (int index = 0; index < count; index++) {
            int offset = (index + 2) * recordSize;
            if (data.getDouble(offset) <= 2459026.5 && data.getDouble(offset + 8) >= 2459025.5) {
                for (int neighbor = index - 1; neighbor <= index + 1; neighbor++) {
                    records.add(neighbor);
                }
            }
        }

        ByteArrayOutputStream subset = new ByteArrayOutputStream();
        subset.write(bytes, 0, 2 * recordSize);
        for (int index : records) {
            subset.write(bytes, (index + 2) * recordSize, recordSize);
        }
        Files.createDirectories(output);
        Files.write(output.resolve(NAME), subset.toByteArray());

        LazyLoadedDataContext full = context(directory);
        LazyLoadedDataContext reduced = context(output);
        JPLEphemeridesLoader fullLoader = loader(full);
        JPLEphemeridesLoader reducedLoader = loader(reduced);
        Map<String, Double> constants = new TreeMap<>();
        for (int slot = 0; slot < 400; slot++) {
            String name = new String(bytes, 252 + 6 * slot, 6, StandardCharsets.US_ASCII).trim();
            if (name.isEmpty()) {
                break;
            }
            constants.put(name, fullLoader.getLoadedConstant(name));
        }
        for (String name : new String[] {"AU", "EMRAT"}) {
            constants.put(name, fullLoader.getLoadedConstant(name));
        }
        constants.forEach((name, value) -> {
            if (!bits(value).equals(bits(reducedLoader.getLoadedConstant(name)))) {
                throw new IllegalStateException("Constant differs: " + name);
            }
        });
        Map<String, Double> gms = new TreeMap<>();
        for (EphemerisType type : EphemerisType.values()) {
            gms.put(type.name(), fullLoader.getLoadedGravitationalCoefficient(type));
        }

        Bodies fullBodies = new Bodies(full);
        Bodies subsetBodies = new Bodies(reduced);
        List<String> rows = new ArrayList<>();
        for (double epoch : epochs) {
            String row = fullBodies.row(epoch);
            String subsetRow = subsetBodies.row(epoch);
            if (!row.equals(subsetRow)) {
                throw new IllegalStateException("State differs at " + epoch + "\n" + row + "\n" + subsetRow);
            }
            // A warm cache can select the preceding endpoint at interior starts.
            if (epoch != coverageEnd && (epoch - coverageStart) % (4 * DAY) == 0) {
                row = new Bodies(context(directory)).row(epoch);
                if (!row.equals(new Bodies(context(output)).row(epoch))) {
                    throw new IllegalStateException("State differs at granule start " + epoch);
                }
            }
            rows.add(row);
        }
        String json = "{\n  \"orekit\": \"" + version(AbsoluteDate.class, "orekit") +
            "\",\n  \"hipparchus\": \"" + version(FastMath.class, "hipparchus-core") +
            "\",\n  \"orekit_data_commit\": \"" + args[2] +
            "\",\n  \"version\": " + data.getInt(2840) +
            ",\n  \"record_size\": " + recordSize +
            ",\n  \"data_records\": " + records.size() +
            ",\n  \"coverage_tdb_s\": [" + bits(coverageStart) + ", " + bits(coverageEnd) + "]" +
            ",\n  \"au_m\": " + bits(fullLoader.getLoadedAstronomicalUnit()) +
            ",\n  \"emrat\": " + bits(fullLoader.getLoadedEarthMoonMassRatio()) +
            ",\n  \"constants\": " + object(constants) +
            ",\n  \"gm_m3_s2\": " + object(gms) +
            ",\n  \"states\": [\n" + String.join(",\n", rows) + "\n  ]\n}\n";
        Files.writeString(output.resolve("de440_orekit.json"), json);
    }
}
