package org.orekit.frames;

import java.io.File;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.data.PoissonSeries;
import org.orekit.data.PoissonSeriesParser;
import org.orekit.data.PolynomialParser;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.time.UTCScale;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;

public final class EarthOrientationOracle {
    private static final IERSConventions IERS = IERSConventions.IERS_2010;
    private static final double ARCSEC = Constants.ARC_SECONDS_TO_RADIANS;

    private static PoissonSeries parse(PoissonSeriesParser parser, String table) throws Exception {
        String name = "/assets/org/orekit/IERS-conventions/2010/" + table + ".txt";
        try (InputStream input = IERSConventions.class.getResourceAsStream(name)) {
            return parser.parse(input, name);
        }
    }

    private static void order(String series) throws Exception {
        double microArcsec = ARCSEC * 1.0e-6;
        List<PoissonSeries> components;
        if (series.equals("celestial")) {
            PoissonSeriesParser parser = new PoissonSeriesParser(17)
                .withPolynomialPart('t', PolynomialParser.Unit.MICRO_ARC_SECONDS)
                .withFirstDelaunay(4).withFirstPlanetary(9)
                .withSinCos(0, 2, microArcsec, 3, microArcsec);
            components = List.of(parse(parser, "tab5.2a"), parse(parser, "tab5.2b"),
                                 parse(parser, "tab5.2d"));
        } else {
            PoissonSeriesParser pole = new PoissonSeriesParser(13)
                .withOptionalColumn(1).withGamma(2).withFirstDelaunay(3);
            PoissonSeriesParser ut1 = new PoissonSeriesParser(11)
                .withOptionalColumn(1).withGamma(2).withFirstDelaunay(3);
            components = List.of(
                parse(pole.withSinCos(0, 10, microArcsec, 11, microArcsec), "tab8.2ab"),
                parse(pole.withSinCos(0, 12, microArcsec, 13, microArcsec), "tab8.2ab"),
                parse(ut1.withSinCos(0, 10, 1.0e-6, 11, 1.0e-6), "tab8.3ab"));
        }
        Field field = PoissonSeries.class.getDeclaredField("series");
        field.setAccessible(true);
        Map<Long, Object> terms = new HashMap<>();
        for (PoissonSeries component : components) {
            Map<?, ?> source = (Map<?, ?>) field.get(component);
            for (Object key : source.keySet()) {
                if (!terms.containsKey(key)) {
                    terms.put((Long) key, source.get(key));
                }
            }
        }
        for (long key : terms.keySet()) {
            System.out.println(Long.toUnsignedString(key, 16));
        }
    }

    private static Frame rawFrame(EOPHistory history) {
        Frame cirf = new Frame(FramesFactory.getGCRF(), new CIRFProvider(history), "CIRF", true);
        Frame tirf = new Frame(cirf, new TIRFProvider(history, TimeScalesFactory.getUT1(history)), "TIRF");
        return new Frame(tirf, new ITRFProvider(history), "ITRF");
    }

    private static double[] eop(EOPHistory history, AbsoluteDate date) {
        PoleCorrection pole = history.getPoleCorrection(date);
        double[] nro = history.getNonRotatinOriginNutationCorrection(date);
        return new double[] {pole.getXp(), pole.getYp(), history.getUT1MinusUTC(date),
                             history.getLOD(date), nro[0], nro[1]};
    }

    private static void row(String label, double... data) {
        System.out.print(label);
        for (double value : data) {
            System.out.print(" " + Double.toString(value));
        }
        System.out.println();
    }

    private static void pv(String label, PVCoordinates state) {
        Vector3D p = state.getPosition();
        Vector3D v = state.getVelocity();
        row(label, p.getX(), p.getY(), p.getZ(), v.getX(), v.getY(), v.getZ());
    }

    public static void main(String[] args) throws Exception {
        if (args.length > 1) {
            order(args[1]);
            return;
        }
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(new File(args[0])));
        UTCScale utc = TimeScalesFactory.getUTC();
        var scales = TimeScalesFactory.getTimeScales();
        EOPHistory factory = FramesFactory.getEOPHistory(IERS, false);
        EOPHistory factoryRaw = factory.getEOPHistoryWithoutCachedTidalCorrection();
        Frame factoryFrame = FramesFactory.getITRF(IERS, false);
        Frame factoryRawFrame = rawFrame(factoryRaw);
        Frame gcrf = FramesFactory.getGCRF();

        // Epochs cover historical UTC drift, modern leap seconds and a GNSS arc.
        int[] days = {38639, 41317, 56109, 57204, 57754, 59025};
        List<EOPEntry> entries = new ArrayList<>();
        for (EOPEntry e : factory.getEntries()) {
            for (int center : days) {
                if (e.getMjd() < center - 5 || e.getMjd() > center + 5) {
                    continue;
                }
                double xp = e.getX() / ARCSEC;
                double yp = e.getY() / ARCSEC;
                double dx = e.getDx() / ARCSEC;
                double dy = e.getDy() / ARCSEC;
                // The MJD 56109 window exercises the missing-rate reconstruction.
                double xRate = center == 56109 ? Double.NaN : e.getXRate();
                double yRate = center == 56109 ? Double.NaN : e.getYRate();
                row("sample", e.getMjd(), xp, yp, e.getUT1MinusUTC(), e.getLOD(),
                    dx, dy, xRate, yRate);
                entries.add(new EOPEntry(e.getMjd(), e.getUT1MinusUTC(), e.getLOD(),
                    ARCSEC * xp, ARCSEC * yp, xRate, yRate, e.getDdPsi(), e.getDdEps(),
                    ARCSEC * dx, ARCSEC * dy, e.getITRFType(), e.getDate(), e.getEopDataType()));
            }
        }
        EOPHistory simple = new EOPHistory(IERS, 3, entries, true, scales);
        EOPHistory raw = new EOPHistory(IERS, 3, entries, false, scales)
            .getEOPHistoryWithoutCachedTidalCorrection();
        Frame rawFrame = rawFrame(raw);
        var xys = IERS.getXYSpXY2Function(scales);
        var era = IERS.getEarthOrientationAngleFunction(TimeScalesFactory.getUT1(raw), scales.getTAI());

        PVCoordinates[] states = {
            new PVCoordinates(new Vector3D(6471.25, -1824.75, 792.0), new Vector3D(2.3125, 6.75, -3.5)),
            new PVCoordinates(new Vector3D(-22001.0, 9000.5, 12600.25), new Vector3D(-1.875, -2.125, -0.75)),
            new PVCoordinates(new Vector3D(7412.25, 41140.0, -512.5), new Vector3D(-3.0, 0.53125, 0.0078125)),
        };
        for (PVCoordinates state : states) {
            pv("state", state);
        }

        double maxAngle = 0.0;
        double maxRate = 0.0;
        for (int center : days) {
            List<double[]> times = new ArrayList<>(List.of(
                new double[] {-1, 86399.75}, new double[] {0, 0}, new double[] {0, 0.125},
                new double[] {0, 43199.25}, new double[] {0, 53347.8125},
                new double[] {1, 6289.0625}, new double[] {2, 69723.4375},
                new double[] {-5, 0}, new double[] {5, 0},
                new double[] {-6, 85518.75}, new double[] {5, 7.625}));
            if (center == 38639) {
                times.add(new double[] {-1, 86400.05});
                times.add(new double[] {-2, 86399.75});
            } else if (center == 41317) {
                times = List.of(new double[] {-1, 86400.05});
            } else if (center == 56109 || center == 57204 || center == 57754) {
                times.set(9, new double[] {-1, 86400.25});
                times.set(10, new double[] {-1, 86400.875});
            }
            for (double[] time : times) {
                int mjd = center + (int) time[0];
                AbsoluteDate date = AbsoluteDate.createMJDDate(mjd, time[1], utc);
                row("epoch", mjd, time[1]);
                row("elapsed", date.durationFrom(scales.getJ2000Epoch()),
                    date.durationFrom(AbsoluteDate.createMJDDate(51544, 43200, scales.getTAI())),
                    utc.offsetFromTAI(date).toDouble(),
                    date.durationFrom(AbsoluteDate.createMJDDate(mjd, 0, utc)));
                row("daily", eop(simple, date));
                double[] corrected = eop(raw, date);
                row("corrected", corrected);
                double[] xy = xys.value(date);
                double x = xy[0] + corrected[4];
                double y = xy[1] + corrected[5];
                row("cip", xy[0], xy[1], xy[2] - xy[0] * xy[1] / 2);
                row("corrected_cip", x, y, xy[2] - x * y / 2);
                row("angles", era.value(date), (-47e-6 * ARCSEC) * IERS.evaluateTC(date, scales));
                Transform transform = gcrf.getTransformTo(rawFrame, date);
                Rotation rotation = transform.getRotation();
                Vector3D rate = transform.getRotationRate();
                row("rotation", rotation.getQ0(), rotation.getQ1(), rotation.getQ2(), rotation.getQ3());
                double[][] m = rotation.getMatrix();
                row("matrix", m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
                row("rate", rate.getX(), rate.getY(), rate.getZ());
                for (PVCoordinates input : states) {
                    pv("pv", transform.transformPVCoordinates(input));
                    pv("inverse_pv", transform.getInverse().transformPVCoordinates(input));
                }

                Transform cached = gcrf.getTransformTo(factoryFrame, date);
                Transform uncached = gcrf.getTransformTo(factoryRawFrame, date);
                maxAngle = Math.max(maxAngle, Rotation.distance(cached.getRotation(), uncached.getRotation()));
                maxRate = Math.max(maxRate, cached.getRotationRate().subtract(uncached.getRotationRate()).getNorm());
            }
        }
        row("factory_difference", maxAngle, maxRate);
    }
}
