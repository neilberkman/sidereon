/*
 * Orekit oracle for the GNSS attitude laws of sidereon-core's
 * astro::attitude::gnss module.
 *
 * Every case is a Keplerian orbit (circular or eccentric) and a Sun moving
 * uniformly on a circle in the ecliptic, both in closed form, in EME2000. The same closed forms are
 * evaluated by the Rust test from the parameters written here, so both sides
 * see the same satellite states and Sun positions; the states at every
 * sample instant are also written so the Rust side can check that.
 *
 * For each case and block, one Orekit provider from org.orekit.gnss.attitude
 * is evaluated at the sample instants in increasing order, as a propagation
 * would call it. The output per instant is the body X and Y axes in EME2000
 * and the yaw angle psi defined by X . n = -sin(psi), Y . n = -cos(psi), with
 * n the unit orbit normal r x v. The GPS and GLONASS providers keep turn spans
 * between calls; for them the yaw from a new provider at each instant is also
 * written ("fresh_yaw"), with the largest difference between that provider's
 * axes and the axes its yaw defines ("fresh_axes_from_yaw_max_error").
 *
 * Usage: java -cp <orekit and hipparchus jars>:<classes> GnssAttitudeOracle
 *        <orekit-data directory> <provenance JSON object>
 * Prints the fixture JSON to standard output.
 */

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.TreeSet;

import org.hipparchus.CalculusFieldElement;
import org.hipparchus.geometry.euclidean.threed.FieldVector3D;
import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.util.FastMath;
import org.orekit.attitudes.Attitude;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.gnss.attitude.BeidouGeo;
import org.orekit.gnss.attitude.BeidouIGSO;
import org.orekit.gnss.attitude.BeidouMeo;
import org.orekit.gnss.attitude.GNSSAttitudeProvider;
import org.orekit.gnss.attitude.GPSBlockIIA;
import org.orekit.gnss.attitude.GPSBlockIIF;
import org.orekit.gnss.attitude.GPSBlockIIR;
import org.orekit.gnss.attitude.Galileo;
import org.orekit.gnss.attitude.Glonass;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.FieldAbsoluteDate;
import org.orekit.utils.ExtendedPositionProvider;
import org.orekit.utils.PVCoordinates;
import org.orekit.utils.PVCoordinatesProvider;
import org.orekit.utils.TimeStampedPVCoordinates;

public final class GnssAttitudeOracle {

    static final double GM = 3.986004418e14;
    static final double AU = 1.495978707e11;
    static final double OBLIQUITY = FastMath.toRadians(23.4392911);
    static final double SUN_RATE = 2.0 * FastMath.PI / (365.2422 * 86400.0);
    static final AbsoluteDate T0 = AbsoluteDate.J2000_EPOCH;

    static Frame frame;

    /**
     * Keplerian orbit in the plane of P (ascending node) and Q. Circular (e = 0):
     * r = a (cos u P + sin u Q), u = u0 + n t. Eccentric: the argument of
     * perigee from the node is argp, the mean anomaly u0 + n t, the eccentric
     * anomaly from Kepler's equation by Newton's method from M + e sin M until
     * the step no longer changes it, and u = argp + true anomaly.
     */
    static final class Circular implements PVCoordinatesProvider {
        final double a, inclination, raan, u0, n, e, argp;
        final Vector3D p, q;

        Circular(double a, double inclination, double raan, double u0) {
            this(a, inclination, raan, u0, 0.0, 0.0);
        }

        Circular(double a, double inclination, double raan, double u0, double e, double argp) {
            this.a = a;
            this.inclination = inclination;
            this.raan = raan;
            this.u0 = u0;
            this.e = e;
            this.argp = argp;
            this.n = FastMath.sqrt(GM / (a * a * a));
            this.p = new Vector3D(FastMath.cos(raan), FastMath.sin(raan), 0.0);
            this.q = new Vector3D(-FastMath.cos(inclination) * FastMath.sin(raan),
                                  FastMath.cos(inclination) * FastMath.cos(raan),
                                  FastMath.sin(inclination));
        }

        PVCoordinates at(double t) {
            if (e == 0.0) {
                double u = u0 + n * t;
                double cu = FastMath.cos(u), su = FastMath.sin(u);
                double av = a * n;
                Vector3D r = new Vector3D(a * cu, p, a * su, q);
                Vector3D v = new Vector3D(-av * su, p, av * cu, q);
                Vector3D acc = new Vector3D(-n * n, r);
                return new PVCoordinates(r, v, acc);
            }
            double m = u0 + n * t;
            double ea = m + e * FastMath.sin(m);
            for (int k = 0; k < 100; k++) {
                double step = (ea - e * FastMath.sin(ea) - m) / (1.0 - e * FastMath.cos(ea));
                ea -= step;
                if (FastMath.abs(step) <= 1e-16 * FastMath.max(FastMath.abs(ea), 1.0)) {
                    break;
                }
            }
            double nu = 2.0 * FastMath.atan2(FastMath.sqrt(1.0 + e) * FastMath.sin(0.5 * ea),
                                             FastMath.sqrt(1.0 - e) * FastMath.cos(0.5 * ea));
            double u = argp + nu;
            double semiLatus = a * (1.0 - e * e);
            double radius = semiLatus / (1.0 + e * FastMath.cos(nu));
            double h = FastMath.sqrt(GM / semiLatus);
            double vr = h * e * FastMath.sin(nu), vt = h * (1.0 + e * FastMath.cos(nu));
            double cu = FastMath.cos(u), su = FastMath.sin(u);
            Vector3D er = new Vector3D(cu, p, su, q);
            Vector3D et = new Vector3D(-su, p, cu, q);
            Vector3D r = new Vector3D(radius, er);
            Vector3D v = new Vector3D(vr, er, vt, et);
            Vector3D acc = new Vector3D(-GM / (radius * radius * radius), r);
            return new PVCoordinates(r, v, acc);
        }

        @Override
        public TimeStampedPVCoordinates getPVCoordinates(AbsoluteDate date, Frame f) {
            if (f != frame) {
                throw new IllegalStateException("unexpected frame " + f);
            }
            return new TimeStampedPVCoordinates(date, at(date.durationFrom(T0)));
        }
    }

    /** Sun on a circle in the ecliptic: lambda = lambda0 + rate t. */
    static final class Sun implements ExtendedPositionProvider {
        final double lambda0;

        Sun(double lambda0) {
            this.lambda0 = lambda0;
        }

        PVCoordinates at(double t) {
            double l = lambda0 + SUN_RATE * t;
            double cl = FastMath.cos(l), sl = FastMath.sin(l);
            double ce = FastMath.cos(OBLIQUITY), se = FastMath.sin(OBLIQUITY);
            Vector3D s = new Vector3D(AU * cl, AU * sl * ce, AU * sl * se);
            Vector3D v = new Vector3D(-AU * SUN_RATE * sl, AU * SUN_RATE * cl * ce, AU * SUN_RATE * cl * se);
            Vector3D acc = new Vector3D(-SUN_RATE * SUN_RATE, s);
            return new PVCoordinates(s, v, acc);
        }

        @Override
        public TimeStampedPVCoordinates getPVCoordinates(AbsoluteDate date, Frame f) {
            if (f != frame) {
                throw new IllegalStateException("unexpected frame " + f);
            }
            return new TimeStampedPVCoordinates(date, at(date.durationFrom(T0)));
        }

        @Override
        public Vector3D getPosition(AbsoluteDate date, Frame f) {
            return getPVCoordinates(date, f).getPosition();
        }

        @Override
        public <T extends CalculusFieldElement<T>> FieldVector3D<T> getPosition(FieldAbsoluteDate<T> date, Frame f) {
            if (f != frame) {
                throw new IllegalStateException("unexpected frame " + f);
            }
            T l = date.durationFrom(T0).multiply(SUN_RATE).add(lambda0);
            double ce = FastMath.cos(OBLIQUITY), se = FastMath.sin(OBLIQUITY);
            return new FieldVector3D<>(l.cos().multiply(AU), l.sin().multiply(AU * ce), l.sin().multiply(AU * se));
        }
    }

    interface ProviderFactory {
        GNSSAttitudeProvider create(Sun sun);
    }

    static final class Block {
        /** Series label: the IGS name, with the PRN or bias where a block appears more than once. */
        final String label;
        final String igsName;
        final String orekitClass;
        final String lawJson;
        final ProviderFactory factory;
        /** Whether the provider caches turn spans between calls. */
        final boolean caching;

        Block(String igsName, String orekitClass, String lawJson, ProviderFactory factory) {
            this(igsName, igsName, orekitClass, lawJson, factory);
        }

        Block(String label, String igsName, String orekitClass, String lawJson, ProviderFactory factory) {
            this.label = label;
            this.igsName = igsName;
            this.orekitClass = orekitClass;
            this.lawJson = lawJson;
            this.factory = factory;
            this.caching = orekitClass.startsWith("GPSBlock") || orekitClass.equals("Glonass");
        }
    }

    static String num(double x) {
        if (Double.isNaN(x) || Double.isInfinite(x)) {
            throw new IllegalStateException("non-finite value");
        }
        return Double.toString(x);
    }

    /** RAAN putting the Sun at elevation beta above the plane at time t. */
    static double raanFor(Sun sun, double inclination, double beta, double t) {
        return raanFor(sun, inclination, beta, t, false);
    }

    /**
     * RAAN putting the Sun at elevation beta above the plane at time t; `other`
     * takes the second solution, on which beta changes the other way.
     */
    static double raanFor(Sun sun, double inclination, double beta, double t, boolean other) {
        Vector3D s = sun.at(t).getPosition().normalize();
        double k = (FastMath.sin(beta) - FastMath.cos(inclination) * s.getZ()) / FastMath.sin(inclination);
        double r = FastMath.hypot(s.getX(), s.getY());
        // n = (sin i sin O, -sin i cos O, cos i); s . n = sin i (sx sin O - sy cos O) + cos i sz
        //   = sin i r sin(O - phi) + cos i sz with phi = atan2(sy, sx).
        double phi = FastMath.atan2(s.getY(), s.getX());
        return other ? phi + FastMath.PI - FastMath.asin(k / r) : phi + FastMath.asin(k / r);
    }

    /** Sun elevation above the plane of c at time t. */
    static double betaOf(Circular c, Sun sun, double t) {
        PVCoordinates pv = c.at(t);
        Vector3D n = Vector3D.crossProduct(pv.getPosition(), pv.getVelocity()).normalize();
        return FastMath.asin(Vector3D.dotProduct(sun.at(t).getPosition().normalize(), n));
    }

    /** u0 placing the satellite at argument `fromNoon` from the Sun's in-plane projection at time t. */
    static double u0For(Sun sun, double a, double inclination, double raan, double fromNoon, double t) {
        Circular c = new Circular(a, inclination, raan, 0.0);
        Vector3D s = sun.at(t).getPosition();
        double uNoon = FastMath.atan2(Vector3D.dotProduct(s, c.q), Vector3D.dotProduct(s, c.p));
        return uNoon + fromNoon - c.n * t;
    }

    /**
     * An eccentric orbit whose perigee lies `perigeeFromNoon` from the Sun's
     * in-plane projection at t = 0 and whose mean anomaly at t = 0 is m0.
     */
    static Circular eccentric(Sun sun, double a, double inclination, double raan, double e,
                              double perigeeFromNoon, double m0) {
        Circular c = new Circular(a, inclination, raan, 0.0);
        Vector3D s = sun.at(0.0).getPosition();
        double uNoon = FastMath.atan2(Vector3D.dotProduct(s, c.q), Vector3D.dotProduct(s, c.p));
        return new Circular(a, inclination, raan, m0, e, uNoon + perigeeFromNoon);
    }

    static final class Case {
        final String name;
        final String description;
        final Circular orbit;
        final Sun sun;
        final List<Block> blocks;
        final double[] times;

        Case(String name, String description, Circular orbit, Sun sun, List<Block> blocks, double[] times) {
            this.name = name;
            this.description = description;
            this.orbit = orbit;
            this.sun = sun;
            this.blocks = blocks;
            this.times = times;
        }
    }

    /** Argument from noon of the satellite at time t. */
    static double fromNoon(Circular c, Sun sun, double t) {
        Vector3D r = c.at(t).getPosition();
        Vector3D v = c.at(t).getVelocity();
        Vector3D n = Vector3D.crossProduct(r, v).normalize();
        Vector3D s = sun.at(t).getPosition();
        Vector3D sp = s.subtract(new Vector3D(Vector3D.dotProduct(s, n), n));
        return FastMath.atan2(Vector3D.dotProduct(Vector3D.crossProduct(sp, r), n), Vector3D.dotProduct(sp, r));
    }

    /**
     * Times from t0 to t1 at `step`, plus `denseStep` samples around each noon
     * crossing (from `noonBefore` before to `noonAfter` after) and each midnight
     * crossing (likewise) in [t0, t1]. A zero width leaves that crossing out.
     */
    static double[] grid(Circular c, Sun sun, double t0, double t1, double step, double denseStep,
                         double noonBefore, double noonAfter, double midnightBefore, double midnightAfter) {
        TreeSet<Double> times = new TreeSet<>();
        for (double t = t0; t <= t1 + 1e-9; t += step) {
            times.add(t);
        }
        if (denseStep > 0) {
            double scan = 10.0;
            double prev = fromNoon(c, sun, t0);
            for (double t = t0 + scan; t <= t1; t += scan) {
                double cur = fromNoon(c, sun, t);
                boolean noon = prev < 0 && cur >= 0;
                boolean midnight = prev > 2.0 && cur < -2.0;
                if (noon || midnight) {
                    double centre = t - scan * (noon ? cur / (cur - prev) : 0.5);
                    double before = noon ? noonBefore : midnightBefore;
                    double after = noon ? noonAfter : midnightAfter;
                    if (before + after > 0) {
                        double first = Math.ceil((centre - before) / denseStep) * denseStep;
                        for (double d = first; d <= centre + after; d += denseStep) {
                            if (d >= t0 && d <= t1) {
                                times.add(d);
                            }
                        }
                    }
                }
                prev = cur;
            }
        }
        double[] out = new double[times.size()];
        int i = 0;
        for (double t : times) {
            out[i++] = t;
        }
        return out;
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(new File(args[0])));
        frame = FramesFactory.getEME2000();
        String provenance = args[1];

        AbsoluteDate past = AbsoluteDate.PAST_INFINITY;
        AbsoluteDate future = AbsoluteDate.FUTURE_INFINITY;

        final int iiaPrn = 1;
        final double iiaRate = GPSBlockIIA.getDefaultYawRate(iiaPrn);
        List<Block> gps = new ArrayList<>();
        gps.add(new Block("BLOCK IIA", "GPSBlockIIA",
            "{\"law\": \"GpsIIA\", \"prn\": " + iiaPrn + ", \"yaw_rate_rad_s\": " + num(iiaRate)
                + ", \"yaw_bias_rad\": " + num(GPSBlockIIA.DEFAULT_YAW_BIAS)
                + ", \"shadow_limit_rad\": " + num(FastMath.toRadians(180.0 - 13.25)) + "}",
            sun -> new GPSBlockIIA(iiaRate, GPSBlockIIA.DEFAULT_YAW_BIAS, past, future, sun, frame)));
        gps.add(new Block("BLOCK IIR-B", "GPSBlockIIR",
            "{\"law\": \"GpsIIR\", \"yaw_rate_rad_s\": " + num(GPSBlockIIR.DEFAULT_YAW_RATE) + "}",
            sun -> new GPSBlockIIR(GPSBlockIIR.DEFAULT_YAW_RATE, past, future, sun, frame)));
        gps.add(new Block("BLOCK IIF", "GPSBlockIIF",
            "{\"law\": \"GpsIIF\", \"yaw_rate_rad_s\": " + num(GPSBlockIIF.DEFAULT_YAW_RATE)
                + ", \"yaw_bias_rad\": " + num(GPSBlockIIF.DEFAULT_YAW_BIAS)
                + ", \"shadow_limit_rad\": " + num(FastMath.toRadians(180.0 - 13.25)) + "}",
            sun -> new GPSBlockIIF(GPSBlockIIF.DEFAULT_YAW_RATE, GPSBlockIIF.DEFAULT_YAW_BIAS, past, future, sun, frame)));

        // GPS IIA at further PRNs (0.2 and 0.0815 deg/s) and with a yaw bias of
        // -0.5 deg (the sign PRN 23 carried in Kouba's eclips.f until 2013).
        List<Block> gpsIia = new ArrayList<>();
        for (int prn : new int[] {11, 14}) {
            final double rate = GPSBlockIIA.getDefaultYawRate(prn);
            gpsIia.add(new Block("BLOCK IIA PRN " + prn, "BLOCK IIA", "GPSBlockIIA",
                "{\"law\": \"GpsIIA\", \"prn\": " + prn + ", \"yaw_rate_rad_s\": " + num(rate)
                    + ", \"yaw_bias_rad\": " + num(GPSBlockIIA.DEFAULT_YAW_BIAS)
                    + ", \"shadow_limit_rad\": " + num(FastMath.toRadians(180.0 - 13.25)) + "}",
                sun -> new GPSBlockIIA(rate, GPSBlockIIA.DEFAULT_YAW_BIAS, past, future, sun, frame)));
        }
        {
            final double rate = GPSBlockIIA.getDefaultYawRate(23);
            final double bias = -GPSBlockIIA.DEFAULT_YAW_BIAS;
            gpsIia.add(new Block("BLOCK IIA PRN 23 BIAS -0.5 DEG", "BLOCK IIA", "GPSBlockIIA",
                "{\"law\": \"GpsIIA\", \"prn\": 23, \"yaw_rate_rad_s\": " + num(rate)
                    + ", \"yaw_bias_rad\": " + num(bias)
                    + ", \"shadow_limit_rad\": " + num(FastMath.toRadians(180.0 - 13.25)) + "}",
                sun -> new GPSBlockIIA(rate, bias, past, future, sun, frame)));
        }

        List<Block> glonass = new ArrayList<>();
        glonass.add(new Block("GLONASS-M", "Glonass",
            "{\"law\": \"GlonassM\", \"yaw_rate_rad_s\": " + num(Glonass.DEFAULT_YAW_RATE)
                + ", \"shadow_limit_rad\": " + num(FastMath.toRadians(180.0 - 14.20)) + "}",
            sun -> new Glonass(Glonass.DEFAULT_YAW_RATE, past, future, sun, frame)));

        final double galileoA = 29599800.0;
        final double galileoN = FastMath.sqrt(GM / (galileoA * galileoA * galileoA));
        // Orekit's Galileo law applies below beta_0 = atan(mu_rate / yaw_rate); this
        // yaw rate makes beta_0 the published 2 deg on this circular orbit.
        final double iovRate = galileoN / FastMath.tan(FastMath.toRadians(2.0));
        List<Block> galileo = new ArrayList<>();
        galileo.add(new Block("GALILEO-1", "Galileo",
            "{\"law\": \"GalileoIov\", \"beta_x_rad\": " + num(FastMath.toRadians(15.0))
                + ", \"beta_y_rad\": " + num(FastMath.toRadians(2.0)) + ", \"orekit_yaw_rate_rad_s\": " + num(iovRate) + "}",
            sun -> new Galileo(iovRate, past, future, sun, frame)));
        galileo.add(new Block("GALILEO-2", "Galileo",
            "{\"law\": \"GalileoFoc\", \"orekit_yaw_rate_rad_s\": " + num(Galileo.DEFAULT_YAW_RATE) + "}",
            sun -> new Galileo(Galileo.DEFAULT_YAW_RATE, past, future, sun, frame)));

        List<Block> bdsMeo = new ArrayList<>();
        bdsMeo.add(new Block("BEIDOU-2M", "BeidouMeo", "{\"law\": \"Beidou2\"}",
            sun -> new BeidouMeo(past, future, sun, frame)));
        bdsMeo.add(new Block("BEIDOU-3M-CAST", "Galileo",
            "{\"law\": \"Beidou3Cast\", \"orekit_yaw_rate_rad_s\": " + num(Galileo.DEFAULT_YAW_RATE) + "}",
            sun -> new Galileo(Galileo.DEFAULT_YAW_RATE, past, future, sun, frame)));
        bdsMeo.add(new Block("BEIDOU-3M-SECM", "Galileo",
            "{\"law\": \"Beidou3Secm\", \"orekit_yaw_rate_rad_s\": " + num(Galileo.DEFAULT_YAW_RATE) + "}",
            sun -> new Galileo(Galileo.DEFAULT_YAW_RATE, past, future, sun, frame)));

        List<Block> bdsIgso = new ArrayList<>();
        bdsIgso.add(new Block("BEIDOU-2I", "BeidouIGSO", "{\"law\": \"Beidou2\"}",
            sun -> new BeidouIGSO(past, future, sun, frame)));
        bdsIgso.add(new Block("BEIDOU-3I", "Galileo",
            "{\"law\": \"Beidou3Cast\", \"orekit_yaw_rate_rad_s\": " + num(Galileo.DEFAULT_YAW_RATE) + "}",
            sun -> new Galileo(Galileo.DEFAULT_YAW_RATE, past, future, sun, frame)));

        List<Block> bdsGeo = new ArrayList<>();
        bdsGeo.add(new Block("BEIDOU-2G", "BeidouGeo", "{\"law\": \"OrbitNormal\"}",
            sun -> new BeidouGeo(past, future, sun, frame)));

        List<Case> cases = new ArrayList<>();
        final double gpsA = 26559700.0, gpsI = FastMath.toRadians(55.0);
        final double gloA = 25508200.0, gloI = FastMath.toRadians(64.8);
        final double galI = FastMath.toRadians(56.0);
        final double meoA = 27906100.0, meoI = FastMath.toRadians(55.0);
        final double geoA = 42164200.0, igsoI = FastMath.toRadians(55.0), geoI = FastMath.toRadians(0.5);
        final Sun sun = new Sun(1.0);

        // GPS III: Montenbruck et al. (2026, Advances in Space Research 77(4):4901)
        // recommend the Galileo IOV law with beta_x = 15 deg, which Orekit's Galileo
        // fixes, and beta_y = 5.8 deg. Orekit's Galileo law applies below beta_0 =
        // atan(mu_rate / yaw_rate) and takes sin(beta_0) as the auxiliary Sun
        // vector's cross-track limit; this yaw rate makes beta_0 5.8 deg on a
        // circular GPS orbit.
        final double gpsN = FastMath.sqrt(GM / (gpsA * gpsA * gpsA));
        final double gpsIiiRate = gpsN / FastMath.tan(FastMath.toRadians(5.8));
        List<Block> gpsIii = new ArrayList<>();
        gpsIii.add(new Block("BLOCK IIIA", "Galileo",
            "{\"law\": \"GalileoIov\", \"beta_x_rad\": " + num(FastMath.toRadians(15.0))
                + ", \"beta_y_rad\": " + num(FastMath.toRadians(5.8)) + ", \"orekit_yaw_rate_rad_s\": " + num(gpsIiiRate) + "}",
            s -> new Galileo(gpsIiiRate, past, future, s, frame)));

        // Orbits start 60 deg before noon; t = 0 is J2000. Dense steps resolve
        // the turns of the laws that have them; BeiDou-2 and the published-law
        // blocks are compared on coarse grids.
        addFamily(cases, "gps", gpsA, gpsI, sun, gps, 20.0, true);
        addFamily(cases, "glonass", gloA, gloI, sun, glonass, 20.0, true);
        addFamily(cases, "galileo", galileoA, galI, sun, galileo, 30.0, false);
        addFamily(cases, "bds_meo", meoA, meoI, sun, bdsMeo, 120.0, false);
        addFamily(cases, "bds_igso", geoA, igsoI, sun, bdsIgso, 240.0, false);

        // GPS IIA variants: low beta of both signs (noon turns reversed by the bias
        // for beta in (0, 0.5] deg with +0.5 deg and [-0.5, 0) deg with -0.5 deg),
        // and Earth-shadow crossings with their recovery.
        {
            double period = 2.0 * FastMath.PI / FastMath.sqrt(GM / (gpsA * gpsA * gpsA));
            double[] betas = {0.3, -0.3, -2.5, 8.0, 0.45};
            for (double betaDeg : betas) {
                double beta = FastMath.toRadians(betaDeg);
                double raan = raanFor(sun, gpsI, beta, 0.0);
                Circular c = new Circular(gpsA, gpsI, raan, u0For(sun, gpsA, gpsI, raan, FastMath.toRadians(-60.0), 0.0));
                String suffix = String.format(Locale.ROOT, "%.2f", FastMath.abs(betaDeg)).replace('.', 'p');
                cases.add(new Case("gps_iia_beta_" + (betaDeg < 0 ? "m" : "p") + suffix,
                    "GPS IIA variants, beta " + betaDeg + " deg: noon and midnight turns, dense windows",
                    c, sun, gpsIia, grid(c, sun, 0.0, period, period / 24.0, 20.0, 600, 3400, 1800, 3600)));
            }
        }

        // GPS III on circular orbits at the beta angles of the paper's Table 1
        // (-5.7, -3.3, -0.4 deg), their opposites and 5.9 deg (beyond the 5.8 deg
        // limit): one orbit with dense windows over the collinearity regions
        // (about 1800 s either side of noon and midnight); and beta crossing zero
        // at mid-arc over two days.
        {
            double period = 2.0 * FastMath.PI / gpsN;
            for (double betaDeg : new double[] {-5.7, -3.3, -0.4, 0.4, 3.3, 5.7, 5.9}) {
                double beta = FastMath.toRadians(betaDeg);
                double raan = raanFor(sun, gpsI, beta, 0.0);
                Circular c = new Circular(gpsA, gpsI, raan, u0For(sun, gpsA, gpsI, raan, FastMath.toRadians(-60.0), 0.0));
                cases.add(new Case("gps_iii_" + caseSuffix(betaDeg),
                    "GPS III, beta " + betaDeg + " deg: noon and midnight collinearity regions, dense windows",
                    c, sun, gpsIii, grid(c, sun, 0.0, period, period / 24.0, 30.0, 2100, 2100, 2100, 2100)));
            }
            double span = 2.0 * 86400.0;
            double raan = raanFor(sun, gpsI, 0.0, 0.5 * span);
            Circular c = new Circular(gpsA, gpsI, raan, u0For(sun, gpsA, gpsI, raan, FastMath.toRadians(-60.0), 0.0));
            cases.add(new Case("gps_iii_beta_crossing", "GPS III, beta crossing zero at mid-arc, two days",
                c, sun, gpsIii, grid(c, sun, 0.0, span, period / 12.0, 60.0, 2100, 2100, 2100, 2100)));
        }

        // Eccentric orbits: e = 0.02 for every family, and 0.1 and 0.3 for GPS,
        // perigee placed at different angles from noon.
        {
            Object[][] ecc = {
                {"gps", gpsA, gpsI, gps, 0.02, 20.0},
                {"glonass", gloA, gloI, glonass, 0.02, 20.0},
                {"galileo", galileoA, galI, galileo, 0.02, 30.0},
                {"bds_meo", meoA, meoI, bdsMeo, 0.02, 120.0},
                {"bds_igso", geoA, igsoI, bdsIgso, 0.02, 240.0},
                {"gps", gpsA, gpsI, gps, 0.1, 20.0},
                {"gps", gpsA, gpsI, gps, 0.3, 20.0},
            };
            double[][] spec = {{0.3, 70.0, 0.4}, {-2.5, 200.0, 2.0}, {8.0, 300.0, 4.5}};
            for (Object[] f : ecc) {
                String family = (String) f[0];
                double a = (Double) f[1], inc = (Double) f[2], e = (Double) f[4], dense = (Double) f[5];
                @SuppressWarnings("unchecked")
                List<Block> blocks = (List<Block>) f[3];
                double period = 2.0 * FastMath.PI / FastMath.sqrt(GM / (a * a * a));
                for (double[] sp : spec) {
                    double beta = FastMath.toRadians(sp[0]);
                    double raan = raanFor(sun, inc, beta, 0.0);
                    Circular c = eccentric(sun, a, inc, raan, e, FastMath.toRadians(sp[1]), sp[2]);
                    String name = family + "_e" + String.format(Locale.ROOT, "%.2f", e).replace('.', 'p')
                        + "_" + caseSuffix(sp[0]);
                    cases.add(new Case(name,
                        "eccentricity " + e + ", beta " + sp[0] + " deg: noon and midnight turns, dense windows",
                        c, sun, blocks, grid(c, sun, 0.0, period, period / 24.0, dense, 1800, 3400, 1800, 3600)));
                }
            }
        }

        // BeiDou-2 orbit-normal switching: beta passes 4.1 and 3.9 deg mid-arc,
        // of either sign and with |beta| decreasing or increasing.
        for (String family : new String[] {"bds_meo", "bds_igso"}) {
            boolean meo = family.equals("bds_meo");
            double a = meo ? meoA : geoA;
            double span = 6.0 * 86400.0;
            for (double betaDeg : new double[] {4.0, -4.0}) {
                for (boolean other : new boolean[] {false, true}) {
                    double raan = raanFor(sun, meoI, FastMath.toRadians(betaDeg), 0.5 * span, other);
                    Circular c = new Circular(a, meoI, raan, u0For(sun, a, meoI, raan, FastMath.toRadians(-60.0), 0.0));
                    double b0 = FastMath.toDegrees(betaOf(c, sun, 0.0));
                    double b1 = FastMath.toDegrees(betaOf(c, sun, span));
                    boolean decreasing = FastMath.abs(b1) < FastMath.abs(b0);
                    String name = family + "_mode_switch" + (betaDeg < 0 ? "_negative" : "")
                        + (decreasing ? "" : "_rising");
                    cases.add(new Case(name,
                        String.format(Locale.ROOT, "beta from %.3f to %.3f deg through 4.1 and 3.9 deg at mid-arc, 6 days at 900 s", b0, b1),
                        c, sun, meo ? bdsMeo : bdsIgso, grid(c, sun, 0.0, span, 900.0, 0.0, 0.0, 0.0, 0.0, 0.0)));
                }
            }
        }

        {
            double raan = 0.3;
            Circular c = new Circular(geoA, geoI, raan, 0.0);
            cases.add(new Case("bds_geo", "geostationary, one day at 600 s", c, sun, bdsGeo,
                grid(c, sun, 0.0, 86400.0, 600.0, 0.0, 0.0, 0.0, 0.0, 0.0)));
        }

        StringBuilder out = new StringBuilder();
        out.append("{\n  \"_source\": ").append(provenance).append(",\n");
        out.append("  \"model\": {\"gm_m3_s2\": ").append(num(GM)).append(", \"au_m\": ").append(num(AU))
           .append(", \"obliquity_rad\": ").append(num(OBLIQUITY)).append(", \"sun_rate_rad_s\": ").append(num(SUN_RATE))
           .append(", \"frame\": \"EME2000\", \"t0\": \"J2000 epoch (TT)\"},\n");
        out.append("  \"cases\": [\n");
        for (int ci = 0; ci < cases.size(); ci++) {
            Case cs = cases.get(ci);
            out.append("    {\"name\": \"").append(cs.name).append("\", \"description\": \"").append(cs.description)
               .append("\",\n     \"orbit\": {\"a_m\": ").append(num(cs.orbit.a)).append(", \"inclination_rad\": ")
               .append(num(cs.orbit.inclination)).append(", \"raan_rad\": ").append(num(cs.orbit.raan))
               .append(", \"u0_rad\": ").append(num(cs.orbit.u0)).append(", \"e\": ").append(num(cs.orbit.e))
               .append(", \"argp_rad\": ").append(num(cs.orbit.argp)).append("}, \"sun_lambda0_rad\": ")
               .append(num(cs.sun.lambda0)).append(",\n");
            out.append("     \"times_s\": [");
            for (int i = 0; i < cs.times.length; i++) {
                out.append(i == 0 ? "" : ",").append(num(cs.times[i]));
            }
            out.append("],\n     \"states\": [");
            for (int i = 0; i < cs.times.length; i++) {
                PVCoordinates pv = cs.orbit.at(cs.times[i]);
                Vector3D s = cs.sun.at(cs.times[i]).getPosition();
                out.append(i == 0 ? "" : ",").append("[")
                   .append(vec(pv.getPosition())).append(",").append(vec(pv.getVelocity())).append(",").append(vec(s)).append("]");
            }
            out.append("],\n     \"blocks\": [\n");
            for (int bi = 0; bi < cs.blocks.size(); bi++) {
                Block b = cs.blocks.get(bi);
                GNSSAttitudeProvider provider = b.factory.create(cs.sun);
                out.append("       {\"block\": \"").append(b.label).append("\", \"igs_block\": \"").append(b.igsName)
                   .append("\", \"orekit_provider\": \"")
                   .append(b.orekitClass).append("\", \"law\": ").append(b.lawJson).append(",\n        \"attitude\": [");
                for (int i = 0; i < cs.times.length; i++) {
                    AbsoluteDate date = T0.shiftedBy(cs.times[i]);
                    Attitude att = provider.getAttitude(cs.orbit, date, frame);
                    Rotation rot = att.getRotation();
                    Vector3D x = rot.applyInverseTo(Vector3D.PLUS_I);
                    Vector3D y = rot.applyInverseTo(Vector3D.PLUS_J);
                    Vector3D z = rot.applyInverseTo(Vector3D.PLUS_K);
                    PVCoordinates pv = cs.orbit.at(cs.times[i]);
                    Vector3D n = Vector3D.crossProduct(pv.getPosition(), pv.getVelocity()).normalize();
                    if (Vector3D.distance(z, pv.getPosition().normalize().negate()) > 1e-12) {
                        throw new IllegalStateException("body Z is not nadir at " + cs.name + " t=" + cs.times[i]);
                    }
                    double yaw = FastMath.atan2(-Vector3D.dotProduct(x, n), -Vector3D.dotProduct(y, n));
                    out.append(i == 0 ? "" : ",").append("[").append(num(yaw)).append(",").append(vec(x))
                       .append(",").append(vec(y)).append("]");
                }
                out.append("]");
                if (b.caching) {
                    // A new provider per instant: no turn span carried between calls.
                    // Only the yaw is written; the axes are checked here against the
                    // axes the yaw defines with the orbit (Z = -r, X . n = -sin(yaw),
                    // Y . n = -cos(yaw)) and the largest deviation is recorded.
                    out.append(",\n        \"fresh_yaw\": [");
                    double worst = 0.0;
                    for (int i = 0; i < cs.times.length; i++) {
                        AbsoluteDate date = T0.shiftedBy(cs.times[i]);
                        Rotation rot = b.factory.create(cs.sun).getAttitude(cs.orbit, date, frame).getRotation();
                        Vector3D x = rot.applyInverseTo(Vector3D.PLUS_I);
                        Vector3D y = rot.applyInverseTo(Vector3D.PLUS_J);
                        Vector3D z = rot.applyInverseTo(Vector3D.PLUS_K);
                        PVCoordinates pv = cs.orbit.at(cs.times[i]);
                        Vector3D r = pv.getPosition().normalize();
                        Vector3D n = Vector3D.crossProduct(pv.getPosition(), pv.getVelocity()).normalize();
                        Vector3D along = Vector3D.crossProduct(n, r);
                        double yaw = FastMath.atan2(-Vector3D.dotProduct(x, n), -Vector3D.dotProduct(y, n));
                        Vector3D xYaw = new Vector3D(-FastMath.sin(yaw), n, FastMath.cos(yaw), along);
                        Vector3D yYaw = new Vector3D(-FastMath.cos(yaw), n, -FastMath.sin(yaw), along);
                        worst = FastMath.max(worst, FastMath.max(Vector3D.distance(x, xYaw),
                            FastMath.max(Vector3D.distance(y, yYaw), Vector3D.distance(z, r.negate()))));
                        out.append(i == 0 ? "" : ",").append(num(yaw));
                    }
                    if (worst > 1e-12) {
                        throw new IllegalStateException("fresh axes differ from their yaw by " + worst);
                    }
                    out.append("],\n        \"fresh_axes_from_yaw_max_error\": ").append(num(worst));
                }
                out.append("}").append(bi + 1 < cs.blocks.size() ? ",\n" : "\n");
            }
            out.append("     ]}").append(ci + 1 < cases.size() ? ",\n" : "\n");
        }
        out.append("  ]\n}\n");
        System.out.print(out);
    }

    static String vec(Vector3D v) {
        return num(v.getX()) + "," + num(v.getY()) + "," + num(v.getZ());
    }

    /**
     * The per-family cases: high beta over one orbit; small beta of both signs
     * with dense noon and midnight windows; one full orbit at small beta at the
     * dense step; a larger negative beta; a shadow-crossing beta; and beta
     * crossing zero.
     */
    static void addFamily(List<Case> cases, String family, double a, double inclination, Sun sun,
                          List<Block> blocks, double dense, boolean turns) {
        double period = 2.0 * FastMath.PI / FastMath.sqrt(GM / (a * a * a));
        double start = FastMath.toRadians(-60.0);
        double coarse = period / 24.0;
        // beta deg, base step, dense step, noon before/after, midnight before/after (s).
        // With turns, the windows cover the GPS IIA reversed noon turn (about 3000 s
        // at its yaw rate) and the shadow crossing plus 1800 s of recovery.
        double[][] specs = turns ? new double[][] {
            {35.0, period / 48.0, 0.0, 0, 0, 0, 0},
            {0.3, coarse, dense, 600, 3400, 1800, 3600},
            {-0.3, coarse, dense, 600, 3400, 1800, 3600},
            {1.2, 2.0 * dense, 0.0, 0, 0, 0, 0},
            {-2.5, coarse, dense, 600, 1200, 1800, 3600},
            {8.0, coarse, dense, 0, 0, 1800, 3600},
        } : new double[][] {
            {35.0, period / 48.0, 0.0, 0, 0, 0, 0},
            {0.3, coarse, dense, 1800, 1800, 1800, 1800},
            {-0.3, coarse, dense, 1800, 1800, 1800, 1800},
            {1.2, 2.0 * dense, 0.0, 0, 0, 0, 0},
            {-2.5, coarse, dense, 1500, 1500, 1500, 1500},
            {8.0, coarse, dense, 0, 0, 1800, 1800},
        };
        String[] descriptions = {
            "beta 35 deg: nominal yaw steering over one orbit",
            "beta +0.3 deg: noon and midnight turns, dense windows",
            "beta -0.3 deg: noon and midnight turns, dense windows",
            "beta +1.2 deg: one full orbit at a fine step",
            "beta -2.5 deg: noon and midnight turns, dense windows",
            "beta 8 deg: Earth-shadow entry and exit, dense midnight window",
        };
        for (int k = 0; k < specs.length; k++) {
            double[] sp = specs[k];
            double beta = FastMath.toRadians(sp[0]);
            double raan = raanFor(sun, inclination, beta, 0.0);
            Circular c = new Circular(a, inclination, raan, u0For(sun, a, inclination, raan, start, 0.0));
            cases.add(new Case(family + "_" + caseSuffix(sp[0]), descriptions[k], c, sun, blocks,
                grid(c, sun, 0.0, period, sp[1], sp[2], sp[3], sp[4], sp[5], sp[6])));
        }
        // beta crossing zero at mid-arc over two days
        double span = 2.0 * 86400.0;
        double raan = raanFor(sun, inclination, 0.0, 0.5 * span);
        Circular c = new Circular(a, inclination, raan, u0For(sun, a, inclination, raan, start, 0.0));
        double cw = turns ? 1200.0 : 900.0;
        cases.add(new Case(family + "_beta_crossing", "beta crossing zero at mid-arc, two days",
            c, sun, blocks, grid(c, sun, 0.0, span, 2.0 * coarse, 2.0 * dense, cw, cw, cw, cw)));
    }

    static String caseSuffix(double betaDeg) {
        String s = String.format(Locale.ROOT, "%.1f", Math.abs(betaDeg)).replace('.', 'p');
        return "beta_" + (betaDeg < 0 ? "m" : "p") + s;
    }
}
