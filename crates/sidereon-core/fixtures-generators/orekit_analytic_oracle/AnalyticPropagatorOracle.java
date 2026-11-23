// Orekit oracle for the Eckstein-Hechler and Brouwer-Lyddane analytic
// propagators and their osculating/mean element conversions.
//
// Prints the body of a JSON object (everything after the "_source" member) to
// standard output. generate.sh wraps it with the version record.
//
// Every case starts from an osculating Cartesian state in EME2000 built from
// the listed Keplerian elements. The state is printed with Double.toString, which
// round-trips, so the Rust tests start from exactly the same doubles. Gravity is
// passed explicitly as mu, the reference radius and unnormalized C20..C60 on
// both sides.

import java.io.File;
import java.util.Locale;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.attitudes.FrameAlignedProvider;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.errors.OrekitException;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.CircularOrbit;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.Orbit;
import org.orekit.orbits.OrbitType;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.PropagationType;
import org.orekit.propagation.analytical.BrouwerLyddanePropagator;
import org.orekit.propagation.analytical.EcksteinHechlerPropagator;
import org.orekit.propagation.conversion.osc2mean.BrouwerLyddaneTheory;
import org.orekit.propagation.conversion.osc2mean.EcksteinHechlerTheory;
import org.orekit.propagation.conversion.osc2mean.FixedPointConverter;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.PVCoordinates;

public final class AnalyticPropagatorOracle {

    // Gravity: the core crate's MU_EARTH, RE_EARTH and C_n0 = -J_n of its
    // J2_EARTH..J6_EARTH constants, in SI units.
    static final double MU  = 3.986004418e14;
    static final double RE  = 6378137.0;
    static final double C20 = -1.08262668e-3;
    static final double C30 = 2.5326564853322355e-6;
    static final double C40 = 1.619621591367e-6;
    static final double C50 = 2.272960828686982e-7;
    static final double C60 = -5.406812391070848e-7;

    static final double MASS = 1000.0;

    // Orekit's defaults for the two converters.
    static final double EH_EPSILON = 1.0e-11;
    static final int EH_MAX_ITER = 100;
    static final double BL_EPSILON = 1.0e-13;
    static final int BL_MAX_ITER = 200;

    // Offsets from the initial epoch, seconds: up to seven days ahead and one day back.
    static final double[] DTS = {0.0, 600.0, 3600.0, 21600.0, 86400.0, 259200.0, 604800.0, -86400.0};

    static final class Case {
        final String name;
        final double aKm, e, iDeg, paDeg, raanDeg, mDeg, m2;
        Case(String name, double aKm, double e, double iDeg, double paDeg, double raanDeg, double mDeg, double m2) {
            this.name = name; this.aKm = aKm; this.e = e; this.iDeg = iDeg; this.paDeg = paDeg;
            this.raanDeg = raanDeg; this.mDeg = mDeg; this.m2 = m2;
        }
    }

    static final Case[] COMMON = {
        new Case("leo_sso_700km",        7078.137,  0.0012,  98.19,  90.0,  30.0,  10.0, 0.0),
        new Case("leo_iss_like",         6778.137,  0.0005,  51.64,  45.0, 120.0, 200.0, 0.0),
        new Case("leo_low_inclination",  7178.137,  0.005,   10.0,  270.0, 300.0,  60.0, 0.0),
        new Case("leo_polar_e0p02",      7500.0,    0.02,    90.0,    0.0,   0.0,   0.0, 0.0),
        new Case("leo_retrograde",       7200.0,    0.003,  120.0,  135.0, 200.0, 330.0, 0.0),
        new Case("leo_e0p08",            8000.0,    0.08,    40.0,  100.0,  50.0, 250.0, 0.0),
        new Case("leo_near_circular",    7000.0,    1.0e-6,  45.0,    0.0,   0.0,   0.0, 0.0),
        new Case("meo_gps_like",        26559.7,    0.01,    55.0,   30.0, 250.0, 100.0, 0.0),
        new Case("meo_glonass_like",    25508.0,    0.001,   64.8,  330.0,  15.0,  45.0, 0.0),
        new Case("meo_galileo_like",    29600.318,  0.0002,  56.0,    0.0,  80.0, 300.0, 0.0),
        new Case("igso_beidou_like",    42164.17,   0.002,   55.0,  180.0, 118.0,  90.0, 0.0),
        new Case("geo_inclined_1p5deg", 42164.17,   0.0003,   1.5,  270.0,  75.0, 180.0, 0.0),
    };

    static final Case[] BL_ONLY = {
        new Case("heo_molniya_critical", 26600.0,   0.72,    63.43, 270.0,  40.0,  10.0, 0.0),
        new Case("gto",                  24396.0,   0.73,     7.0,  178.0,  10.0,   0.0, 0.0),
        new Case("leo_critical",          7200.0,   0.001,   63.43,  60.0, 210.0,  80.0, 0.0),
        new Case("leo_equatorial",        7000.0,   0.001,    0.0,    0.0,   0.0,  30.0, 0.0),
        new Case("leo_low_inclination_m2", 7178.137, 0.005,  10.0,  270.0, 300.0,  60.0, 1.0e-14),
    };

    static final Case[] EH_ERRORS = {
        new Case("eh_eccentricity_too_large", 7500.0, 0.15, 50.0, 0.0, 0.0, 0.0, 0.0),
        new Case("eh_critical_inclination",   7200.0, 0.001, 63.43, 0.0, 0.0, 0.0, 0.0),
        new Case("eh_critical_retrograde",    7200.0, 0.001, 116.57, 0.0, 0.0, 0.0, 0.0),
        new Case("eh_equatorial",             7000.0, 0.001, 0.0, 0.0, 0.0, 30.0, 0.0),
        new Case("eh_inside_reference_sphere", 6000.0, 0.001, 45.0, 0.0, 0.0, 0.0, 0.0),
    };

    static final Case[] BL_ERRORS = {
        new Case("bl_inside_reference_sphere", 6000.0, 0.001, 45.0, 0.0, 0.0, 0.0, 0.0),
    };

    static Frame frame;
    static AbsoluteDate epoch;

    public static void main(String[] args) {
        Locale.setDefault(Locale.ROOT);
        String data = System.getenv("OREKIT_DATA");
        if (data == null) {
            throw new IllegalStateException("OREKIT_DATA is not set");
        }
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(new File(data)));
        frame = FramesFactory.getEME2000();
        epoch = new AbsoluteDate(2024, 3, 15, 0, 0, 0.0, TimeScalesFactory.getUTC());

        StringBuilder out = new StringBuilder();
        out.append("  \"gravity\": {\"mu_m3_s2\": ").append(num(MU))
           .append(", \"re_m\": ").append(num(RE))
           .append(", \"c20\": ").append(num(C20))
           .append(", \"c30\": ").append(num(C30))
           .append(", \"c40\": ").append(num(C40))
           .append(", \"c50\": ").append(num(C50))
           .append(", \"c60\": ").append(num(C60)).append("},\n");
        out.append("  \"epoch_utc\": \"").append(epoch.toString()).append("\",\n");
        out.append("  \"eckstein_hechler_defaults\": {\"tolerance\": ").append(num(EH_EPSILON))
           .append(", \"max_iterations\": ").append(EH_MAX_ITER).append("},\n");
        out.append("  \"brouwer_lyddane_defaults\": {\"tolerance\": ").append(num(BL_EPSILON))
           .append(", \"max_iterations\": ").append(BL_MAX_ITER).append("},\n");

        out.append("  \"eckstein_hechler\": [\n");
        for (int k = 0; k < COMMON.length; k++) {
            out.append(ehCase(COMMON[k])).append(k + 1 < COMMON.length ? ",\n" : "\n");
        }
        out.append("  ],\n");

        out.append("  \"brouwer_lyddane\": [\n");
        Case[] bl = new Case[COMMON.length + BL_ONLY.length];
        System.arraycopy(COMMON, 0, bl, 0, COMMON.length);
        System.arraycopy(BL_ONLY, 0, bl, COMMON.length, BL_ONLY.length);
        for (int k = 0; k < bl.length; k++) {
            out.append(blCase(bl[k])).append(k + 1 < bl.length ? ",\n" : "\n");
        }
        out.append("  ],\n");

        out.append("  \"errors\": [\n");
        StringBuilder errs = new StringBuilder();
        for (Case c : EH_ERRORS) {
            appendError(errs, c.name, "eckstein_hechler", c, EH_EPSILON, EH_MAX_ITER);
        }
        for (Case c : BL_ERRORS) {
            appendError(errs, c.name, "brouwer_lyddane", c, BL_EPSILON, BL_MAX_ITER);
        }
        appendError(errs, "eh_not_converged", "eckstein_hechler", COMMON[7], EH_EPSILON, 1);
        appendError(errs, "bl_not_converged", "brouwer_lyddane", COMMON[7], BL_EPSILON, 1);
        out.append(errs.substring(0, errs.length() - 2)).append("\n");
        out.append("  ]\n");
        System.out.print(out);
    }

    static CartesianOrbit initial(Case c) {
        KeplerianOrbit kep = new KeplerianOrbit(c.aKm * 1000.0, c.e, Math.toRadians(c.iDeg),
                                                Math.toRadians(c.paDeg), Math.toRadians(c.raanDeg),
                                                Math.toRadians(c.mDeg), PositionAngleType.MEAN,
                                                frame, epoch, MU);
        return new CartesianOrbit(kep.getPVCoordinates(), frame, MU);
    }

    /** The case's defining elements, read as mean elements. */
    static KeplerianOrbit defining(Case c) {
        return new KeplerianOrbit(c.aKm * 1000.0, c.e, Math.toRadians(c.iDeg),
                                  Math.toRadians(c.paDeg), Math.toRadians(c.raanDeg),
                                  Math.toRadians(c.mDeg), PositionAngleType.MEAN,
                                  frame, epoch, MU);
    }

    static String ehCase(Case c) {
        System.err.println("eckstein_hechler " + c.name);
        CartesianOrbit osc = initial(c);
        StringBuilder s = new StringBuilder();
        s.append("    {\"name\": \"").append(c.name).append("\",\n");
        s.append("     \"initial\": ").append(pv(osc.getPVCoordinates())).append(",\n");

        // Orekit's default tolerance first; when its fixed point does not reach it,
        // the next looser one, so every case records the tolerance it was run at.
        double eps = EH_EPSILON;
        CircularOrbit mean = null;
        int iterations = 0;
        StringBuilder refused = new StringBuilder();
        String error = "";
        while (mean == null && eps <= 1.0e-9) {
            FixedPointConverter conv = new FixedPointConverter(
                new EcksteinHechlerTheory(RE, MU, C20, C30, C40, C50, C60), eps, EH_MAX_ITER, 1.0);
            try {
                mean = (CircularOrbit) OrbitType.CIRCULAR.convertType(conv.convertToMean(osc));
                iterations = conv.getIterationsNb();
            } catch (OrekitException e) {
                refused.append(refused.length() == 0 ? "" : ", ").append(num(eps));
                error = e.getSpecifier().toString();
                eps = Double.parseDouble("1e" + (Math.round(Math.log10(eps)) + 1));
            }
        }
        s.append("     \"max_iterations\": ").append(EH_MAX_ITER).append(",\n");
        s.append("     \"orekit_not_converged_at\": [").append(refused).append("],\n");
        if (mean == null) {
            s.append("     \"orekit_conversion_error\": \"").append(error).append("\",\n");
        } else {
            s.append("     \"tolerance\": ").append(num(eps)).append(",\n");
            s.append("     \"mean\": ").append(circ(mean)).append(",\n");
            s.append("     \"mean_iterations\": ").append(iterations).append(",\n");
            EcksteinHechlerPropagator prop = new EcksteinHechlerPropagator(
                osc, FrameAlignedProvider.of(frame), MASS, RE, MU, C20, C30, C40, C50, C60,
                PropagationType.OSCULATING, eps, EH_MAX_ITER);
            s.append("     \"samples\": [\n");
            for (int k = 0; k < DTS.length; k++) {
                AbsoluteDate date = epoch.shiftedBy(DTS[k]);
                s.append("       {\"dt_s\": ").append(num(DTS[k]))
                 .append(", \"state\": ").append(pv(prop.propagateOrbit(date).getPVCoordinates()))
                 .append(", \"osculating\": ").append(circ(prop.getOsculatingCircularOrbit(date))).append("}")
                 .append(k + 1 < DTS.length ? ",\n" : "\n");
            }
            s.append("     ],\n");
        }

        // Mean to osculating: a propagator started from the defining elements as
        // mean circular elements.
        CircularOrbit meanIn = (CircularOrbit) OrbitType.CIRCULAR.convertType(defining(c));
        meanIn = new CircularOrbit(meanIn.getA(), meanIn.getCircularEx(), meanIn.getCircularEy(),
                                   meanIn.getI(), meanIn.getRightAscensionOfAscendingNode(),
                                   meanIn.getAlphaM(), PositionAngleType.MEAN, frame, epoch, MU);
        EcksteinHechlerPropagator fromMean = new EcksteinHechlerPropagator(
            meanIn, FrameAlignedProvider.of(frame), MASS, RE, MU, C20, C30, C40, C50, C60,
            PropagationType.MEAN);
        s.append("     \"from_mean\": {\"mean\": ").append(circ(meanIn)).append(", \"samples\": [\n");
        for (int k = 0; k < DTS.length; k++) {
            AbsoluteDate date = epoch.shiftedBy(DTS[k]);
            s.append("       {\"dt_s\": ").append(num(DTS[k]))
             .append(", \"state\": ").append(pv(fromMean.propagateOrbit(date).getPVCoordinates()))
             .append(", \"osculating\": ").append(circ(fromMean.getOsculatingCircularOrbit(date))).append("}")
             .append(k + 1 < DTS.length ? ",\n" : "\n");
        }
        s.append("     ]}\n");
        s.append("    }");
        return s.toString();
    }

    static String blCase(Case c) {
        System.err.println("brouwer_lyddane " + c.name);
        CartesianOrbit osc = initial(c);
        StringBuilder s = new StringBuilder();
        s.append("    {\"name\": \"").append(c.name).append("\",\n");
        s.append("     \"m2\": ").append(num(c.m2)).append(",\n");
        s.append("     \"initial\": ").append(pv(osc.getPVCoordinates())).append(",\n");

        double eps = BL_EPSILON;
        KeplerianOrbit mean = null;
        int iterations = 0;
        StringBuilder refused = new StringBuilder();
        String error = "";
        while (mean == null && eps <= 1.0e-9) {
            FixedPointConverter conv = new FixedPointConverter(
                new BrouwerLyddaneTheory(RE, MU, C20, C30, C40, C50, c.m2), eps, BL_MAX_ITER, 1.0);
            try {
                mean = (KeplerianOrbit) OrbitType.KEPLERIAN.convertType(conv.convertToMean(osc));
                iterations = conv.getIterationsNb();
            } catch (OrekitException e) {
                refused.append(refused.length() == 0 ? "" : ", ").append(num(eps));
                error = e.getSpecifier().toString();
                eps = Double.parseDouble("1e" + (Math.round(Math.log10(eps)) + 1));
            }
        }
        s.append("     \"max_iterations\": ").append(BL_MAX_ITER).append(",\n");
        s.append("     \"orekit_not_converged_at\": [").append(refused).append("],\n");
        if (mean == null) {
            s.append("     \"orekit_conversion_error\": \"").append(error).append("\",\n");
        } else {
            s.append("     \"tolerance\": ").append(num(eps)).append(",\n");
            s.append("     \"mean\": ").append(kep(mean)).append(",\n");
            s.append("     \"mean_iterations\": ").append(iterations).append(",\n");
            BrouwerLyddanePropagator prop = new BrouwerLyddanePropagator(
                osc, FrameAlignedProvider.of(frame), MASS, RE, MU, C20, C30, C40, C50,
                PropagationType.OSCULATING, c.m2, eps, BL_MAX_ITER);
            s.append("     \"samples\": [\n");
            for (int k = 0; k < DTS.length; k++) {
                KeplerianOrbit el = prop.propagateOrbit(epoch.shiftedBy(DTS[k]));
                s.append("       {\"dt_s\": ").append(num(DTS[k]))
                 .append(", \"state\": ").append(pv(el.getPVCoordinates()))
                 .append(", \"osculating\": ").append(kep(el)).append("}")
                 .append(k + 1 < DTS.length ? ",\n" : "\n");
            }
            s.append("     ],\n");
        }

        KeplerianOrbit meanIn = defining(c);
        BrouwerLyddanePropagator fromMean = new BrouwerLyddanePropagator(
            meanIn, FrameAlignedProvider.of(frame), MASS, RE, MU, C20, C30, C40, C50,
            PropagationType.MEAN, c.m2);
        s.append("     \"from_mean\": {\"mean\": ").append(kep(meanIn)).append(", \"samples\": [\n");
        for (int k = 0; k < DTS.length; k++) {
            KeplerianOrbit el = fromMean.propagateOrbit(epoch.shiftedBy(DTS[k]));
            s.append("       {\"dt_s\": ").append(num(DTS[k]))
             .append(", \"state\": ").append(pv(el.getPVCoordinates()))
             .append(", \"osculating\": ").append(kep(el)).append("}")
             .append(k + 1 < DTS.length ? ",\n" : "\n");
        }
        s.append("     ]}\n");
        s.append("    }");
        return s.toString();
    }

    static void appendError(StringBuilder s, String name, String theory, Case c, double eps, int maxIter) {
        CartesianOrbit osc = initial(c);
        String specifier;
        String message;
        try {
            if (theory.equals("eckstein_hechler")) {
                new EcksteinHechlerPropagator(osc, FrameAlignedProvider.of(frame), MASS, RE, MU,
                                              C20, C30, C40, C50, C60, PropagationType.OSCULATING,
                                              eps, maxIter);
            } else {
                new BrouwerLyddanePropagator(osc, FrameAlignedProvider.of(frame), MASS, RE, MU,
                                             C20, C30, C40, C50, PropagationType.OSCULATING, c.m2,
                                             eps, maxIter);
            }
            specifier = "NONE";
            message = "";
        } catch (OrekitException e) {
            specifier = e.getSpecifier().toString();
            message = e.getMessage();
        }
        s.append("    {\"name\": \"").append(name)
         .append("\", \"theory\": \"").append(theory)
         .append("\", \"tolerance\": ").append(num(eps))
         .append(", \"max_iterations\": ").append(maxIter)
         .append(", \"initial\": ").append(pv(osc.getPVCoordinates()))
         .append(", \"orekit_error\": \"").append(specifier)
         .append("\", \"orekit_message\": \"").append(message.replace("\\", "\\\\").replace("\"", "\\\""))
         .append("\"},\n");
    }

    static String pv(PVCoordinates pv) {
        Vector3D p = pv.getPosition();
        Vector3D v = pv.getVelocity();
        return "{\"position_m\": [" + num(p.getX()) + ", " + num(p.getY()) + ", " + num(p.getZ()) +
               "], \"velocity_m_s\": [" + num(v.getX()) + ", " + num(v.getY()) + ", " + num(v.getZ()) + "]}";
    }

    static String circ(CircularOrbit o) {
        return "{\"a_m\": " + num(o.getA()) + ", \"ex\": " + num(o.getCircularEx()) +
               ", \"ey\": " + num(o.getCircularEy()) + ", \"i\": " + num(o.getI()) +
               ", \"raan\": " + num(o.getRightAscensionOfAscendingNode()) +
               ", \"alpha_m\": " + num(o.getAlphaM()) + "}";
    }

    static String kep(KeplerianOrbit o) {
        return "{\"a_m\": " + num(o.getA()) + ", \"e\": " + num(o.getE()) +
               ", \"i\": " + num(o.getI()) + ", \"pa\": " + num(o.getPerigeeArgument()) +
               ", \"raan\": " + num(o.getRightAscensionOfAscendingNode()) +
               ", \"m\": " + num(o.getMeanAnomaly()) + "}";
    }

    static String num(double x) {
        if (!Double.isFinite(x)) {
            throw new IllegalStateException("non-finite value " + x);
        }
        return Double.toString(x);
    }
}
