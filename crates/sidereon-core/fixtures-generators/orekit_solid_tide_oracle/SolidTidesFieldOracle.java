// Oracle for the solid Earth tide corrections to the degree 2, 3 and 4
// geopotential coefficients, IERS Conventions (2010) Section 6.2.1 Steps 1 and
// 2 and Section 6.2.2 (Step 3), as Orekit's SolidTidesField forms them.
//
// SolidTidesField is package-private, so this class sits in its package. It is
// given everything it depends on, so the comparison isolates the formulas:
//
// - the Sun and Moon are fixed Earth-fixed positions supplied here (low-precision
//   Astronomical Almanac formulas rotated by GMST, realistic but not accurate;
//   both programs receive the same numbers), with the Sun and Moon GM of this
//   crate;
// - the gravity constants are this crate's EGM96 mu and reference radius;
// - everything is in km, km^3/s^2: SolidTidesField uses only the ratios GM/mu
//   and ae/r, so the Rust side forms the same ratios;
// - Step 2 is Orekit's IERS 2010 frequency-dependence function with TAI as the
//   UT1 scale, and the fundamental arguments Orekit formed are recorded;
// - the pole tide is off, and the field is evaluated for a tide-free and for a
//   zero-tide central field with Orekit's IERS 2010 permanent tide,
//   4.4228e-8 * -0.31460 * k20.
//
// For each epoch this prints the inputs and both coefficient sets as the
// elements of a JSON array; generate.sh wraps them. Doubles are printed with
// Double.toString, the shortest decimal that reads back to the same value.

package org.orekit.forces.gravity;

import java.util.Locale;

import org.hipparchus.CalculusFieldElement;
import org.hipparchus.geometry.euclidean.threed.FieldVector3D;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.bodies.CelestialBody;
import org.orekit.data.BodiesElements;
import org.orekit.data.DataContext;
import org.orekit.data.FundamentalNutationArguments;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider.NormalizedSphericalHarmonics;
import org.orekit.forces.gravity.potential.TideSystem;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.FieldAbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScales;
import org.orekit.time.TimeVectorFunction;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.TimeStampedPVCoordinates;

public class SolidTidesFieldOracle {

    // sidereon-core GM_SUN_KM3_S2, GM_MOON_KM3_S2, EGM96_MU_KM3_S2 and
    // EGM96_REFERENCE_RADIUS_KM.
    static final double GM_SUN = 132712440041.93938;
    static final double GM_MOON = 4902.800066;
    static final double MU = 398600.4415;
    static final double AE = 6378.1363;

    /** A tide-raising body at a position set before each evaluation. */
    static final class FixedBody implements CelestialBody {
        private final String name;
        private final double gm;
        private Vector3D position = Vector3D.ZERO;

        FixedBody(String name, double gm) {
            this.name = name;
            this.gm = gm;
        }

        void set(Vector3D position) {
            this.position = position;
        }

        public Frame getInertiallyOrientedFrame() {
            throw new UnsupportedOperationException();
        }

        public Frame getBodyOrientedFrame() {
            throw new UnsupportedOperationException();
        }

        public String getName() {
            return name;
        }

        public double getGM() {
            return gm;
        }

        @Override
        public Vector3D getPosition(AbsoluteDate date, Frame frame) {
            return position;
        }

        @Override
        public TimeStampedPVCoordinates getPVCoordinates(AbsoluteDate date, Frame frame) {
            return new TimeStampedPVCoordinates(date, position, Vector3D.ZERO);
        }

        public <T extends CalculusFieldElement<T>> FieldVector3D<T> getPosition(
                FieldAbsoluteDate<T> date, Frame frame) {
            throw new UnsupportedOperationException();
        }
    }

    static String d(double x) {
        return Double.toString(x);
    }

    static String vec(Vector3D v) {
        return "[" + d(v.getX()) + ", " + d(v.getY()) + ", " + d(v.getZ()) + "]";
    }

    static String coefficients(NormalizedSphericalHarmonics h) {
        StringBuilder out = new StringBuilder("[");
        boolean first = true;
        for (int n = 2; n <= 4; n++) {
            for (int m = 0; m <= n; m++) {
                if (!first) {
                    out.append(", ");
                }
                first = false;
                out.append(String.format(Locale.ROOT, "{\"degree\": %d, \"order\": %d, \"c\": %s, \"s\": %s}",
                    n, m, d(h.getNormalizedCnm(n, m)), d(h.getNormalizedSnm(n, m))));
            }
        }
        return out.append("]").toString();
    }

    public static void main(String[] args) {
        TimeScales scales = DataContext.getDefault().getTimeScales();
        TimeScale tai = scales.getTAI();
        TimeScale tt = scales.getTT();
        IERSConventions conventions = IERSConventions.IERS_2010;
        TimeVectorFunction step2 = conventions.getTideFrequencyDependenceFunction(tai, scales);
        FundamentalNutationArguments arguments = conventions.getNutationArguments(tai, scales);
        double permanentTide = conventions.getPermanentTide();
        Frame frame = FramesFactory.getGCRF();
        FixedBody sun = new FixedBody("Sun", GM_SUN);
        FixedBody moon = new FixedBody("Moon", GM_MOON);
        SolidTidesField tideFree = new SolidTidesField(conventions.getLoveNumbers(), step2,
            permanentTide, null, frame, AE, MU, TideSystem.TIDE_FREE, sun, moon);
        SolidTidesField zeroTide = new SolidTidesField(conventions.getLoveNumbers(), step2,
            permanentTide, null, frame, AE, MU, TideSystem.ZERO_TIDE, sun, moon);
        AbsoluteDate j2000 = new AbsoluteDate(2000, 1, 1, 12, 0, 0.0, tt);
        double d2r = Math.PI / 180.0;

        StringBuilder out = new StringBuilder();
        int count = 48;
        for (int k = 0; k < count; k++) {
            // Whole days from J2000.0 TT, from 1980 to 2045, and a day fraction
            // of (k * 613) mod 1024 parts in 1024.
            long days = -7305L + Math.round(k * (23742.0 / (count - 1)));
            int parts = (k * 613) % 1024;
            double fraction = parts / 1024.0;
            AbsoluteDate date = j2000.shiftedBy(days * 86400L).shiftedBy(fraction * 86400.0);
            BodiesElements e = arguments.evaluateAll(date);

            // Low-precision Sun and Moon (Astronomical Almanac, sections C and
            // D), equatorial of date, rotated to Earth-fixed axes by GMST.
            double t = days + fraction;
            double eps = (23.439 - 0.00000036 * t) * d2r;
            double g = (357.529 + 0.98560028 * t) * d2r;
            double q = 280.459 + 0.98564736 * t;
            double lam = (q + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * d2r;
            double rs = (1.00014 - 0.01671 * Math.cos(g) - 0.00014 * Math.cos(2 * g)) * 149597870.7;
            double lm = 218.316 + 13.176396 * t;
            double mm = (134.963 + 13.064993 * t) * d2r;
            double fm = (93.272 + 13.229350 * t) * d2r;
            double lamm = (lm + 6.289 * Math.sin(mm)) * d2r;
            double betm = 5.128 * Math.sin(fm) * d2r;
            double rm = 385001.0 - 20905.0 * Math.cos(mm);
            double gmst = e.getGamma() - Math.PI;
            Vector3D sunEq = new Vector3D(rs * Math.cos(lam), rs * Math.cos(eps) * Math.sin(lam),
                rs * Math.sin(eps) * Math.sin(lam));
            Vector3D moonEq = new Vector3D(rm * Math.cos(betm) * Math.cos(lamm),
                rm * (Math.cos(eps) * Math.cos(betm) * Math.sin(lamm) - Math.sin(eps) * Math.sin(betm)),
                rm * (Math.sin(eps) * Math.cos(betm) * Math.sin(lamm) + Math.cos(eps) * Math.sin(betm)));
            double c = Math.cos(gmst);
            double s = Math.sin(gmst);
            Vector3D sunFixed = new Vector3D(c * sunEq.getX() + s * sunEq.getY(),
                -s * sunEq.getX() + c * sunEq.getY(), sunEq.getZ());
            Vector3D moonFixed = new Vector3D(c * moonEq.getX() + s * moonEq.getY(),
                -s * moonEq.getX() + c * moonEq.getY(), moonEq.getZ());
            sun.set(sunFixed);
            moon.set(moonFixed);

            if (k > 0) {
                out.append(",\n");
            }
            out.append(String.format(Locale.ROOT,
                "    {\"jd_whole\": %s, \"tt_fraction\": %s, \"gamma\": %s, \"l\": %s, \"l_prime\": %s,"
                    + " \"f\": %s, \"d\": %s, \"omega\": %s, \"sun_km\": %s, \"moon_km\": %s,"
                    + " \"tide_free\": %s, \"zero_tide\": %s}",
                d(2451545.0 + days), d(fraction), d(e.getGamma()), d(e.getL()), d(e.getLPrime()),
                d(e.getF()), d(e.getD()), d(e.getOmega()), vec(sunFixed), vec(moonFixed),
                coefficients(tideFree.onDate(date)), coefficients(zeroTide.onDate(date))));
        }
        System.out.println(out);
    }
}
