/*
 * Bit-exact oracle for Orekit 13.1.10 org.orekit.forces.radiation.ECOM2
 * accelerations and FieldSpacecraftState<Gradient> state and parameter
 * partials (with position in m, velocity in m/s, and all ECOM2 coefficients
 * in m/s^2 as free parameters), combined with Orekit's own
 * SolarRadiationPressure.getLightingRatio in penumbra and umbra.
 *
 * Covers:
 * - truncations D0B0 (0, 0), D2B1 (1, 1), D4B1 (2, 1), and D6B3 (3, 2);
 * - spherical and WGS84 Earth, with and without the Moon;
 * - Sun elevation beta from near 0 deg (0.02 deg, 0.5 deg) to near +/-90 deg
 *   (-89 deg, 89.5 deg);
 * - orbit noon, orbit midnight, sunlit, Earth penumbra (entry, exit, grazing
 *   contact overshoot), Earth umbra, Moon partial, Moon annular (off-axis),
 *   and Moon total umbra;
 * - GPS, GLONASS, Galileo, and BeiDou MEO, IGSO, and GEO (equatorial and
 *   inclined) orbits.
 *
 * Every input coordinate, coefficient, value, and partial is recorded as raw
 * IEEE 754 binary64 hex bits.
 *
 * Usage: java -cp <orekit and hipparchus jars>:<classes> EcomOracle
 *        <orekit-data directory> <provenance JSON object>
 * Prints the fixture JSON to standard output.
 */

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.hipparchus.CalculusFieldElement;
import org.hipparchus.analysis.differentiation.Gradient;
import org.hipparchus.analysis.differentiation.GradientField;
import org.hipparchus.geometry.euclidean.threed.FieldVector3D;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.util.FastMath;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.forces.radiation.ECOM2;
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
import org.orekit.utils.FieldAbsolutePVCoordinates;
import org.orekit.utils.FieldPVCoordinates;
import org.orekit.utils.PVCoordinates;
import org.orekit.utils.TimeStampedPVCoordinates;

public final class EcomOracle {

    static final double M_PER_KM = 1000.0;
    static final double MU = 3.986004418e14;
    static final double AU_KM = 149597870.7;
    static final double EARTH_RADIUS_M = 6371000.0;
    static final double WGS84_FLATTENING = 1.0 / 298.257223563;
    static final double MOON_RADIUS_M = 1737400.0;
    static final AbsoluteDate T0 = AbsoluteDate.J2000_EPOCH;

    // Realistic GNSS coefficients in Orekit driver order:
    // [B0, Bcos..., Bsin..., D0, Dcos..., Dsin..., Y0], m/s^2.
    static final double[] COEFFS_D0B0 = {
        -1.0e-9, -1.0e-7, 5.0e-10
    };
    static final double[] COEFFS_D2B1 = {
        -1.0e-9, 3.0e-10, -4.0e-10,
        -1.0e-7, 2.0e-9, -5.0e-10,
        5.0e-10
    };
    static final double[] COEFFS_D4B1 = {
        -1.0e-9, 3.0e-10, -4.0e-10,
        -1.0e-7, 2.0e-9, 1.0e-10, -5.0e-10, 2.0e-10,
        5.0e-10
    };
    static final double[] COEFFS_D6B3 = {
        -1.0e-9, 3.0e-10, -1.5e-10, -4.0e-10, 2.5e-10,
        -1.0e-7, 2.0e-9, 1.0e-10, -8.0e-11, -5.0e-10, 2.0e-10, 6.0e-11,
        5.0e-10
    };

    static Frame gcrf;

    static final class FixedBody implements ExtendedPositionProvider {
        final Vector3D position;

        FixedBody(final Vector3D position) {
            this.position = position;
        }

        @Override
        public TimeStampedPVCoordinates getPVCoordinates(final AbsoluteDate date, final Frame f) {
            final Vector3D p = getPosition(date, f);
            return new TimeStampedPVCoordinates(date, new PVCoordinates(p, Vector3D.ZERO));
        }

        @Override
        public Vector3D getPosition(final AbsoluteDate date, final Frame f) {
            if (f == gcrf) {
                return position;
            }
            return gcrf.getStaticTransformTo(f, date).transformPosition(position);
        }

        @Override
        public <T extends CalculusFieldElement<T>> FieldVector3D<T> getPosition(
                final FieldAbsoluteDate<T> date, final Frame f) {
            final FieldVector3D<T> p = new FieldVector3D<>(date.getField(), position);
            if (f == gcrf) {
                return p;
            }
            return gcrf.getStaticTransformTo(f, date).transformPosition(p);
        }
    }

    static String hex(final double v) {
        return String.format(Locale.ROOT, "\"0x%016x\"", Double.doubleToRawLongBits(v));
    }

    static String hexVec3(final Vector3D v) {
        return "[" + hex(v.getX()) + ", " + hex(v.getY()) + ", " + hex(v.getZ()) + "]";
    }

    static String hexArr(final double[] a) {
        final StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(hex(a[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    static Vector3D toMetres(final Vector3D km) {
        return new Vector3D(km.getX() * M_PER_KM, km.getY() * M_PER_KM, km.getZ() * M_PER_KM);
    }

    static Vector3D clearMoonKm(final Vector3D sunKm) {
        final Vector3D uSun = sunKm.normalize();
        final Vector3D ref = FastMath.abs(uSun.getZ()) < 0.9 ? Vector3D.PLUS_K : Vector3D.PLUS_I;
        final Vector3D perp = Vector3D.crossProduct(uSun, ref).normalize();
        return perp.scalarMultiply(384400.0);
    }

    static Vector3D moonAtOffsetKm(final Vector3D satKm, final Vector3D sunKm,
                                   final double distKm, final double offAxisRad) {
        final Vector3D toSun = sunKm.subtract(satKm).normalize();
        final Vector3D ref = FastMath.abs(toSun.getZ()) < 0.9 ? Vector3D.PLUS_K : Vector3D.PLUS_I;
        final Vector3D side = Vector3D.crossProduct(toSun, ref).normalize();
        final Vector3D dir = new Vector3D(FastMath.cos(offAxisRad), toSun, FastMath.sin(offAxisRad), side);
        return satKm.add(dir.scalarMultiply(distKm));
    }

    static final class SampleSpec {
        final String label;
        final double du;
        final Vector3D moonKm; // null when includeMoon == false

        SampleSpec(final String label, final double du, final Vector3D moonKm) {
            this.label = label;
            this.du = du;
            this.moonKm = moonKm;
        }
    }

    static final class CaseSpec {
        final String name;
        final int nD;
        final int nB;
        final double[] coeffs;
        final double aKm;
        final double incRad;
        final double raanRad;
        final double uSunRad;
        final double betaRad;
        final double sunDistKm;
        final double earthFlattening;
        final boolean includeMoon;
        final Vector3D p;
        final Vector3D q;
        final Vector3D n;
        final Vector3D sunKm;
        final FixedBody sunBody;
        final ECOM2 ecom2;
        final OneAxisEllipsoid earthBody;
        final List<SampleSpec> samples = new ArrayList<>();

        CaseSpec(final String name, final int nD, final int nB, final double[] coeffs,
                 final double aKm, final double incDeg, final double raanDeg,
                 final double uSunDeg, final double betaDeg, final double sunAu,
                 final double earthFlattening, final boolean includeMoon) {
            this.name = name;
            this.nD = nD;
            this.nB = nB;
            this.coeffs = coeffs;
            this.aKm = aKm;
            this.incRad = FastMath.toRadians(incDeg);
            this.raanRad = FastMath.toRadians(raanDeg);
            this.uSunRad = FastMath.toRadians(uSunDeg);
            this.betaRad = FastMath.toRadians(betaDeg);
            this.sunDistKm = sunAu * AU_KM;
            this.earthFlattening = earthFlattening;
            this.includeMoon = includeMoon;
            this.p = new Vector3D(FastMath.cos(raanRad), FastMath.sin(raanRad), 0.0);
            this.q = new Vector3D(-FastMath.cos(incRad) * FastMath.sin(raanRad),
                                  FastMath.cos(incRad) * FastMath.cos(raanRad),
                                  FastMath.sin(incRad));
            this.n = Vector3D.crossProduct(p, q);
            final Vector3D inPlane = new Vector3D(FastMath.cos(uSunRad), p, FastMath.sin(uSunRad), q);
            this.sunKm = new Vector3D(sunDistKm * FastMath.cos(betaRad), inPlane,
                                      sunDistKm * FastMath.sin(betaRad), n);
            this.sunBody = new FixedBody(toMetres(sunKm));
            this.ecom2 = new ECOM2(nD, nB, 0.0, sunBody, EARTH_RADIUS_M);
            this.earthBody = new OneAxisEllipsoid(EARTH_RADIUS_M, earthFlattening, gcrf);
        }

        Vector3D satPosKm(final double du) {
            final double u = uSunRad + du;
            return new Vector3D(aKm * FastMath.cos(u), p, aKm * FastMath.sin(u), q);
        }

        Vector3D satVelKmS(final double du) {
            final double u = uSunRad + du;
            final double speedKmS = FastMath.sqrt(MU / (aKm * M_PER_KM)) / M_PER_KM;
            return new Vector3D(-speedKmS * FastMath.sin(u), p, speedKmS * FastMath.cos(u), q);
        }

        SolarRadiationPressure makeSrp(final Vector3D moonKm) {
            final SolarRadiationPressure srp = new SolarRadiationPressure(
                sunBody, earthBody, new IsotropicRadiationSingleCoefficient(1.0, 1.0));
            if (includeMoon) {
                srp.addOccultingBody(new FixedBody(toMetres(moonKm)), MOON_RADIUS_M);
            }
            return srp;
        }

        double earthLighting(final double du) {
            final Vector3D satM = toMetres(satPosKm(du));
            final Vector3D velMs = toMetres(satVelKmS(du));
            final SpacecraftState s = new SpacecraftState(
                new AbsolutePVCoordinates(gcrf, T0, new PVCoordinates(satM, velMs)));
            final SolarRadiationPressure srp = new SolarRadiationPressure(
                sunBody, earthBody, new IsotropicRadiationSingleCoefficient(1.0, 1.0));
            return srp.getLightingRatio(s);
        }

        double bisectEarth(double lo, double hi, final double target) {
            double fLo = earthLighting(lo) - target;
            for (int i = 0; i < 200; i++) {
                final double mid = 0.5 * (lo + hi);
                final double fMid = earthLighting(mid) - target;
                if ((fMid > 0.0) == (fLo > 0.0)) {
                    lo = mid;
                    fLo = fMid;
                } else {
                    hi = mid;
                }
            }
            return 0.5 * (lo + hi);
        }

        void addStandardSamples(final double[] sunlitDegs, final boolean includeShadowSearch,
                                final boolean includeGrazingOvershoot) {
            final Vector3D clearMoon = includeMoon ? clearMoonKm(sunKm) : null;
            for (double deg : sunlitDegs) {
                final String lbl = deg == 0.0 ? "noon" : (FastMath.abs(deg) == 180.0 ? "midnight" : "grid_" + (int) deg);
                samples.add(new SampleSpec(lbl, FastMath.toRadians(deg), clearMoon));
            }
            if (includeShadowSearch && earthLighting(FastMath.PI) < 1.0) {
                final double step = FastMath.toRadians(0.01);
                double lit = FastMath.PI / 2.0;
                while (earthLighting(lit + step) == 1.0 && lit < FastMath.PI) {
                    lit += step;
                }
                double dark = lit + step;
                while (earthLighting(dark) > 0.0 && dark < FastMath.PI) {
                    dark += step;
                }
                for (double target : new double[] {0.9, 0.5, 0.1}) {
                    samples.add(new SampleSpec("penumbra_entry_" + target, bisectEarth(lit, dark, target), clearMoon));
                }
                if (includeGrazingOvershoot) {
                    // Search right outside ANGULAR_MARGIN at both the penumbra-sunlit
                    // and penumbra-umbra boundaries for Orekit's unclamped maskingRatio
                    // overshoots (> 1.0 or < 0.0).
                    double lo = lit;
                    double hi = bisectEarth(lit, dark, 0.9999);
                    for (int i = 0; i < 80; i++) {
                        final double mid = 0.5 * (lo + hi);
                        if (earthLighting(mid) == 1.0) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    double bestGt1Du = hi;
                    double bestGt1 = earthLighting(hi);
                    for (double eps = 1.0e-15; eps <= 1.0e-7; eps *= 1.5) {
                        final double r = earthLighting(hi + eps);
                        if (r > bestGt1 && r != 1.0) {
                            bestGt1 = r;
                            bestGt1Du = hi + eps;
                        }
                    }
                    if (bestGt1 > 1.0) {
                        samples.add(new SampleSpec("grazing_overshoot_gt1", bestGt1Du, clearMoon));
                    }

                    double uLo = bisectEarth(lit, dark, 1.0e-4);
                    double uHi = dark;
                    for (int i = 0; i < 80; i++) {
                        final double mid = 0.5 * (uLo + uHi);
                        if (earthLighting(mid) == 0.0) {
                            uHi = mid;
                        } else {
                            uLo = mid;
                        }
                    }
                    double bestLt0Du = uLo;
                    double bestLt0 = earthLighting(uLo);
                    for (double eps = 1.0e-15; eps <= 1.0e-7; eps *= 1.5) {
                        final double r = earthLighting(uLo - eps);
                        if (r < bestLt0 && r != 0.0) {
                            bestLt0 = r;
                            bestLt0Du = uLo - eps;
                        }
                    }
                    if (bestLt0 < 0.0) {
                        samples.add(new SampleSpec("grazing_overshoot_lt0", bestLt0Du, clearMoon));
                    }
                }
                double litExit = 3.0 * FastMath.PI / 2.0;
                while (earthLighting(litExit - step) == 1.0 && litExit > FastMath.PI) {
                    litExit -= step;
                }
                double darkExit = litExit - step;
                while (earthLighting(darkExit) > 0.0 && darkExit > FastMath.PI) {
                    darkExit -= step;
                }
                for (double target : new double[] {0.9, 0.5, 0.1}) {
                    samples.add(new SampleSpec("penumbra_exit_" + target, bisectEarth(litExit, darkExit, target), clearMoon));
                }
                if (earthLighting(FastMath.PI) == 0.0) {
                    samples.add(new SampleSpec("umbra_midnight", FastMath.PI, clearMoon));
                    samples.add(new SampleSpec("umbra_interior", 0.5 * (dark + FastMath.PI), clearMoon));
                }
            }
        }

        void addMoonEclipseSamples() {
            if (!includeMoon) {
                return;
            }
            // Sunlit side of Earth (du = 40 deg, 60 deg, 80 deg): Earth is behind
            // satellite so Earth does not occult the Moon (mutual correction = 0).
            final double duPartial = FastMath.toRadians(40.0);
            final Vector3D satP = satPosKm(duPartial);
            final double rSun = Constants.SUN_RADIUS / M_PER_KM;
            final double rMoon = MOON_RADIUS_M / M_PER_KM;
            final double dSun = sunKm.subtract(satP).getNorm();
            final double aSun = FastMath.asin(rSun / dSun);
            final double dMoonP = 384400.0;
            final double aMoonP = FastMath.asin(rMoon / dMoonP);
            samples.add(new SampleSpec("moon_partial", duPartial,
                moonAtOffsetKm(satP, sunKm, dMoonP, 0.5 * (aSun + aMoonP))));

            // Annular Moon eclipse (Moon at apogee, slightly off-axis so AD gradient is finite).
            final double duAnnular = FastMath.toRadians(60.0);
            final Vector3D satA = satPosKm(duAnnular);
            samples.add(new SampleSpec("moon_annular", duAnnular,
                moonAtOffsetKm(satA, sunKm, 405000.0, 2.0e-4)));

            // Total Moon umbra (Moon at perigee, centered on Sun).
            final double duTotal = FastMath.toRadians(80.0);
            final Vector3D satT = satPosKm(duTotal);
            samples.add(new SampleSpec("moon_umbra", duTotal,
                moonAtOffsetKm(satT, sunKm, 356000.0, 0.0)));
        }
    }

    static String evalSample(final CaseSpec c, final SampleSpec sp) {
        final Vector3D satKm = c.satPosKm(sp.du);
        final Vector3D velKmS = c.satVelKmS(sp.du);
        final Vector3D satM = toMetres(satKm);
        final Vector3D velMs = toMetres(velKmS);
        final Vector3D sunM = toMetres(c.sunKm);

        final SpacecraftState state = new SpacecraftState(
            new AbsolutePVCoordinates(gcrf, T0, new PVCoordinates(satM, velMs)));
        final SolarRadiationPressure srp = c.makeSrp(sp.moonKm);
        final double ratioD = srp.getLightingRatio(state);

        // Sanity check: for Earth+Moon cases, verify Earth and Moon do not trigger
        // Orekit's defective mutual overlap subtraction (at least one body is clear or umbra).
        if (c.includeMoon && ratioD != 0.0) {
            final double earthOnly = c.earthLighting(sp.du);
            final Vector3D satRelMoon = satM.subtract(toMetres(sp.moonKm));
            final Vector3D sunRelMoon = sunM.subtract(toMetres(sp.moonKm));
            final SolarRadiationPressure moonCheck = new SolarRadiationPressure(
                new FixedBody(sunRelMoon), new OneAxisEllipsoid(MOON_RADIUS_M, 0.0, gcrf),
                new IsotropicRadiationSingleCoefficient(1.0, 1.0));
            final SpacecraftState moonState = new SpacecraftState(
                new AbsolutePVCoordinates(gcrf, T0, new PVCoordinates(satRelMoon, velMs)));
            final double moonOnly = moonCheck.getLightingRatio(moonState);
            if (earthOnly != 1.0 && moonOnly != 1.0) {
                throw new IllegalStateException("Simultaneous Earth+Moon partial in " + c.name + "/" + sp.label);
            }
            if (FastMath.abs(ratioD - (earthOnly + moonOnly - 1.0)) > 1.0e-15) {
                throw new IllegalStateException("Mutual eclipse subtraction fired in " + c.name + "/" + sp.label
                    + ": ratio=" + ratioD + " earth=" + earthOnly + " moon=" + moonOnly);
            }
        }

        final int pCount = c.coeffs.length;
        final int nFree = 6 + pCount;
        final GradientField field = GradientField.getField(nFree);
        final FieldAbsoluteDate<Gradient> fDate = new FieldAbsoluteDate<>(field, T0);
        final FieldVector3D<Gradient> fSat = new FieldVector3D<>(
            Gradient.variable(nFree, 0, satM.getX()),
            Gradient.variable(nFree, 1, satM.getY()),
            Gradient.variable(nFree, 2, satM.getZ()));
        final FieldVector3D<Gradient> fVel = new FieldVector3D<>(
            Gradient.variable(nFree, 3, velMs.getX()),
            Gradient.variable(nFree, 4, velMs.getY()),
            Gradient.variable(nFree, 5, velMs.getZ()));
        final FieldSpacecraftState<Gradient> fState = new FieldSpacecraftState<>(
            new FieldAbsolutePVCoordinates<>(gcrf, fDate, new FieldPVCoordinates<>(fSat, fVel)));
        final Gradient[] fParams = new Gradient[pCount];
        for (int k = 0; k < pCount; k++) {
            fParams[k] = Gradient.variable(nFree, 6 + k, c.coeffs[k]);
        }

        final Gradient ratioF = srp.getLightingRatio(fState);

        final Vector3D z = state.getPVCoordinates().getMomentum();
        final Vector3D y = z.crossProduct(sunM).normalize();
        final Vector3D x = y.crossProduct(z).normalize();
        final double orekitDu = FastMath.atan2(satM.dotProduct(y), satM.dotProduct(x));

        final Vector3D accelD;
        final double[] fVal = new double[3];
        final double[][] fWrtPos = new double[3][3];
        final double[][] fWrtVel = new double[3][3];
        final double[][] fWrtParam = new double[pCount][3];

        if (ratioD == 0.0) {
            if (!ratioF.isZero()) {
                throw new IllegalStateException("ratioD == 0 but ratioF != 0 in " + c.name + "/" + sp.label);
            }
            accelD = Vector3D.ZERO;
        } else {
            final Vector3D rawD = c.ecom2.acceleration(state, c.coeffs);
            final FieldVector3D<Gradient> rawF = c.ecom2.acceleration(fState, fParams);
            final FieldVector3D<Gradient> accelF;
            if (ratioD == 1.0 && ratioF.getValue() == 1.0
                    && ratioF.getPartialDerivative(0) == 0.0
                    && ratioF.getPartialDerivative(1) == 0.0
                    && ratioF.getPartialDerivative(2) == 0.0) {
                accelD = rawD;
                accelF = rawF;
            } else {
                accelD = new Vector3D(ratioD, rawD);
                accelF = rawF.scalarMultiply(ratioF);
            }
            final Gradient[] comps = {accelF.getX(), accelF.getY(), accelF.getZ()};
            for (int row = 0; row < 3; row++) {
                fVal[row] = comps[row].getValue();
                for (int col = 0; col < 3; col++) {
                    fWrtPos[row][col] = comps[row].getPartialDerivative(col);
                    fWrtVel[row][col] = comps[row].getPartialDerivative(3 + col);
                }
                for (int k = 0; k < pCount; k++) {
                    fWrtParam[k][row] = comps[row].getPartialDerivative(6 + k);
                }
            }
        }

        final double[] ratioGrad = {
            ratioF.getPartialDerivative(0),
            ratioF.getPartialDerivative(1),
            ratioF.getPartialDerivative(2)
        };

        final StringBuilder sb = new StringBuilder();
        sb.append("        {\"label\": \"").append(sp.label).append("\"")
          .append(", \"delta_u_rad\": ").append(Double.toString(sp.du))
          .append(", \"u_rad\": ").append(Double.toString(c.uSunRad + sp.du))
          .append(",\n         \"sat_km_bits\": ").append(hexVec3(satKm))
          .append(",\n         \"vel_km_s_bits\": ").append(hexVec3(velKmS))
          .append(",\n         \"sun_km_bits\": ").append(hexVec3(c.sunKm));
        if (c.includeMoon) {
            sb.append(",\n         \"moon_km_bits\": ").append(hexVec3(sp.moonKm));
        }
        sb.append(",\n         \"orekit_delta_u_bits\": ").append(hex(orekitDu))
          .append(", \"lighting_ratio_double_bits\": ").append(hex(ratioD))
          .append(", \"lighting_ratio_field_val_bits\": ").append(hex(ratioF.getValue()))
          .append(", \"lighting_ratio_field_grad_bits\": ").append(hexArr(ratioGrad))
          .append(",\n         \"accel_double_bits\": ").append(hexVec3(accelD))
          .append(",\n         \"accel_field_val_bits\": ").append(hexArr(fVal))
          .append(",\n         \"accel_field_wrt_pos_bits\": [")
          .append(hexArr(fWrtPos[0])).append(", ")
          .append(hexArr(fWrtPos[1])).append(", ")
          .append(hexArr(fWrtPos[2])).append("]")
          .append(",\n         \"accel_field_wrt_vel_bits\": [")
          .append(hexArr(fWrtVel[0])).append(", ")
          .append(hexArr(fWrtVel[1])).append(", ")
          .append(hexArr(fWrtVel[2])).append("]")
          .append(",\n         \"accel_field_wrt_param_bits\": [");
        for (int k = 0; k < pCount; k++) {
            if (k > 0) {
                sb.append(", ");
            }
            sb.append(hexArr(fWrtParam[k]));
        }
        sb.append("]}");
        return sb.toString();
    }

    public static void main(final String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        DataContext.getDefault().getDataProvidersManager()
            .addProvider(new DirectoryCrawler(new File(args[0])));
        gcrf = FramesFactory.getGCRF();
        final String provenance = args[1];

        final List<CaseSpec> cases = new ArrayList<>();

        // 1. GPS, beta = 0.5 deg, perihelion, spherical Earth, D4B1 (incl. grazing overshoot)
        final CaseSpec gps0p5 = new CaseSpec(
            "gps_beta_0p5_perihelion_sphere_d4b1", 2, 1, COEFFS_D4B1,
            26560.0, 55.0, 30.0, 40.0, 0.5, 0.98329, 0.0, false);
        gps0p5.addStandardSamples(new double[] {0, 25, 90, 150, -135, -45}, true, true);
        cases.add(gps0p5);

        // 2. GPS, beta = 0.02 deg (near 0), noon/midnight, spherical Earth, D2B1
        final CaseSpec gps0p02 = new CaseSpec(
            "gps_beta_0p02_sphere_d2b1", 1, 1, COEFFS_D2B1,
            26560.0, 55.0, 110.0, 15.0, 0.02, 1.0, 0.0, false);
        gps0p02.addStandardSamples(new double[] {0, 1, -1, 90, -90}, true, false);
        cases.add(gps0p02);

        // 3. GPS, beta = -9 deg, WGS84 Earth, D4B1
        final CaseSpec gpsM9 = new CaseSpec(
            "gps_beta_minus_9_wgs84_d4b1", 2, 1, COEFFS_D4B1,
            26560.0, 55.0, 200.0, -120.0, -9.0, 1.0, WGS84_FLATTENING, false);
        gpsM9.addStandardSamples(new double[] {0, 60, -120}, true, false);
        cases.add(gpsM9);

        // 4. Galileo, beta = 25 deg, aphelion, spherical Earth, D6B3
        final CaseSpec gal25 = new CaseSpec(
            "galileo_beta_25_aphelion_sphere_d6b3", 3, 2, COEFFS_D6B3,
            29600.0, 56.0, 80.0, 10.0, 25.0, 1.01671, 0.0, false);
        gal25.addStandardSamples(new double[] {0, 60, 120, 180, -120, -45}, false, false);
        cases.add(gal25);

        // 5. Galileo, beta = 1 deg, WGS84 Earth + Moon, D4B1
        final CaseSpec gal1 = new CaseSpec(
            "galileo_beta_1_wgs84_moon_d4b1", 2, 1, COEFFS_D4B1,
            29600.0, 56.0, 250.0, -60.0, 1.0, 1.0, WGS84_FLATTENING, true);
        gal1.addStandardSamples(new double[] {0, 120, -60}, true, false);
        gal1.addMoonEclipseSamples();
        cases.add(gal1);

        // 6. GLONASS, beta = 60 deg, spherical Earth, D2B1
        final CaseSpec glo60 = new CaseSpec(
            "glonass_beta_60_sphere_d2b1", 1, 1, COEFFS_D2B1,
            25510.0, 64.8, 300.0, 150.0, 60.0, 1.0, 0.0, false);
        glo60.addStandardSamples(new double[] {0, 90, 180, -90}, false, false);
        cases.add(glo60);

        // 7. GLONASS, beta = 89.5 deg (near 90), spherical Earth, D4B1
        final CaseSpec glo89p5 = new CaseSpec(
            "glonass_beta_89p5_sphere_d4b1", 2, 1, COEFFS_D4B1,
            25510.0, 64.8, 120.0, -30.0, 89.5, 1.0, 0.0, false);
        glo89p5.addStandardSamples(new double[] {0, 60, 180, -120}, false, false);
        cases.add(glo89p5);

        // 8. BeiDou MEO, beta = -15 deg, spherical Earth + Moon, D6B3
        final CaseSpec bdsMeo = new CaseSpec(
            "beidou_meo_beta_minus_15_sphere_moon_d6b3", 3, 2, COEFFS_D6B3,
            27906.0, 55.0, 45.0, 110.0, -15.0, 1.0, 0.0, true);
        bdsMeo.addStandardSamples(new double[] {0, 135, 180, -45}, false, false);
        bdsMeo.addMoonEclipseSamples();
        cases.add(bdsMeo);

        // 9. BeiDou IGSO, beta = 4 deg, WGS84 Earth + Moon, D2B1
        final CaseSpec bdsIgso4 = new CaseSpec(
            "beidou_igso_beta_4_wgs84_moon_d2b1", 1, 1, COEFFS_D2B1,
            42164.0, 55.0, 95.0, 200.0, 4.0, 1.0, WGS84_FLATTENING, true);
        bdsIgso4.addStandardSamples(new double[] {0, -90}, true, false);
        bdsIgso4.addMoonEclipseSamples();
        cases.add(bdsIgso4);

        // 10. BeiDou IGSO, beta = -89 deg (near -90), spherical Earth, D0B0
        final CaseSpec bdsIgsoM89 = new CaseSpec(
            "beidou_igso_beta_minus_89_sphere_d0b0", 0, 0, COEFFS_D0B0,
            42164.0, 55.0, 15.0, 80.0, -89.0, 1.0, 0.0, false);
        bdsIgsoM89.addStandardSamples(new double[] {0, 90, 180, -60}, false, false);
        cases.add(bdsIgsoM89);

        // 11. BeiDou GEO equatorial (i = 0), beta = 3 deg, WGS84 Earth, D4B1
        final CaseSpec bdsGeoEq = new CaseSpec(
            "beidou_geo_equatorial_beta_3_wgs84_d4b1", 2, 1, COEFFS_D4B1,
            42164.0, 0.0, 0.0, 70.0, 3.0, 1.0, WGS84_FLATTENING, false);
        bdsGeoEq.addStandardSamples(new double[] {0, 90, -90}, true, false);
        cases.add(bdsGeoEq);

        // 12. BeiDou GEO inclined (i = 0.5 deg), beta = 12 deg, spherical Earth + Moon, D4B1
        final CaseSpec bdsGeoInc = new CaseSpec(
            "beidou_geo_beta_12_sphere_moon_d4b1", 2, 1, COEFFS_D4B1,
            42164.0, 0.5, 40.0, -45.0, 12.0, 1.0, 0.0, true);
        bdsGeoInc.addStandardSamples(new double[] {0, 180, -100}, false, false);
        bdsGeoInc.addMoonEclipseSamples();
        cases.add(bdsGeoInc);

        final StringBuilder out = new StringBuilder();
        out.append("{\n  \"_source\": ").append(provenance).append(",\n");
        out.append("  \"frame\": \"GCRF\"")
           .append(", \"earth_radius_m_bits\": ").append(hex(EARTH_RADIUS_M))
           .append(", \"wgs84_flattening_bits\": ").append(hex(WGS84_FLATTENING))
           .append(", \"moon_radius_m_bits\": ").append(hex(MOON_RADIUS_M))
           .append(", \"sun_radius_m_bits\": ").append(hex(Constants.SUN_RADIUS))
           .append(",\n  \"cases\": [\n");
        for (int i = 0; i < cases.size(); i++) {
            final CaseSpec c = cases.get(i);
            out.append("    {\"name\": \"").append(c.name).append("\"")
               .append(", \"n_d\": ").append(c.nD)
               .append(", \"n_b\": ").append(c.nB)
               .append(", \"earth_flattening\": ").append(Double.toString(c.earthFlattening))
               .append(", \"include_moon\": ").append(c.includeMoon)
               .append(", \"beta_deg\": ").append(Double.toString(FastMath.toDegrees(c.betaRad)))
               .append(", \"inclination_deg\": ").append(Double.toString(FastMath.toDegrees(c.incRad)))
               .append(",\n      \"coefficients_bits\": ").append(hexArr(c.coeffs))
               .append(",\n      \"samples\": [\n");
            for (int j = 0; j < c.samples.size(); j++) {
                out.append(evalSample(c, c.samples.get(j)))
                   .append(j + 1 < c.samples.size() ? ",\n" : "\n");
            }
            out.append("      ]}").append(i + 1 < cases.size() ? ",\n" : "\n");
        }
        out.append("  ]\n}\n");
        System.out.print(out);
    }
}
