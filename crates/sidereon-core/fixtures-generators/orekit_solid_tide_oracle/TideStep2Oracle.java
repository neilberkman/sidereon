// Oracle for the IERS Conventions (2010) Chapter 6 Step 2 corrections: the
// frequency-dependent parts of the degree-2 Love numbers k20, k21 and k22
// (Tables 6.5a, 6.5b and 6.5c, Equations 6.8a and 6.8b). Orekit's
// IERSConventions.IERS_2010.getTideFrequencyDependenceFunction reads its own
// electronic copies of the three tables and returns
// [dC20, dC21, dS21, dC22, dS22] for a date.
//
// For each epoch this prints the fundamental arguments Orekit formed (gamma =
// GMST + pi and the Delaunay arguments l, l', F, D, Omega, in radians) and the
// five corrections, as the elements of a JSON array; generate.sh wraps them.
// Every double is printed with Double.toString, which gives the shortest
// decimal that reads back to the same binary64 value.
//
// Epochs are J2000.0 TT plus a whole number of days plus a day fraction that
// is a multiple of 1/1024, so the TT instant is exact in both programs. The
// scale Orekit uses as UT1 for GMST is TAI, so no Earth-orientation data are
// needed: UT1 is TT - 32.184 s at every epoch, and the Rust test forms its time
// scales the same way.

import java.util.Locale;

import org.orekit.data.BodiesElements;
import org.orekit.data.DataContext;
import org.orekit.data.FundamentalNutationArguments;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScales;
import org.orekit.time.TimeVectorFunction;
import org.orekit.utils.IERSConventions;

public class TideStep2Oracle {

    public static void main(String[] args) {
        TimeScales scales = DataContext.getDefault().getTimeScales();
        TimeScale tai = scales.getTAI();
        TimeScale tt = scales.getTT();
        IERSConventions conventions = IERSConventions.IERS_2010;
        TimeVectorFunction step2 = conventions.getTideFrequencyDependenceFunction(tai, scales);
        FundamentalNutationArguments arguments = conventions.getNutationArguments(tai, scales);
        AbsoluteDate j2000 = new AbsoluteDate(2000, 1, 1, 12, 0, 0.0, tt);

        StringBuilder out = new StringBuilder();
        int count = 64;
        for (int k = 0; k < count; k++) {
            // Whole days from J2000.0 TT, from 1975 to 2050, and a day fraction
            // of (k * 389) mod 1024 parts in 1024.
            long days = -9131L + Math.round(k * (27394.0 / (count - 1)));
            int parts = (k * 389) % 1024;
            double fraction = parts / 1024.0;
            AbsoluteDate date = j2000.shiftedBy(days * 86400L).shiftedBy(fraction * 86400.0);
            BodiesElements e = arguments.evaluateAll(date);
            double[] d = step2.value(date);
            if (k > 0) {
                out.append(",\n");
            }
            out.append(String.format(Locale.ROOT,
                "    {\"jd_whole\": %s, \"tt_fraction\": %s, \"gamma\": %s, \"l\": %s, \"l_prime\": %s,"
                    + " \"f\": %s, \"d\": %s, \"omega\": %s,"
                    + " \"c20\": %s, \"c21\": %s, \"s21\": %s, \"c22\": %s, \"s22\": %s}",
                Double.toString(2451545.0 + days), Double.toString(fraction),
                Double.toString(e.getGamma()), Double.toString(e.getL()),
                Double.toString(e.getLPrime()), Double.toString(e.getF()),
                Double.toString(e.getD()), Double.toString(e.getOmega()),
                Double.toString(d[0]), Double.toString(d[1]), Double.toString(d[2]),
                Double.toString(d[3]), Double.toString(d[4])));
        }
        System.out.println(out);
    }
}
