import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;

import org.hipparchus.analysis.differentiation.Gradient;
import org.hipparchus.analysis.differentiation.GradientField;
import org.hipparchus.geometry.euclidean.threed.FieldVector3D;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.forces.gravity.J2OnlyPerturbation;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.propagation.FieldSpacecraftState;
import org.orekit.propagation.SpacecraftState;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.FieldAbsoluteDate;
import org.orekit.utils.AbsolutePVCoordinates;
import org.orekit.utils.FieldAbsolutePVCoordinates;

public final class J2Oracle {
    private static final int VARIABLES = 3;

    private record Sample(String name, double x, double y, double z) {}

    private static void append(StringBuilder out, double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException("non-finite oracle entry");
        }
        out.append(' ').append(HexFormat.of().toHexDigits(Double.doubleToRawLongBits(value)));
    }

    public static void main(String[] args) throws IOException {
        // Use km and seconds consistently for both position and gravity constants.
        final Frame gcrf = FramesFactory.getGCRF();
        final J2OnlyPerturbation j2 = new J2OnlyPerturbation(398600.4418, 6378.137, 1.08262668e-3, gcrf);
        final AbsoluteDate date = AbsoluteDate.J2000_EPOCH;
        final GradientField field = GradientField.getField(VARIABLES);
        final FieldAbsoluteDate<Gradient> fieldDate = new FieldAbsoluteDate<>(field, date);
        final Sample[] samples = {
            new Sample("leo", 7000, -1210, 1300),
            new Sample("meo", -13832.75, 21021.5, 8651.25),
            new Sample("geo", 25298.125, -33733.875, 41.5),
            new Sample("polar", 0, 0, -7112.5),
            new Sample("equatorial", -6600.75, 2300.125, -0.0),
            new Sample("near_zero_z", 7134.25, 1100.5, -0x1.42p-511),
            new Sample("high_radius", -9.15e11, 1.3e12, -5.54e11),
            new Sample("low_radius", -1.125, 0.625, 0.25),
            new Sample("large_exponent", 0x1.2p169, -0x1.4p167, 0x1.3p168),
            new Sample("small_exponent", 0x1.2p-163, -0x1.4p-165, 0x1.3p-164),
        };

        final StringBuilder out = new StringBuilder(
            "# Orekit 13.1.10 / Hipparchus 4.0.3, GCRF\n" +
            "# Case, position (km), acceleration (km/s^2), position partials (1/s^2, row-major)\n"
        );
        for (Sample sample : samples) {
            final Vector3D position = new Vector3D(sample.x(), sample.y(), sample.z());
            final SpacecraftState state = new SpacecraftState(
                new AbsolutePVCoordinates(gcrf, date, position, Vector3D.ZERO)
            );
            final FieldVector3D<Gradient> fieldPosition = new FieldVector3D<>(
                Gradient.variable(VARIABLES, 0, sample.x()),
                Gradient.variable(VARIABLES, 1, sample.y()),
                Gradient.variable(VARIABLES, 2, sample.z())
            );
            final FieldSpacecraftState<Gradient> fieldState = new FieldSpacecraftState<>(
                new FieldAbsolutePVCoordinates<>(gcrf, fieldDate, fieldPosition, FieldVector3D.getZero(field))
            );
            out.append(sample.name());
            for (double coordinate : position.toArray()) {
                append(out, coordinate);
            }
            for (double component : j2.acceleration(state, new double[0]).toArray()) {
                append(out, component);
            }
            final FieldVector3D<Gradient> acceleration = j2.acceleration(fieldState, j2.getParameters(field, fieldDate));
            for (Gradient component : acceleration.toArray()) {
                for (double derivative : component.getGradient()) {
                    append(out, derivative);
                }
            }
            out.append('\n');
        }
        Files.writeString(Path.of(args[0]), out);
    }
}
