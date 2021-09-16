package filters;

import datasets.Sequence;
import datasets.Sequences;

public class DerivativeFilter {
    public static Sequences getFirstDerivative(final Sequences data) {
        final Sequences output = new Sequences(data.size(), data.length());

        for (int i = 0; i < data.size(); i++) { // for each data
            final Sequence sequence = data.get(i);
            final double[] rawData = sequence.data[0];
            final double[] derivative = getFirstDerivative(rawData);
            final Sequence toAdd = new Sequence(derivative, sequence.classificationLabel);
            output.add(toAdd);
        }
        return output;
    }

    /**
     * Convert a single time series to its derivative form
     *
     * @param input input time series
     * @return derivative of input time series
     */
    public static Sequence getFirstDerivative(final Sequence input) {
        final double[] derivative = new double[input.length()];

        for (int i = 1; i < input.length() - 1; i++) {
            derivative[i] = ((input.value(i) - input.value(i - 1)) + ((input.value(i + 1) - input.value(i - 1)) / 2)) / 2;
        }

        derivative[0] = derivative[1];
        derivative[derivative.length - 1] = derivative[derivative.length - 2];

        return new Sequence(derivative, input.classificationLabel);
    }

    /**
     * Convert a single time series to its derivative form
     *
     * @param input input time series
     * @return derivative of input time series
     */
    public static double[] getFirstDerivative(final double[] input) {
        final double[] derivative = new double[input.length];

        for (int i = 1; i < input.length - 1; i++) {
            derivative[i] = ((input[i] - input[i - 1]) + ((input[i + 1] - input[i - 1]) / 2)) / 2;
        }

        derivative[0] = derivative[1];
        derivative[derivative.length - 1] = derivative[derivative.length - 2];

        return derivative;
    }
}
