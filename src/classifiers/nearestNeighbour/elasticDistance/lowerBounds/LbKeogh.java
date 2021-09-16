package classifiers.nearestNeighbour.elasticDistance.lowerBounds;

import datasets.Sequence;

public class LbKeogh {
    public void fillUL(final double[] sequence, final int window, final double[] U, final double[] L) {
        final int length = sequence.length;
        double min, max;

        for (int i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            final int startR = Math.max(0, i - window);
            final int stopR = Math.min(length - 1, i + window);
            for (int j = startR; j <= stopR; j++) {
                final double value = sequence[j];
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            L[i] = min;
            U[i] = max;
        }
    }

    public double distance(final Sequence a, final double[] U, final double[] L, final double cutOffValue) {
        final int length = Math.min(U.length, a.length());
        double res = 0;

        for (int i = 0; i < length; i++) {
            final double c = a.value(i);
            if (c < L[i]) {
                final double diff = L[i] - c;
                res += diff * diff;
                if (res >= cutOffValue)
                    return Double.MAX_VALUE;
            } else if (U[i] < c) {
                final double diff = U[i] - c;
                res += diff * diff;
                if (res >= cutOffValue)
                    return Double.MAX_VALUE;
            }
        }

        return res;
    }
}
