/* Copyright (C) 2018 Chang Wei Tan, Francois Petitjean, Geoff Webb
 This file is part of FastEE.
 FastEE is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, version 3 of the License.
 LbEnhanced is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with LbEnhanced.  If not, see <http://www.gnu.org/licenses/>. */
package classifiers.nearestNeighbour.elasticDistance.lowerBounds;


import datasets.Sequence;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 * <p>
 * Lower bound for ERP (Keogh version)
 * See paper http://www.vldb.org/conf/2004/RS21P2.PDF
 */
public class LbErp {
    public double distance(final Sequence a, final Sequence b, final double g) {
        final int m = a.length();
        final int n = b.length();

        if (m == n) {
            double[] s = sum2(a, b, g);
            return Math.abs(s[0] - s[1]);
        } else {
            return Math.abs(sum(a, g) - sum(b, g));
        }
    }

    private double sum(final Sequence a, final double g) {
        double s = 0;
        for (int i = 0; i < a.length(); i++) {
            s += Math.abs(a.value(i) - g);
        }

        return s;
    }

    private double[] sum2(final Sequence a, final Sequence b, final double g) {
        double[] s = new double[2];
        for (int i = 0; i < a.length(); i++) {
            s[0] += Math.abs(a.value(i) - g);
            s[1] += Math.abs(b.value(i) - g);
        }
        return s;
    }

    public void fillUL(final double[] a, final double g, final double bandSize, final double[] U, final double[] L) {
        final int length = a.length;
        final int r = (int) Math.ceil(length * bandSize);
        double min, max;

        for (int i = 0; i < length; i++) {
            min = g;
            max = g;
            final int startR = Math.max(0, i - r);
            final int stopR = Math.min(length - 1, i + r);
            for (int j = startR; j <= stopR; j++) {
                final double value = a[j];
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            U[i] = max;
            L[i] = min;
        }
    }

    public double distance(final Sequence a, final double[] U, final double[] L) {
        return distance(a, U, L, Double.POSITIVE_INFINITY);
    }

    public double distance(final Sequence a, final double[] U, final double[] L, final double cutOffValue) {
        return new LbKeogh().distance(a, U, L, cutOffValue);
    }
}
