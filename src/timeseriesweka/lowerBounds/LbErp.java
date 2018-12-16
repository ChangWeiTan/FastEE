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
package timeseriesweka.lowerBounds;

import weka.core.Instance;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Lower bound for ERP (Keogh version)
 * See paper http://www.vldb.org/conf/2004/RS21P2.PDF
 */
public class LbErp {
    private static double[] s = new double[2];

    public static double distance(final Instance a, final Instance b, final double g) {
        final int m = a.numAttributes() - 1;
        final int n = b.numAttributes() - 1;

        if (m == n) {
            sum2(a, b, g);
            return Math.abs(s[0] - s[1]);
        } else {
            return Math.abs(sum(a, g) - sum(b, g));
        }
    }

    private static double sum(final Instance a, final double g) {
        double s = 0;
        for (int i = 0; i < a.numAttributes() - 1; i++) {
            s += Math.abs(a.value(i) - g);
        }

        return s;
    }

    private static void sum2(final Instance a, final Instance b, final double g) {
        s = new double[2];
        for (int i = 0; i < a.numAttributes() - 1; i++) {
            s[0] += Math.abs(a.value(i) - g);
            s[1] += Math.abs(b.value(i) - g);
        }
    }

    public static void fillUL(final Instance a, final double g, final double bandSize, final double[] U, final double[] L) {
        final int length = a.numAttributes() - 1;
        final int r = (int) Math.ceil(length * bandSize);
        double min, max;

        for (int i = 0; i < length; i++) {
            min = g;
            max = g;
            final int startR = Math.max(0, i - r);
            final int stopR = Math.min(length - 1, i + r);
            for (int j = startR; j <= stopR; j++) {
                final double value = a.value(j);
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            U[i] = max;
            L[i] = min;
        }
    }

    public static double distance(final Instance a, final double[] U, final double[] L) {
        return LbKeogh.distance(a, U, L);
    }

    public static double distance(final Instance a, final double[] U, final double[] L, final double cutOffValue) {
        return LbKeogh.distance(a, U, L, cutOffValue);
    }
}
