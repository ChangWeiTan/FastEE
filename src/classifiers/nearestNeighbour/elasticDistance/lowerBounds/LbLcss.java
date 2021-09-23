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
import weka.core.Instance;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 * <p>
 * Lower bound for LCSS
 * See paper https://www.cs.ucr.edu/~eamonn/SIGKDD-03-Indexing.pdf
 */
public class LbLcss {
    public void fillUL(final Sequence sequence, final double epsilon, final int delta, final double[] U, final double[] L) {
        final int length = sequence.length();
        double min, max;

        for (int i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            final int startR = Math.max(i - delta, 0);
            final int stopR = Math.min(i + delta + 1, length-1);
            for (int j = startR; j <= stopR; j++) {
                final double value = sequence.value(j);
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            L[i] = min - epsilon;
            U[i] = max + epsilon;
        }
    }

    public double distance(final Sequence a, final double[] U, final double[] L) {
        final int length = Math.min(U.length, a.length());

        double lcs = 0;

        for (int i = 0; i < length; i++) {
            if (a.value(i) <= U[i] && a.value(i) >= L[i]) {
                lcs++;
            }
        }

        return 1 - lcs / length;
    }

    public double distance(Sequence a, double[] U, double[] L, double cutOffValue) {
        final int length = Math.min(U.length, a.length());
        final double ub = (1.0 - cutOffValue) * length;

        double lcs = 0;

        for (int i = 0; i < length; i++) {
            if (a.value(i) <= U[i] && a.value(i) >= L[i]) {
                lcs++;
                if (lcs <= ub) return 1;
            }
        }

        return 1 - lcs / length;
    }
}
