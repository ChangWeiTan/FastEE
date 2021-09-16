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
 *
 * Lower bound for WDTW
 */
public class LbWdtw {
    public double distance(final Sequence a, final double weight, final double max, final double min) {
        double res = 0;

        for (int i = 0; i < a.length(); i++) {
            final double c = a.value(i);
            if (c < min) {
                final double diff = min - c;
                res += diff * diff;
            } else if (max < c) {
                final double diff = max - c;
                res += diff * diff;
            }
        }

        return weight * res;
    }

    public double distance(final Sequence a, final double weight, final double max, final double min, final double cutOffValue) {
        double res = 0;
        double cutoff = cutOffValue/weight;

        for (int i = 0; i < a.length(); i++) {
            final double c = a.value(i);
            if (c < min) {
                final double diff = min - c;
                res += diff * diff;
                if (res >= cutoff)
                    return Double.MAX_VALUE;
            } else if (max < c) {
                final double diff = max - c;
                res += diff * diff;
                if (res >= cutoff)
                    return Double.MAX_VALUE;
            }
        }
        return weight * res;
    }
}
