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
 * Lower bound for TWED
 */
public class LbTwed {
    public double distance(final Sequence q, final Sequence c, final double qMax, final double qMin,
                                  final double nu, final double lambda) {
        final int length = q.length();
        final double q0 = q.value(0);
        final double c0 = c.value(0);
        double diff = q0 - c0;
        double res = Math.min(diff * diff,
                Math.min(q0 * q0 + nu + lambda,
                        c0 * c0 + nu + lambda));

        for (int i = 1; i < length; i++) {
            final double curr = c.value(i);
            final double prev = c.value(i-1);
            final double max = Math.max(qMax, prev);
            final double min = Math.min(qMin, prev);
            if (curr < min) {
                diff = min - curr;
                res += Math.min(nu, diff * diff);
            } else if (max < curr) {
                diff = max - curr;
                res += Math.min(nu, diff * diff);
            }
        }

        return res;
    }

    public double distance(final Sequence q, final Sequence c, final double qMax, final double qMin,
                                  final double nu, final double lambda, final double cutOffValue) {
        final int length = q.length();
        final double q0 = q.value(0);
        final double c0 = c.value(0);
        double diff = q0 - c0;
        double res = Math.min(diff * diff,
                Math.min(q0 * q0 + nu + lambda,
                        c0 * c0 + nu + lambda));
        if (res >= cutOffValue)
            return res;

        for (int i = 1; i < length; i++) {
            final double curr = c.value(i);
            final double prev = c.value(i-1);
            final double max = Math.max(qMax, prev);
            final double min = Math.min(qMin, prev);
            if (curr < min) {
                diff = min - curr;
                res += Math.min(nu, diff * diff);
                if (res >= cutOffValue)
                    return res;
            } else if (max < curr) {
                diff = max - curr;
                res += Math.min(nu, diff * diff);
                if (res >= cutOffValue)
                    return res;
            }
        }

        return res;
    }

}
