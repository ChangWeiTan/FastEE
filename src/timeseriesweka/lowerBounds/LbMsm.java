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

import timeseriesweka.fastWWS.SequenceStatsCache;
import weka.core.Instance;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Lower bound for MSM
 */
public class LbMsm {
    public static double distance(final Instance q, final Instance c, final double cc, final double qMax, final double qMin) {
        final int len = q.numAttributes() - 1;

        double d = Math.abs(q.value(0) - c.value(0));

        for (int i = 1; i < len; i++) {
            final double curr = c.value(i);
            final double prev = c.value(i - 1);
            if (prev >= curr && curr > qMax) {
                d += Math.min(Math.abs(curr - qMax), cc);
            } else if (prev <= curr && curr < qMin) {
                d += Math.min(Math.abs(curr - qMin), cc);
            }
        }

        return d;
    }

    public static double distance(final Instance q, final Instance c, final double cc, final double qMax, final double qMin, final double cutOffValue) {
        final int len = q.numAttributes() - 1;

        double d = Math.abs(q.value(0) - c.value(0));

        for (int i = 1; i < len; i++) {
            final double curr = c.value(i);
            final double prev = c.value(i - 1);
            if (prev >= curr && curr > qMax) {
                d += Math.min(Math.abs(curr - qMax), cc);
                if (d >= cutOffValue)
                    return Double.MAX_VALUE;
            } else if (prev <= curr && curr < qMin) {
                d += Math.min(Math.abs(curr - qMin), cc);
                if (d >= cutOffValue)
                    return Double.MAX_VALUE;
            }
        }

        return d;
    }

    public static double distance(final Instance query, final Instance reference,
                                  final SequenceStatsCache queryCache, final SequenceStatsCache referenceCache,
                                  final int indexQuery, final int indexReference) {
        final double diffFirsts = Math.abs(query.value(0) - reference.value(0));
        final double diffLasts = Math.abs(query.value(query.numAttributes() - 2) - reference.value(reference.numAttributes() - 2));
        double minDist = diffFirsts + diffLasts;

        if (!queryCache.isMinFirst(indexQuery) && !referenceCache.isMinFirst(indexReference) &&
                !queryCache.isMinLast(indexQuery) && !referenceCache.isMinLast(indexReference)) {
            minDist += Math.abs(queryCache.getMin(indexQuery) - referenceCache.getMin(indexReference));
        }
        if (!queryCache.isMaxFirst(indexQuery) && !referenceCache.isMaxFirst(indexReference) &&
                !queryCache.isMaxLast(indexQuery) && !referenceCache.isMaxLast(indexReference)) {
            minDist += Math.abs(queryCache.getMax(indexQuery) - referenceCache.getMax(indexReference));
        }

        return minDist;
    }
}
