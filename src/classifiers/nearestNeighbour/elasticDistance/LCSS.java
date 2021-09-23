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
package classifiers.nearestNeighbour.elasticDistance;

import results.WarpingPathResults;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 * <p>
 * Original code from http://www.timeseriesclassification.com/code.php
 * https://bitbucket.org/TonyBagnall/time-series-classification/src/f47c300b937af1d06aac86bdbe228cb9a9d5e00d?at=default
 * <p>
 * LCSS distance
 */
public class LCSS extends ElasticDistances {

    public double distance(final double[] first, final double[] second, final double epsilon, final int delta) {
        final int m = first.length;
        final int n = second.length;
        int[][] lcss = new int[m + 1][n + 1];
        int i, j;

        for (i = 0; i < m; i++) {
            for (j = i - delta; j <= i + delta; j++) {
                if (j < 0) {
                    j = -1;
                } else if (j >= n) {
                    j = i + delta;
                } else if (Math.abs(first[i]-second[j]) < epsilon) {
                    lcss[i + 1][j + 1] = lcss[i][j] + 1;
                } else if (delta == 0) {
                    lcss[i + 1][j + 1] = lcss[i][j];
                } else
                    lcss[i + 1][j + 1] = Math.max(lcss[i][j + 1], lcss[i + 1][j]);
            }
        }

        int max = -1;
        for (i = 1; i < m + 1; i++) {
            if (lcss[m][i] > max) {
                max = lcss[m][i];
            }
        }
        return 1.0 - 1.0 * lcss[m][n] / m;
    }

    public WarpingPathResults distanceExt(final double[] first, final double[] second, final double epsilon, final int delta) {
        final int m = first.length;
        final int n = second.length;
        int i, j, absIJ;
        final int[][] lcss = new int[m + 1][n + 1];
        final int[][] minDelta = new int[m + 1][n + 1];

        for (i = 0; i < m; i++) {
            for (j = i - delta; j <= i + delta; j++) {
                if (j < 0) {
                    j = -1;
                } else if (j >= n) {
                    j = i + delta;
                } else if (Math.abs(first[i]-second[j]) < epsilon) {
                    absIJ = Math.abs(i - j);
                    lcss[i + 1][j + 1] = lcss[i][j] + 1;
                    minDelta[i + 1][j + 1] = Math.max(absIJ, minDelta[i][j]);
                } else if (delta == 0) {
                    lcss[i + 1][j + 1] = lcss[i][j];
                    minDelta[i + 1][j + 1] = 0;
                } else if (lcss[i][j + 1] > lcss[i + 1][j]) {
                    lcss[i + 1][j + 1] = lcss[i][j + 1];
                    minDelta[i + 1][j + 1] = minDelta[i][j + 1];
                } else {
                    lcss[i + 1][j + 1] = lcss[i + 1][j];
                    minDelta[i + 1][j + 1] = minDelta[i + 1][j];
                }
            }
        }

        int max = -1, maxR = -1;
        for (i = 1; i < m + 1; i++) {
            if (lcss[m][i] > max) {
                max = lcss[m][i];
                maxR = minDelta[m][i];
            }
        }
        return new WarpingPathResults(1.0 - 1.0 * lcss[m][n] / m, maxR);
    }
}
