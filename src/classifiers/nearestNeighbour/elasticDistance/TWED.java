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

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 * <p>
 * Original code from http://www.timeseriesclassification.com/code.php
 * https://bitbucket.org/TonyBagnall/time-series-classification/src/f47c300b937af1d06aac86bdbe228cb9a9d5e00d?at=default
 * <p>
 * TWED distance
 */
public class TWED extends ElasticDistances {
    private final static double[][] D = new double[MAX_SEQ_LENGTH + 1][MAX_SEQ_LENGTH + 1];
    private final static double[] Di1 = new double[MAX_SEQ_LENGTH + 1];
    private final static double[] Dj1 = new double[MAX_SEQ_LENGTH + 1];


    public double distance(final double[] first, final double[] second, final double nu, final double lambda) {
        final int m = first.length;
        final int n = second.length;
        final double[][] D = new double[m + 1][m + 1];
        final double[] Di1 = new double[m + 1];
        final double[] Dj1 = new double[n + 1];

        double diff, dist;
        double dmin, htrans;
        int i, j;

        // local costs initializations
        for (j = 1; j <= n; j++) {
            if (j > 1) {
                Dj1[j] = second[j - 2] - second[j - 1];
                Dj1[j] = Dj1[j] * Dj1[j];
            } else {
                Dj1[j] = second[j - 1] * second[j - 1];
            }
        }

        for (i = 1; i <= m; i++) {
            if (i > 1) {
                Di1[i] = first[i - 2] - first[i - 1];
                Di1[i] = Di1[i] * Di1[i];
            } else {
                Di1[i] = first[i - 1] * first[i - 1];
            }

            for (j = 1; j <= n; j++) {
                D[i][j] = first[i - 1] - second[j - 1];
                D[i][j] = D[i][j] * D[i][j];
                if (i > 1 && j > 1) {
                    diff = first[i - 2] - second[j - 2];
                    D[i][j] += diff * diff;
                }
            }
        }

        // border of the cost matrix initialization
        D[0][0] = 0;
        for (i = 1; i <= m; i++) {
            D[i][0] = D[i - 1][0] + Di1[i];
        }
        for (j = 1; j <= n; j++) {
            D[0][j] = D[0][j - 1] + Dj1[j];
        }

        for (i = 1; i <= m; i++) {
            for (j = 1; j <= n; j++) {
                htrans = Math.abs(i - j);
                if (j > 1 && i > 1) {
                    htrans *= 2;
                }
                dmin = D[i - 1][j - 1] + nu * htrans + D[i][j];

                dist = Di1[i] + D[i - 1][j] + lambda + nu;
                if (dmin > dist) {
                    dmin = dist;
                }
                dist = Dj1[j] + D[i][j - 1] + lambda + nu;
                if (dmin > dist) {
                    dmin = dist;
                }

                D[i][j] = dmin;
            }
        }

        dist = D[m][n];
        return dist;
    }
}
