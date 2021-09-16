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
 * ERP distance
 */
public class ERP extends ElasticDistances {
    public double distance(double[] first, double[] second, final double g, final double bandSize) {
        final int m = first.length;
        final int n = second.length;
        final int minLen = Math.min(m, n);
        final int band = getWindowSize(minLen, bandSize);
        double[] prev = new double[minLen];
        double[] curr = new double[minLen];

        double diff, d1, d2, d12, cost;
        int i, j, left, right, absIJ;

        double[] tmp;
        if (n < m) {
            tmp = first;
            first = second;
            second = tmp;
        }

        for (i = 0; i < m; i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            tmp = prev;
            prev = curr;
            curr = tmp;

            left = i - (band + 1);
            if (left < 0) {
                left = 0;
            }
            right = i + (band + 1);
            if (right > (m - 1)) {
                right = (m - 1);
            }

            for (j = left; j <= right; j++) {
                absIJ = Math.abs(i - j);
                if (absIJ <= band) {
                    diff = first[i] - g;
                    d1 = (diff * diff);

                    diff = g - second[j];
                    d2 = (diff * diff);

                    diff = first[i] - second[j];
                    d12 = (diff * diff);

                    if ((i + j) != 0) {
                        if ((i == 0) || ((j != 0) &&
                                (((prev[j - 1] + d12) >= (curr[j - 1] + d2)) &&
                                        ((curr[j - 1] + d2) <= (prev[j] + d1))))) {
                            cost = curr[j - 1] + d2;
                        } else if (j == 0 || prev[j - 1] + d12 >= prev[j] + d1 && prev[j] + d1 <= curr[j - 1] + d2) {
                            cost = prev[j] + d1;
                        } else {
                            cost = prev[j - 1] + d12;
                        }
                    } else {
                        cost = 0;
                    }

                    curr[j] = cost;
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
        }

        return (curr[m - 1]);
    }

    public WarpingPathResults distanceExt(double[] first, double[] second, final double g, final double bandSize) {
        final int m = first.length;
        final int n = second.length;
        final int minLen = Math.min(m, n);
        final int band = getWindowSize(minLen, bandSize);

        final int[][] minWarpingWindow = new int[minLen][minLen];
        double[] prev = new double[minLen];
        double[] curr = new double[minLen];

        double diff, d1, d2, d12, cost;
        int i, j, left, right, absIJ;

        double[] tmp;
        if (n < m) {
            tmp = first;
            first = second;
            second = tmp;
        }


        minWarpingWindow[0][0] = 0;
        for (i = 0; i < m; i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            double[] temp = prev;
            prev = curr;
            curr = temp;

            left = i - (band + 1);
            if (left < 0) {
                left = 0;
            }
            right = i + (band + 1);
            if (right > (m - 1)) {
                right = (m - 1);
            }

            for (j = left; j <= right; j++) {
                absIJ = Math.abs(i - j);
                if (absIJ <= band) {
                    diff = first[i] - g;
                    d1 = (diff * diff);

                    diff = g - second[j];
                    d2 = (diff * diff);

                    diff = first[i] - second[j];
                    d12 = (diff * diff);

                    if ((i + j) != 0) {
                        if ((i == 0) || ((j != 0) &&
                                (((prev[j - 1] + d12) >= (curr[j - 1] + d2)) &&
                                        ((curr[j - 1] + d2) <= (prev[j] + d1))))) {
                            // del
                            cost = curr[j - 1] + d2;
                            minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i][j - 1]);
                        } else if (j == 0 || prev[j - 1] + d12 >= prev[j] + d1 && prev[j] + d1 <= curr[j - 1] + d2) {
                            // ins
                            cost = prev[j] + d1;
                            minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i - 1][j]);
                        } else {
                            // match
                            cost = prev[j - 1] + d12;
                            minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i - 1][j - 1]);
                        }
                    } else {
                        cost = 0;
                        minWarpingWindow[i][j] = 0;
                    }

                    curr[j] = cost;
                    // steps[i][j] = step;
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
        }

        return new WarpingPathResults(curr[m - 1], minWarpingWindow[n - 1][m - 1]);
    }

    public static int getWindowSize(int n, double bandSize) {
        return (int) Math.ceil(bandSize * n);
    }

}
