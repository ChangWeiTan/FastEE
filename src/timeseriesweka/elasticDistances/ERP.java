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
package timeseriesweka.elasticDistances;

import utilities.Tools;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Original code from http://www.timeseriesclassification.com/code.php
 * https://bitbucket.org/TonyBagnall/time-series-classification/src/f47c300b937af1d06aac86bdbe228cb9a9d5e00d?at=default
 *
 * ERP distance
 */
public class ERP extends ElasticDistances {
    private static double[] prev = new double[MAX_SEQ_LENGTH];
    private static double[] curr = new double[MAX_SEQ_LENGTH];
    private final static int[][] minWarpingWindow = new int[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH];
    private double g;
    private double bandSize;
    private double[] windowSizes;
    private double[] gValues;

    public ERP() {

    }

    public ERP(final double g, final double bandSize) {
        setG(g);
        setBandSize(bandSize);
    }

    @Override
    public final double distance(final Instance first, final Instance second) {
        return distance(first, second, g, bandSize);
    }

    @Override
    public final double distance(final Instance first, final Instance second, final double cutOffValue) {
        //Todo: add cutoff ERP
        return distance(first, second, g, bandSize);
    }

    public final DistanceResults distanceExt(final Instance first, final Instance second) {
        return distanceExt(first, second, g, bandSize);
    }

    public final DistanceResults distanceExt(final Instance first, final Instance second, final double cutOffValue) {
        return distanceExt(first, second, g, bandSize);
    }

    public static double distance(Instance first, Instance second, final double g, final double bandSize) {
        final int m = first.numAttributes() - 1;
        final int n = second.numAttributes() - 1;
        final int band = getWindowSize(bandSize, m);
        double diff, d1, d2, d12, cost;
        int i, j, left, right, absIJ;

        Instance tmp;
        if (n < m) {
            tmp = first;
            first = second;
            second = tmp;
        }

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
                    diff = first.value(i) - g;
                    d1 = (diff * diff);

                    diff = g - second.value(j);
                    d2 = (diff * diff);

                    diff = first.value(i) - second.value(j);
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

    public static DistanceResults distanceExt(Instance first, Instance second, final double g, final double bandSize) {
        final int m = first.numAttributes() - 1;
        final int n = second.numAttributes() - 1;
        final int band = getWindowSize(bandSize, m);
        double diff, d1, d2, d12, cost;
        int i, j, left, right, absIJ;

        Instance tmp;
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
                    diff = first.value(i) - g;
                    d1 = (diff * diff);

                    diff = g - second.value(j);
                    d2 = (diff * diff);

                    diff = first.value(i) - second.value(j);
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

        DistanceResults resExt = new DistanceResults();
        resExt.distance = curr[m - 1];
        resExt.r = minWarpingWindow[n - 1][m - 1];
        return resExt;
    }

    @Override
    public final String toString() {
        return "ERP";
    }

    @Override
    public final void setParamsFromParamID(final Instances train, final int paramId) {
        if (!paramsRefreshed) {
            final double stdv = Tools.stdv_p(train);
            windowSizes = Tools.getInclusive10(0, 0.25);
            gValues = Tools.getInclusive10(0.2 * stdv, stdv);
            paramsRefreshed = true;
        }

        g = gValues[paramId / 10];
        bandSize = windowSizes[paramId % 10];
    }

    public final String getParams() {
        return "g = " + df.format(g) + ", bandSize = " + df.format(bandSize);
    }

    public static int getWindowSize(double bandSize, int n) {
        return (int) Math.ceil(bandSize * n);
    }

    public final double getBandSize() {
        return bandSize;
    }

    public final double getG() {
        return g;
    }

    public final void setG(final double g) {
        this.g = g;
    }

    public void setBandSize(final double bandSize) throws IllegalArgumentException {
        if (bandSize < 0 || bandSize > 1) {
            throw new IllegalArgumentException("Band Size must be between 0 and 1");
        }

        this.bandSize = bandSize;
    }
}
