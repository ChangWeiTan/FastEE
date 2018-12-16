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
 * LCSS distance
 */
public class LCSS extends ElasticDistances {
    private static int[][] lcss = new int[MAX_SEQ_LENGTH + 1][MAX_SEQ_LENGTH + 1];
    private static int[][] minDelta = new int[MAX_SEQ_LENGTH + 1][MAX_SEQ_LENGTH + 1];
    private double epsilon;
    private int delta;
    private double[] epsilons;
    private int[] deltas;

    public LCSS() {
    }

    public LCSS(double epsilon, int delta) {
        setDelta(delta);
        setEpsilon(epsilon);
    }

    @Override
    public final double distance(final Instance first, final Instance second) {
        return distance(first, second, epsilon, delta);
    }

    @Override
    public final double distance(final Instance first, final Instance second, final double cutOffValue) {
        return distance(first, second, epsilon, delta);
    }

    public final DistanceResults distanceExt(final Instance first, final Instance second) {
        return distanceExt(first, second, epsilon, delta);
    }

    public final DistanceResults distanceExt(final Instance first, final Instance second, final double cutOffValue) {
        return distanceExt(first, second, epsilon, delta);
    }

    public static double distance(final Instance first, final Instance second, final double epsilon, final int delta) {
        final int m = first.numAttributes() - 1;
        final int n = second.numAttributes() - 1;
        int i, j;
        lcss = new int[m + 1][n + 1];

        for (i = 0; i < m; i++) {
            for (j = i - delta; j <= i + delta; j++) {
                if (j < 0) {
                    j = -1;
                } else if (j >= n) {
                    j = i + delta;
                } else if (second.value(j) + epsilon >= first.value(i) &&
                        second.value(j) - epsilon <= first.value(i)) {
                    lcss[i + 1][j + 1] = lcss[i][j] + 1;
                } else if (delta == 0) {
                    lcss[i + 1][j + 1] = lcss[i][j];
                } else if (lcss[i][j + 1] > lcss[i + 1][j]) {
                    lcss[i + 1][j + 1] = lcss[i][j + 1];
                } else {
                    lcss[i + 1][j + 1] = lcss[i + 1][j];
                }
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

    public static DistanceResults distanceExt(final Instance first, final Instance second, final double epsilon, final int delta) {
        final int m = first.numAttributes() - 1;
        final int n = second.numAttributes() - 1;
        int i, j, absIJ;
        lcss = new int[m + 1][n + 1];
        minDelta = new int[m + 1][n + 1];

        for (i = 0; i < m; i++) {
            for (j = i - delta; j <= i + delta; j++) {
                if (j < 0) {
                    j = -1;
                } else if (j >= n) {
                    j = i + delta;
                } else if (second.value(j) + epsilon >= first.value(i) &&
                        second.value(j) - epsilon <= first.value(i)) {
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
        DistanceResults resExt = new DistanceResults();
        resExt.distance = 1.0 - 1.0 * lcss[m][n] / m;
        resExt.r = maxR;
        return resExt;
    }

    @Override
    public final String toString() {
        return "LCSS";
    }

    @Override
    public final void setParamsFromParamID(final Instances train, final int paramId) {
        if (!paramsRefreshed) {
            double stdTrain = Tools.stdv_p(train);
            double stdFloor = stdTrain * 0.2;
            epsilons = Tools.getInclusive10(stdFloor, stdTrain);
            deltas = Tools.getInclusive10(0, (train.numAttributes() - 1) / 4);
            paramsRefreshed = true;
        }

        delta = deltas[paramId % 10];
        epsilon = epsilons[paramId / 10];
    }

    public final String getParams() {
        return "delta = " + delta + ", epsilon = " + df.format(epsilon);
    }

    public final void setEpsilon(final double epsilon) {
        this.epsilon = epsilon;
    }

    public final void setDelta(final int delta) throws IllegalArgumentException {
        if (delta < 0) {
            throw new IllegalArgumentException("Delta must be > 0");
        }

        this.delta = delta;
    }

}
