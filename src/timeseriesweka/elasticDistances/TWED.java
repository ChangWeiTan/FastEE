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
 * TWED distance
 */
public class TWED extends ElasticDistances {
    private final static double[][] D = new double[MAX_SEQ_LENGTH + 1][MAX_SEQ_LENGTH + 1];
    private final static double[] Di1 = new double[MAX_SEQ_LENGTH + 1];
    private final static double[] Dj1 = new double[MAX_SEQ_LENGTH + 1];

    public static double[] twe_nuParams = {
            // <editor-fold defaultstate="collapsed" desc="hidden for space">
            0.00001,
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1,// </editor-fold>
    };
    public static double[] twe_lamdaParams = {
            // <editor-fold defaultstate="collapsed" desc="hidden for space">
            0,
            0.011111111,
            0.022222222,
            0.033333333,
            0.044444444,
            0.055555556,
            0.066666667,
            0.077777778,
            0.088888889,
            0.1,// </editor-fold>
    };
    private double nu;
    private double lambda;  // parameters for TWED

    public TWED() {
    }

    public TWED(final double nu, final double lambda) {
        setNu(nu);
        setLambda(lambda);
    }

    @Override
    public final double distance(final Instance first, final Instance second) {
        return distance(first, second, nu, lambda);
    }

    @Override
    public final double distance(final Instance first, final Instance second, final double cutOffValue) {
        return distance(first, second, nu, lambda);
    }

    public static double distance(final Instance first, final Instance second, final double nu, final double lambda) {
        final int m = first.numAttributes() - 1;
        final int n = second.numAttributes() - 1;

        double diff, dist;
        double dmin, htrans;
        int i, j;

        // local costs initializations
        for (j = 1; j <= n; j++) {
            if (j > 1) {
                Dj1[j] = second.value(j - 2) - second.value(j - 1);
                Dj1[j] = Dj1[j] * Dj1[j];
            } else {
                Dj1[j] = second.value(j - 1) * second.value(j - 1);
            }
        }

        for (i = 1; i <= m; i++) {
            if (i > 1) {
                Di1[i] = first.value(i - 2) - first.value(i - 1);
                Di1[i] = Di1[i] * Di1[i];
            } else {
                Di1[i] = first.value(i - 1) * first.value(i - 1);
            }

            for (j = 1; j <= n; j++) {
                D[i][j] = first.value(i - 1) - second.value(j - 1);
                D[i][j] = D[i][j] * D[i][j];
                if (i > 1 && j > 1) {
                    diff = first.value(i - 2) - second.value(j - 2);
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

    @Override
    public final String toString() {
        return "TWED";
    }

    @Override
    public final void setParamsFromParamID(final Instances train, final int paramId) {
        nu = twe_nuParams[paramId / 10];
        lambda = twe_lamdaParams[paramId % 10];
    }

    public final String getParams() {
        return "nu = " + nu + ", lambda = " + df.format(lambda);
    }

    public final double getNu() {
        return nu;
    }

    public final double getLambda() {
        return lambda;
    }

    public final void setNu(final double nu) {
        this.nu = nu;
    }

    public final void setLambda(final double lambda) {
        this.lambda = lambda;
    }
}
