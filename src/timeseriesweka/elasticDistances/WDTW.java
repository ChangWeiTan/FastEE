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
 * WDTW distance
 */
public class WDTW extends ElasticDistances {
    private final static double WEIGHT_MAX = 1;
    // initialise the matrices with the maximum sequence length for faster computation
    private final static double[][] matrixD = new double[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH];
    private double g;
    private double[] weightVector;

    public WDTW() {
        g = 0;
        weightVector = null;
    }

    public WDTW(final double g) {
        this.g = g;
        weightVector = null;
    }

    public WDTW(final double g, final double[] weightVector) {
        this.g = g;
        this.weightVector = weightVector;
    }

    @Override
    public final double distance(final Instance first, final Instance second) {
        return distance(first, second, weightVector);
    }

    @Override
    public final double distance(final Instance first, final Instance second, final double cutOffValue) {
        return distance(first, second, weightVector, cutOffValue);
    }

    public static double distance(final Instance first, final Instance second, final double[] weightVector) {
        final int m = first.numAttributes() - 1;
        final int n = second.numAttributes() - 1;
        double diff;
        double minDistance;
        int i, j;

        //first value
        diff = first.value(0) - second.value(0);
        matrixD[0][0] = weightVector[0] * diff * diff;

        //first column
        for (i = 1; i < m; i++) {
            diff = first.value(i) - second.value(0);
            matrixD[i][0] = matrixD[i - 1][0] + weightVector[i] * diff * diff;
        }

        //top row
        for (j = 1; j < n; j++) {
            diff = first.value(0) - second.value(j);
            matrixD[0][j] = matrixD[0][j - 1] + weightVector[j] * diff * diff;
        }

        //warp rest
        for (i = 1; i < m; i++) {
            for (j = 1; j < n; j++) {
                //calculate distances
                minDistance = Math.min(matrixD[i][j - 1], Math.min(matrixD[i - 1][j], matrixD[i - 1][j - 1]));
                diff = first.value(i) - second.value(j);
                matrixD[i][j] = minDistance + weightVector[Math.abs(i - j)] * diff * diff;
            }
        }
        return matrixD[m - 1][n - 1];
    }

    public static double distance(final Instance first, final Instance second, final double[] weightVector, final double cutOffValue) {
        boolean tooBig;
        int m = first.numAttributes() - 1;
        int n = second.numAttributes() - 1;
        double diff;
        double minDistance;

        //first value
        diff = first.value(0) - second.value(0);
        matrixD[0][0] = weightVector[0] * diff * diff;
        if (matrixD[0][0] > cutOffValue) {
            return Double.MAX_VALUE;
        }

        //first column
        for (int i = 1; i < m; i++) {
            diff = first.value(i) - second.value(0);
            matrixD[i][0] = matrixD[i - 1][0] + weightVector[i] * diff * diff;
        }

        //top row
        for (int j = 1; j < n; j++) {
            diff = first.value(0) - second.value(j);
            matrixD[0][j] = matrixD[0][j - 1] + weightVector[j] * diff * diff;
        }

        //warp rest
        for (int i = 1; i < m; i++) {
            tooBig = true;
            for (int j = 1; j < n; j++) {
                //calculate distances
                minDistance = Math.min(matrixD[i][j - 1], Math.min(matrixD[i - 1][j], matrixD[i - 1][j - 1]));
                diff = first.value(i) - second.value(j);
                matrixD[i][j] = minDistance + weightVector[Math.abs(i - j)] * diff * diff;
                if (tooBig && matrixD[i][j] < cutOffValue) {
                    tooBig = false;
                }
            }
            //Early abandon
            if (tooBig) {
                return Double.MAX_VALUE;
            }
        }
        return matrixD[m - 1][n - 1];
    }

    @Override
    public final String toString() {
        return "WDTW";
    }

    @Override
    public final void setParamsFromParamID(final Instances train, final int paramId) {
        g = (double) paramId / 100;
        paramsRefreshed = true;
        initWeights(train.numAttributes() - 1);
    }

    public final double[] initWeights(final int seriesLength) {
        weightVector = new double[seriesLength];
        double halfLength = (double) seriesLength / 2;

        for (int i = 0; i < seriesLength; i++) {
            weightVector[i] = WEIGHT_MAX / (1 + Math.exp(-g * (i - halfLength)));
        }
        paramsRefreshed = false;
        return weightVector;
    }

    public final String getParams() {
        return "g = " + g;
    }

    public final double[] getWeightVector() {
        return weightVector;
    }

    public final double getG() {
        return g;
    }

    public final void setWeightVector(final double[] wv) {
        weightVector = wv;
    }

    public final void setG(final int gg) {
        g = gg;
    }
}
