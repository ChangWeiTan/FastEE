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
 * DTW distance
 */
public class DTW extends ElasticDistances {
    private final static double[][] matrixD = new double[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH];
    private final static int[][] minWarpingWindow = new int[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH];
    private int w;       // warping window in terms of sequence length
    private double r;    // warping window in terms of percentage

    public DTW() {
    }

    public DTW(final int win) {
        setWindow(win);
    }

    public DTW(final double r) {
        setBandPercent(r);
    }

    @Override
    public final double distance(final Instance first, final Instance second) {
        return distance(first, second, w);
    }

    @Override
    public final double distance(final Instance first, final Instance second, final double cutOffValue) {
        return distance(first, second, w, cutOffValue);
    }

    public final DistanceResults distanceExt(final Instance first, final Instance second) {
        return distanceExt(first, second, w);
    }

    public final DistanceResults distanceExt(final Instance first, final Instance second, final double cutOffValue) {
        return distanceExt(first, second, w, cutOffValue);
    }

    public static double distance(final Instance first, final Instance second, final int windowSize) {
        final int n = first.numAttributes() - 1;
        final int m = second.numAttributes() - 1;

        final int winPlus1 = windowSize + 1;
        double diff;
        int i, j, jStart, jEnd, indexInfyLeft;

        diff = first.value(0) - second.value(0);
        matrixD[0][0] = diff * diff;
        for (i = 1; i < Math.min(n, winPlus1); i++) {
            diff = first.value(i) - second.value(0);
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
        }

        for (j = 1; j < Math.min(m, winPlus1); j++) {
            diff = first.value(0) - second.value(j);
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
        }
        if (j < m)
            matrixD[0][j] = Double.POSITIVE_INFINITY;

        for (i = 1; i < n; i++) {
            jStart = Math.max(1, i - windowSize);
            jEnd = Math.min(m, i + winPlus1);
            indexInfyLeft = i - windowSize - 1;
            if (indexInfyLeft >= 0)
                matrixD[i][indexInfyLeft] = Double.POSITIVE_INFINITY;

            for (j = jStart; j < jEnd; j++) {
                diff = first.value(i) - second.value(j);
                matrixD[i][j] = Tools.min3(matrixD[i - 1][j - 1], matrixD[i][j - 1], matrixD[i - 1][j]) + diff * diff;
            }
            if (j < m)
                matrixD[i][j] = Double.POSITIVE_INFINITY;
        }

        return matrixD[n - 1][m - 1];
    }

    public static double distance(final Instance first, final Instance second, final int windowSize, final double cutOffValue) {
        boolean tooBig;
        final int n = first.numAttributes() - 1;
        final int m = second.numAttributes() - 1;

        double diff;
        int i, j, jStart, jEnd, indexInfyLeft;

        diff = first.value(0) - second.value(0);
        matrixD[0][0] = diff * diff;
        for (i = 1; i < Math.min(n, 1 + windowSize); i++) {
            diff = first.value(i) - second.value(0);
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
        }

        for (j = 1; j < Math.min(m, 1 + windowSize); j++) {
            diff = first.value(0) - second.value(j);
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
        }
        if (j < m)
            matrixD[0][j] = Double.POSITIVE_INFINITY;

        for (i = 1; i < n; i++) {
            tooBig = true;
            jStart = Math.max(1, i - windowSize);
            jEnd = Math.min(m, i + windowSize + 1);
            indexInfyLeft = i - windowSize - 1;
            if (indexInfyLeft >= 0)
                matrixD[i][indexInfyLeft] = Double.POSITIVE_INFINITY;

            for (j = jStart; j < jEnd; j++) {
                diff = first.value(i) - second.value(j);
                matrixD[i][j] = Tools.min3(matrixD[i - 1][j - 1], matrixD[i][j - 1], matrixD[i - 1][j]) + diff * diff;
                if (tooBig && matrixD[i][j] < cutOffValue)
                    tooBig = false;
            }
            //Early abandon
            if (tooBig)
                return Double.POSITIVE_INFINITY;

            if (j < m)
                matrixD[i][j] = Double.POSITIVE_INFINITY;
        }

        return matrixD[n - 1][m - 1];
    }

    public static DistanceResults distanceExt(final Instance first, final Instance second, final int windowSize) {
        double minDist = 0.0;
        int n = first.numAttributes() - 1;
        int m = second.numAttributes() - 1;

        double diff;
        int i, j, indiceRes, absIJ;
        int jStart, jEnd, indexInfyLeft;

        diff = first.value(0) - second.value(0);
        matrixD[0][0] = diff * diff;
        minWarpingWindow[0][0] = 0;
        for (i = 1; i < Math.min(n, 1 + windowSize); i++) {
            diff = first.value(i) - second.value(0);
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
            minWarpingWindow[i][0] = i;
        }

        for (j = 1; j < Math.min(m, 1 + windowSize); j++) {
            diff = first.value(0) - second.value(j);
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
            minWarpingWindow[0][j] = j;
        }
        if (j < m)
            matrixD[0][j] = Double.POSITIVE_INFINITY;

        for (i = 1; i < n; i++) {
            jStart = Math.max(1, i - windowSize);
            jEnd = Math.min(m, i + windowSize + 1);
            indexInfyLeft = i - windowSize - 1;
            if (indexInfyLeft >= 0)
                matrixD[i][indexInfyLeft] = Double.POSITIVE_INFINITY;

            for (j = jStart; j < jEnd; j++) {
                absIJ = Math.abs(i - j);
                indiceRes = Tools.argMin3(matrixD[i - 1][j - 1], matrixD[i][j - 1], matrixD[i - 1][j]);
                switch (indiceRes) {
                    case DIAGONAL:
                        minDist = matrixD[i - 1][j - 1];
                        minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i - 1][j - 1]);
                        break;
                    case LEFT:
                        minDist = matrixD[i][j - 1];
                        minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i][j - 1]);
                        break;
                    case UP:
                        minDist = matrixD[i - 1][j];
                        minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i - 1][j]);
                        break;
                }
                diff = first.value(i) - second.value(j);
                matrixD[i][j] = minDist + diff * diff;
            }
            if (j < m)
                matrixD[i][j] = Double.POSITIVE_INFINITY;
        }

        DistanceResults resExt = new DistanceResults();
        resExt.distance = matrixD[n - 1][m - 1];
        resExt.r = minWarpingWindow[n - 1][m - 1];
        return resExt;
    }

    public static DistanceResults distanceExt(final Instance first, final Instance second, final int windowSize, final double cutOffValue) {
        boolean tooBig;
        double minDist = 0.0;
        int n = first.numAttributes() - 1;
        int m = second.numAttributes() - 1;

        double diff;
        int i, j, indiceRes, absIJ;
        int jStart, jEnd, indexInfyLeft;

        diff = first.value(0) - second.value(0);
        matrixD[0][0] = diff * diff;
        minWarpingWindow[0][0] = 0;
        for (i = 1; i < Math.min(n, 1 + windowSize); i++) {
            diff = first.value(i) - second.value(0);
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
            minWarpingWindow[i][0] = i;
        }

        for (j = 1; j < Math.min(m, 1 + windowSize); j++) {
            diff = first.value(0) - second.value(j);
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
            minWarpingWindow[0][j] = j;
        }
        if (j < m) matrixD[0][j] = Double.POSITIVE_INFINITY;

        for (i = 1; i < n; i++) {
            tooBig = true;
            jStart = Math.max(1, i - windowSize);
            jEnd = Math.min(m, i + windowSize + 1);
            indexInfyLeft = i - windowSize - 1;
            if (indexInfyLeft >= 0) matrixD[i][indexInfyLeft] = Double.POSITIVE_INFINITY;

            for (j = jStart; j < jEnd; j++) {
                absIJ = Math.abs(i - j);
                indiceRes = Tools.argMin3(matrixD[i - 1][j - 1], matrixD[i][j - 1], matrixD[i - 1][j]);
                switch (indiceRes) {
                    case DIAGONAL:
                        minDist = matrixD[i - 1][j - 1];
                        minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i - 1][j - 1]);
                        break;
                    case LEFT:
                        minDist = matrixD[i][j - 1];
                        minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i][j - 1]);
                        break;
                    case UP:
                        minDist = matrixD[i - 1][j];
                        minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i - 1][j]);
                        break;
                }
                diff = first.value(i) - second.value(j);
                matrixD[i][j] = minDist + diff * diff;
                if (tooBig && matrixD[i][j] < cutOffValue) tooBig = false;
            }
            //Early abandon
            if (tooBig) return new DistanceResults(Double.POSITIVE_INFINITY, windowSize);

            if (j < m) matrixD[i][j] = Double.POSITIVE_INFINITY;
        }

        DistanceResults resExt = new DistanceResults();
        resExt.distance = matrixD[n - 1][m - 1];
        resExt.r = minWarpingWindow[n - 1][m - 1];
        return resExt;
    }

    @Override
    public String toString() {
        return "DTW";
    }

    @Override
    public final void setParamsFromParamID(final Instances train, final int paramId) {
        r = 1.0 * paramId / 100;
    }

    public final String getParams() {
        return "r = " + r;
    }

    public final int getWindowSize(final int n) {
        int w = (int) (r * n);
        if (w < 1)
            w = 1;
        else if (w < n)
            w++;
        return w;
    }

    public final void setWindow(final int window) throws IllegalArgumentException {
        if (window < 0) {
            throw new IllegalArgumentException("Window must be > 0");
        }

        w = window;
    }

    public final void setBandPercent(final double bandPercent) throws IllegalArgumentException {
        if (bandPercent < 0 || bandPercent > 1) {
            throw new IllegalArgumentException("Band Size must be between 0 and 1");
        }

        r = bandPercent;
    }
}
