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
 * WDTW distance
 */
public class WDTW extends ElasticDistances {

    public double distance(final double[] first, final double[] second, final double[] weightVector) {
        final int m = first.length;
        final int n = second.length;
        final double[][] matrixD = new double[n][m];
        double diff;
        double minDistance;
        int i, j;

        //first value
        diff = first[0] - second[0];
        matrixD[0][0] = weightVector[0] * diff * diff;

        //first column
        for (i = 1; i < m; i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + weightVector[i] * diff * diff;
        }

        //top row
        for (j = 1; j < n; j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + weightVector[j] * diff * diff;
        }

        //warp rest
        for (i = 1; i < m; i++) {
            for (j = 1; j < n; j++) {
                //calculate classifiers.nearestNeighbour.distances
                minDistance = Math.min(matrixD[i][j - 1], Math.min(matrixD[i - 1][j], matrixD[i - 1][j - 1]));
                diff = first[i] - second[j];
                matrixD[i][j] = minDistance + weightVector[Math.abs(i - j)] * diff * diff;
            }
        }
        return matrixD[m - 1][n - 1];
    }

    public double distance(final double[] first, final double[] second, final double[] weightVector, final double cutOffValue) {
        boolean tooBig;
        int m = first.length;
        int n = second.length;
        final double[][] matrixD = new double[n][m];
        double diff;
        double minDistance;

        //first value
        diff = first[0] - second[0];
        matrixD[0][0] = weightVector[0] * diff * diff;
        if (matrixD[0][0] > cutOffValue) {
            return Double.MAX_VALUE;
        }

        //first column
        for (int i = 1; i < m; i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + weightVector[i] * diff * diff;
        }

        //top row
        for (int j = 1; j < n; j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + weightVector[j] * diff * diff;
        }

        //warp rest
        for (int i = 1; i < m; i++) {
            tooBig = !(matrixD[i][0] < cutOffValue);
            for (int j = 1; j < n; j++) {
                //calculate classifiers.nearestNeighbour.distances
                minDistance = Math.min(matrixD[i][j - 1], Math.min(matrixD[i - 1][j], matrixD[i - 1][j - 1]));
                diff = first[i] - second[j];
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

}
