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
 * MSM distance
 */
public class MSM extends ElasticDistances {

    public double distance(final double[] first, final double[] second, final double c) {
        final int m = first.length;
        final int n = second.length;
        final double[][] matrixD = new double[m][n];
        int i, j;
        double d1, d2, d3;

        // Initialization
        matrixD[0][0] = Math.abs(first[0] - second[0]);
        for (i = 1; i < m; i++) {
            matrixD[i][0] = matrixD[i - 1][0] + editCost(first[i], first[i - 1], second[0], c);
        }
        for (i = 1; i < n; i++) {
            matrixD[0][i] = matrixD[0][i - 1] + editCost(second[i], first[0], second[i - 1], c);
        }

        // Main Loop
        for (i = 1; i < m; i++) {
            for (j = 1; j < n; j++) {
                d1 = matrixD[i - 1][j - 1] + Math.abs(first[i] - second[j]);
                d2 = matrixD[i - 1][j] + editCost(first[i], first[i - 1], second[j], c);
                d3 = matrixD[i][j - 1] + editCost(second[j], first[i], second[j - 1], c);
                matrixD[i][j] = Math.min(d1, Math.min(d2, d3));
            }
        }
        // Output
        return matrixD[m - 1][n - 1];
    }

    public double distance(final double[] first, final double[] second, final double c, final double cutOffValue) {
        final int m = first.length;
        final int n = second.length;
        final double[][] matrixD = new double[m][n];
        int i, j;
        double d1, d2, d3;
        double min;

        // Initialization
        matrixD[0][0] = Math.abs(first[0] - second[0]);
        for (i = 1; i < m; i++) {
            matrixD[i][0] = matrixD[i - 1][0] + editCost(first[i], first[i - 1], second[0], c);
        }
        for (i = 1; i < n; i++) {
            matrixD[0][i] = matrixD[0][i - 1] + editCost(second[i], first[0], second[i - 1], c);
        }

        // Main Loop
        for (i = 1; i < m; i++) {
            min = cutOffValue;
            for (j = 1; j < n; j++) {
                d1 = matrixD[i - 1][j - 1] + Math.abs(first[i] - second[j]);
                d2 = matrixD[i - 1][j] + editCost(first[i], first[i - 1], second[j], c);
                d3 = matrixD[i][j - 1] + editCost(second[j], first[i], second[j - 1], c);
                matrixD[i][j] = Math.min(d1, Math.min(d2, d3));

                if (matrixD[i][j] >= cutOffValue) {
                    matrixD[i][j] = Double.MAX_VALUE;
                }

                if (matrixD[i][j] < min) {
                    min = matrixD[i][j];
                }
            }
            if (min >= cutOffValue) {
                return Double.MAX_VALUE;
            }
        }
        // Output
        return matrixD[m - 1][n - 1];
    }

    private double editCost(final double new_point, final double x, final double y, final double c) {
        double dist;

        if (((x <= new_point) && (new_point <= y)) || ((y <= new_point) && (new_point <= x))) {
            dist = c;
        } else {
            dist = c + Math.min(Math.abs(new_point - x), Math.abs(new_point - y));
        }

        return dist;
    }
}
