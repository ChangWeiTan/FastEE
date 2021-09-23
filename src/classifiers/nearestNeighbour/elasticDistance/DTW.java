package classifiers.nearestNeighbour.elasticDistance;

import results.WarpingPathResults;
import utils.GenericTools;

import java.util.Arrays;

import static java.lang.Integer.max;
import static java.lang.Integer.min;

public class DTW extends ElasticDistances {
    public double distance(final double[] first, final double[] second) {
        final int n = first.length;
        final int m = second.length;
        final double[][] matrixD = new double[n][m];

        double diff;
        int i, j;

        diff = first[0] - second[0];
        matrixD[0][0] = diff * diff;
        for (i = 1; i < n; i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
        }

        for (j = 1; j < m; j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
        }

        for (i = 1; i < n; i++) {
            for (j = 1; j < m; j++) {
                diff = first[i] - second[j];
                matrixD[i][j] = GenericTools.min3(
                        matrixD[i - 1][j - 1],
                        matrixD[i][j - 1],
                        matrixD[i - 1][j]
                ) + diff * diff;
            }
        }

        return matrixD[n - 1][m - 1];
    }

    public double distance(final double[] first, final double[] second, final double cutOffValue) {
        boolean tooBig;
        final int n = first.length;
        final int m = second.length;
        final double[][] matrixD = new double[n][m];

        double diff;
        int i, j;

        diff = first[0] - second[0];
        matrixD[0][0] = diff * diff;
        for (i = 1; i < n; i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
        }

        for (j = 1; j < m; j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
        }

        for (i = 1; i < n; i++) {
            tooBig = true;

            for (j = 1; j < m; j++) {
                diff = first[i] - second[j];
                matrixD[i][j] = GenericTools.min3(
                        matrixD[i - 1][j - 1],
                        matrixD[i][j - 1],
                        matrixD[i - 1][j]
                ) + diff * diff;
                if (tooBig && matrixD[i][j] < cutOffValue)
                    tooBig = false;
            }
            //Early abandon
            if (tooBig)
                return Double.POSITIVE_INFINITY;
        }

        return matrixD[n - 1][m - 1];
    }

    public double distance(double[] first, double[] second, final int windowSize) {
        final int n = first.length;
        final int m = second.length;
        final double[][] matrixD = new double[n][m];

        double diff;
        int i, j, jStart, jEnd, indexInfyLeft;

        diff = first[0] - second[0];
        matrixD[0][0] = diff * diff;
        for (i = 1; i < Math.min(n, 1 + windowSize); i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
        }

        for (j = 1; j < Math.min(m, 1 + windowSize); j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
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
                diff = first[i] - second[j];
                matrixD[i][j] = GenericTools.min3(
                        matrixD[i - 1][j - 1],
                        matrixD[i][j - 1],
                        matrixD[i - 1][j]
                ) + diff * diff;
            }
            if (j < m)
                matrixD[i][j] = Double.POSITIVE_INFINITY;
        }

        return matrixD[n - 1][m - 1];
    }

    public double distance(final double[] first, final double[] second, final int windowSize, final double cutOffValue) {
        boolean tooBig;
        final int n = first.length;
        final int m = second.length;
        final double[][] matrixD = new double[n][m];

        double diff;
        int i, j, jStart, jEnd, indexInfyLeft;

        diff = first[0] - second[0];
        matrixD[0][0] = diff * diff;
        for (i = 1; i < Math.min(n, 1 + windowSize); i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
        }

        for (j = 1; j < Math.min(m, 1 + windowSize); j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
        }
        if (j < m)
            matrixD[0][j] = Double.POSITIVE_INFINITY;

        for (i = 1; i < n; i++) {
            tooBig = !(matrixD[i][0] < cutOffValue);
            jStart = Math.max(1, i - windowSize);
            jEnd = Math.min(m, i + windowSize + 1);
            indexInfyLeft = i - windowSize - 1;
            if (indexInfyLeft >= 0)
                matrixD[i][indexInfyLeft] = Double.POSITIVE_INFINITY;

            for (j = jStart; j < jEnd; j++) {
                diff = first[i] - second[j];
                matrixD[i][j] = GenericTools.min3(
                        matrixD[i - 1][j - 1],
                        matrixD[i][j - 1],
                        matrixD[i - 1][j]
                ) + diff * diff;
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

    public int getWindowSize(final int n, final double r) {
//        return (int) Math.ceil(r * n);
        return (int) (r * n);
    }

    public WarpingPathResults distanceExt(final double[] first, final double[] second, final int windowSize) {
        double minDist = 0.0;
        final int n = first.length;
        final int m = second.length;
        final double[][] matrixD = new double[n][m];
        final int[][] minDistanceToDiagonal = new int[n][m];

        double diff;
        int i, j, indiceRes, absIJ;
        int jStart, jEnd, indexInfyLeft;

        diff = first[0] - second[0];
        matrixD[0][0] = diff * diff;
        minDistanceToDiagonal[0][0] = 0;
        for (i = 1; i < Math.min(n, 1 + windowSize); i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
            minDistanceToDiagonal[i][0] = i;
        }

        for (j = 1; j < Math.min(m, 1 + windowSize); j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
            minDistanceToDiagonal[0][j] = j;
        }
        if (j < m) matrixD[0][j] = Double.POSITIVE_INFINITY;

        for (i = 1; i < n; i++) {
            jStart = Math.max(1, i - windowSize);
            jEnd = Math.min(m, i + windowSize + 1);
            indexInfyLeft = i - windowSize - 1;
            if (indexInfyLeft >= 0) matrixD[i][indexInfyLeft] = Double.POSITIVE_INFINITY;

            for (j = jStart; j < jEnd; j++) {
                absIJ = Math.abs(i - j);
                indiceRes = GenericTools.argMin3(
                        matrixD[i - 1][j - 1],  // diagonal
                        matrixD[i][j - 1],      // left
                        matrixD[i - 1][j]       // up
                );
                switch (indiceRes) {
                    case 0:
                        minDist = matrixD[i - 1][j - 1];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i - 1][j - 1]);
                        break;
                    case 1:
                        minDist = matrixD[i][j - 1];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i][j - 1]);
                        break;
                    case 2:
                        minDist = matrixD[i - 1][j];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i - 1][j]);
                        break;
                }
                diff = first[i] - second[j];
                matrixD[i][j] = minDist + diff * diff;
            }
            if (j < m) matrixD[i][j] = Double.POSITIVE_INFINITY;
        }

        return new WarpingPathResults(matrixD[n - 1][m - 1], minDistanceToDiagonal[n - 1][m - 1]);
    }

    public WarpingPathResults distanceExt(final double[] first, final double[] second,
                                          final int windowSize, final double cutOffValue) {
        boolean tooBig;
        double minDist = 0.0;
        final int n = first.length;
        final int m = second.length;
        final double[][] matrixD = new double[n][m];
        final int[][] minDistanceToDiagonal = new int[n][m];

        double diff;
        int i, j, indiceRes, absIJ;
        int jStart, jEnd, indexInfyLeft;

        diff = first[0] - second[0];
        matrixD[0][0] = diff * diff;
        minDistanceToDiagonal[0][0] = 0;
        for (i = 1; i < Math.min(n, 1 + windowSize); i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
            minDistanceToDiagonal[i][0] = i;
        }

        for (j = 1; j < Math.min(m, 1 + windowSize); j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
            minDistanceToDiagonal[0][j] = j;
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
                indiceRes = GenericTools.argMin3(
                        matrixD[i - 1][j - 1], // 0
                        matrixD[i][j - 1],  // 1
                        matrixD[i - 1][j] // 2
                );
                switch (indiceRes) {
                    case 0:
                        minDist = matrixD[i - 1][j - 1];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i - 1][j - 1]);
                        break;
                    case 1:
                        minDist = matrixD[i][j - 1];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i][j - 1]);
                        break;
                    case 2:
                        minDist = matrixD[i - 1][j];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i - 1][j]);
                        break;
                }
                diff = first[i] - second[j];
                matrixD[i][j] = minDist + diff * diff;
                if (tooBig && matrixD[i][j] < cutOffValue) tooBig = false;
            }
            //Early abandon
            if (tooBig) return new WarpingPathResults();

            if (j < m) matrixD[i][j] = Double.POSITIVE_INFINITY;
        }

        return new WarpingPathResults(matrixD[n - 1][m - 1], minDistanceToDiagonal[n - 1][m - 1]);
    }
}
