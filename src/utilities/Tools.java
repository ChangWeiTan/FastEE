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
package utilities;

import weka.core.Instances;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Some simple tools
 */
public class Tools {
    public static int argMin3(final double a, final double b, final double c) {
        return (a <= b) ? ((a <= c) ? 0 : 2) : (b <= c) ? 1 : 2;
    }

    public static double min3(final double a, final double b, final double c) {
        return (a <= b) ? ((a <= c) ? a : c) : (b <= c) ? b : c;
    }

    public static double stdv_s(double[] input) {
        double sumx = 0;
        double sumx2 = 0;

        for (double anInput : input) {
            sumx += anInput;
            sumx2 += anInput * anInput;
        }

        int n = input.length;
        double mean = sumx / n;

        return Math.sqrt(sumx2 / (n - 1) - mean * mean);
    }

    public static double stdv_p(Instances input) {
        double sumx = 0;
        double sumx2 = 0;
        double[] ins2array;
        for (int i = 0; i < input.numInstances(); i++) {
            ins2array = input.instance(i).toDoubleArray();
            for (int j = 0; j < ins2array.length - 1; j++) {//-1 to avoid classVal
                sumx += ins2array[j];
                sumx2 += ins2array[j] * ins2array[j];
            }
        }
        int n = input.numInstances() * (input.numAttributes() - 1);
        double mean = sumx / n;
        return Math.sqrt(sumx2 / (n) - mean * mean);

    }

    public static int[] getInclusive10(int min, int max) {
        int[] output = new int[10];

        double diff = (double) (max - min) / 9;
        double[] doubleOut = new double[10];
        doubleOut[0] = min;
        output[0] = min;
        for (int i = 1; i < 9; i++) {
            doubleOut[i] = doubleOut[i - 1] + diff;
            output[i] = (int) Math.round(doubleOut[i]);
        }
        output[9] = max; // to make sure max isn't omitted due to double imprecision
        return output;
    }

    public static double[] getInclusive10(double min, double max) {
        double[] output = new double[10];
        double diff = (max - min) / 9;
        output[0] = min;
        for (int i = 1; i < 9; i++) {
            output[i] = output[i - 1] + diff;
        }
        output[9] = max;

        return output;
    }
}
