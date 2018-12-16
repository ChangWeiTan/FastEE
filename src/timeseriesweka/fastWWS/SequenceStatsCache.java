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
package timeseriesweka.fastWWS;

import timeseriesweka.lowerBounds.LbErp;
import timeseriesweka.lowerBounds.LbKeogh;
import timeseriesweka.lowerBounds.LbLcss;
import weka.core.Instances;

import java.util.Arrays;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Original code from https://github.com/ChangWeiTan/FastWWSearch
 *
 * Cache for training dataset
 */
public class SequenceStatsCache {
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    private double[][] dtwLEs, dtwUEs, erpLEs, erpUEs, lcssLEs, lcssUEs;
    private double[] mins, maxs;
    private int[] indexMaxs, indexMins;
    private boolean[] isMinFirst, isMinLast, isMaxFirst, isMaxLast;
    private int[] lastDTWWindowComputed;
    private double[] lastERPBandComputed;
    private int[] lastLCSSDeltaComputed;
    private int currentWindow;
    private Instances train;
    private IndexedDouble[][] indicesSortedByAbsoluteValue;
    private double[][] lbDistances;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    public SequenceStatsCache(Instances train, int startingWindow) {
        this.train = train;
        int nSequences = train.numInstances();
        int length = train.instance(0).numAttributes() - 1;
        this.dtwLEs = new double[nSequences][length];
        this.dtwUEs = new double[nSequences][length];
        this.erpLEs = new double[nSequences][length];
        this.erpUEs = new double[nSequences][length];
        this.lcssLEs = new double[nSequences][length];
        this.lcssUEs = new double[nSequences][length];
        this.lastDTWWindowComputed = new int[nSequences];
        Arrays.fill(this.lastDTWWindowComputed, -1);
        this.lastERPBandComputed = new double[nSequences];
        Arrays.fill(this.lastERPBandComputed, -1);
        this.lastLCSSDeltaComputed = new int[nSequences];
        Arrays.fill(this.lastLCSSDeltaComputed, -1);
        this.currentWindow = startingWindow;
        this.mins = new double[nSequences];
        this.maxs = new double[nSequences];
        this.indexMins = new int[nSequences];
        this.indexMaxs = new int[nSequences];
        this.isMinFirst = new boolean[nSequences];
        this.isMinLast = new boolean[nSequences];
        this.isMaxFirst = new boolean[nSequences];
        this.isMaxLast = new boolean[nSequences];
        this.indicesSortedByAbsoluteValue = new IndexedDouble[nSequences][length];
        for (int i = 0; i < train.numInstances(); i++) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            int indexMin = -1, indexMax = -1;
            for (int j = 0; j < train.instance(i).numAttributes() - 1; j++) {
                double elt = train.instance(i).value(j);
                if (elt > max) {
                    max = elt;
                    indexMax = j;
                }
                if (elt < min) {
                    min = elt;
                    indexMin = j;
                }
                indicesSortedByAbsoluteValue[i][j] = new IndexedDouble(j, Math.abs(elt));
            }
            indexMaxs[i] = indexMax;
            indexMins[i] = indexMin;
            mins[i] = min;
            maxs[i] = max;
            isMinFirst[i] = (indexMin == 0);
            isMinLast[i] = (indexMin == (train.instance(i).numAttributes() - 2));
            isMaxFirst[i] = (indexMax == 0);
            isMaxLast[i] = (indexMax == (train.instance(i).numAttributes() - 2));
            Arrays.sort(indicesSortedByAbsoluteValue[i], (v1, v2) -> -Double.compare(v1.value, v2.value));
        }
    }


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Methods
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    public double[] getDTWLE(int i, int w) {
        if (lastDTWWindowComputed[i] != w) {
            computeDTWLEandUE(i, w);
        }
        return dtwLEs[i];
    }

    public double[] getDTWUE(int i, int w) {
        if (lastDTWWindowComputed[i] != w) {
            computeDTWLEandUE(i, w);
        }
        return dtwUEs[i];
    }

    public double[] getERPLE(int i, double g, double bandSize) {
        if (lastERPBandComputed[i] != bandSize) {
            computeERPLEandUE(i, g, bandSize);
        }
        return erpLEs[i];
    }

    public double[] getERPUE(int i, double g, double bandSize) {
        if (lastERPBandComputed[i] != bandSize) {
            computeERPLEandUE(i, g, bandSize);
        }
        return erpUEs[i];
    }

    public double[] getLCSSLE(int i, int delta, double epsilon) {
        if (lastLCSSDeltaComputed[i] != delta) {
            computeLCSSLEandUE(i, delta, epsilon);
        }
        return lcssLEs[i];
    }

    public double[] getLCSSUE(int i, int delta, double epsilon) {
        if (lastLCSSDeltaComputed[i] != delta) {
            computeERPLEandUE(i, delta, epsilon);
        }
        return lcssUEs[i];
    }

    private void computeDTWLEandUE(int i, int w) {
        LbKeogh.fillUL(train.instance(i), w, dtwUEs[i], dtwLEs[i]);
        this.lastDTWWindowComputed[i] = w;
    }

    private void computeERPLEandUE(int i, double g, double bandSize) {
        LbErp.fillUL(train.instance(i), g, bandSize, erpUEs[i], erpLEs[i]);
        this.lastERPBandComputed[i] = bandSize;
    }

    private void computeLCSSLEandUE(int i, int delta, double epsilon) {
        LbLcss.fillUL(train.instance(i), epsilon, delta, lcssUEs[i], lcssLEs[i]);
        this.lastLCSSDeltaComputed[i] = delta;
    }

    public boolean isMinFirst(int i) {
        return isMinFirst[i];
    }

    public boolean isMaxFirst(int i) {
        return isMaxFirst[i];
    }

    public boolean isMinLast(int i) {
        return isMinLast[i];
    }

    public boolean isMaxLast(int i) {
        return isMaxLast[i];
    }

    public double getMin(int i) {
        return mins[i];
    }

    public double getMax(int i) {
        return maxs[i];
    }

    public int getIMax(int i) {
        return indexMaxs[i];
    }

    public int getIMin(int i) {
        return indexMins[i];
    }

    public int getIndexNthHighestVal(int i, int n) {
        return indicesSortedByAbsoluteValue[i][n].index;
    }

    public void initLbDistances() {
        lbDistances = new double[train.numInstances()][train.numInstances()];
    }

    public void setLbDistances(double lb, int qIndex, int cIndex){
        lbDistances[qIndex][cIndex] = lb;
    }

    public double getLbDistances(int qIndex, int cIndex) {
        return lbDistances[qIndex][cIndex];
    }

    public boolean lbDistanceExist(int qIndex, int cIndex){
        return lbDistances[qIndex][cIndex] != 0;
    }

    class IndexedDouble {
        double value;
        int index;

        IndexedDouble(int index, double value) {
            this.value = value;
            this.index = index;
        }
    }
}
