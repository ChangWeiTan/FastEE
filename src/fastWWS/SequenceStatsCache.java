package fastWWS;


import classifiers.nearestNeighbour.elasticDistance.lowerBounds.LbErp;
import classifiers.nearestNeighbour.elasticDistance.lowerBounds.LbKeogh;
import classifiers.nearestNeighbour.elasticDistance.lowerBounds.LbLcss;
import classifiers.nearestNeighbour.elasticDistance.lowerBounds.LbTwed;
import datasets.Sequences;
import utils.IndexedDouble;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * This class stores the status of the sequences.
 */
public class SequenceStatsCache {
    protected ArrayList<double[]> LEs, UEs;
    protected double[] mins, maxs;
    protected int[] indexMaxs, indexMins;
    protected boolean[] isMinFirst, isMinLast, isMaxFirst, isMaxLast;
    protected double[] lastWindowComputed;
    protected double[] lastERPWindowComputed;
    protected double[] lastLCSSWindowComputed;
    protected int currentWindow;
    protected Sequences train;
    protected IndexedDouble[][] indicesSortedByAbsoluteValue;
    protected double[][] lbDistances;
    protected LbKeogh lbKeoghComputer = new LbKeogh();
    protected LbErp lbERPcomputer = new LbErp();
    protected LbLcss lbLCSScomputer = new LbLcss();


    public SequenceStatsCache(final Sequences train, final int startingWindow) {
        this.train = train;

        int nSequences = train.size();
        int length = train.get(0).length();
        this.currentWindow = startingWindow;

        this.lastWindowComputed = new double[nSequences];
        this.lastERPWindowComputed = new double[nSequences];
        this.lastLCSSWindowComputed = new double[nSequences];
        Arrays.fill(this.lastWindowComputed, -1);
        Arrays.fill(this.lastERPWindowComputed, -1);
        Arrays.fill(this.lastLCSSWindowComputed, -1);

        this.LEs = new ArrayList<>(nSequences);
        this.UEs = new ArrayList<>(nSequences);
        this.mins = new double[nSequences];
        this.maxs = new double[nSequences];
        this.indexMins = new int[nSequences];
        this.indexMaxs = new int[nSequences];
        this.isMinFirst = new boolean[nSequences];
        this.isMinLast = new boolean[nSequences];
        this.isMaxFirst = new boolean[nSequences];
        this.isMaxLast = new boolean[nSequences];
        this.indicesSortedByAbsoluteValue = new IndexedDouble[nSequences][length];
        for (int i = 0; i < train.size(); i++) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            int indexMin = -1, indexMax = -1;
            for (int j = 0; j < train.get(i).length(); j++) {
                double val = train.get(i).value(j);
                if (val > max) {
                    max = val;
                    indexMax = j;
                }
                if (val < min) {
                    min = val;
                    indexMin = j;
                }
                indicesSortedByAbsoluteValue[i][j] = new IndexedDouble(j, Math.abs(val));
            }
            for (int j = train.get(i).length(); j < train.get(0).length(); j++) {
                indicesSortedByAbsoluteValue[i][j] = new IndexedDouble(j, Double.POSITIVE_INFINITY);
            }
            indexMaxs[i] = indexMax;
            indexMins[i] = indexMin;
            mins[i] = min;
            maxs[i] = max;
            isMinFirst[i] = (indexMin == 0);
            isMinLast[i] = (indexMin == (train.get(0).length() - 1));
            isMaxFirst[i] = (indexMax == 0);
            isMaxLast[i] = (indexMax == (train.get(0).length() - 1));
            Arrays.sort(indicesSortedByAbsoluteValue[i], (v1, v2) -> -Double.compare(v1.value, v2.value));
            this.LEs.add(new double[length]);
            this.UEs.add(new double[length]);
        }
    }

    public double[] getLE(final int i, final int w) {
        if (lastWindowComputed[i] != w) {
            LEs.set(i, new double[train.get(i).length()]);
            UEs.set(i, new double[train.get(i).length()]);
            computeLEandUE(i, w);
        }
        return LEs.get(i);
    }


    public double[] getUE(final int i, final int w) {
        if (lastWindowComputed[i] != w) {
            LEs.set(i, new double[train.get(i).length()]);
            UEs.set(i, new double[train.get(i).length()]);
            computeLEandUE(i, w);
        }

        return UEs.get(i);
    }

    public void computeLEandUE(final int i, final int r) {
        lbKeoghComputer.fillUL(train.get(i).data[0], r, UEs.get(i), LEs.get(i));
        this.lastWindowComputed[i] = r;
    }

    public double[] getLE(final int i, final double g, final double bandSize) {
        if (lastERPWindowComputed[i] != bandSize) {
            LEs.set(i, new double[train.get(i).length()]);
            UEs.set(i, new double[train.get(i).length()]);
            computeLEandUE(i, g, bandSize);
        }
        return LEs.get(i);
    }

    public double[] getUE(final int i, final double g, final double bandSize) {
        if (lastERPWindowComputed[i] != bandSize) {
            LEs.set(i, new double[train.get(i).length()]);
            UEs.set(i, new double[train.get(i).length()]);
            computeLEandUE(i, g, bandSize);
        }
        return UEs.get(i);
    }

    public void computeLEandUE(final int i, final double g, final double bandSize) {
        lbERPcomputer.fillUL(train.get(i).data[0], g, bandSize, UEs.get(i), LEs.get(i));
        this.lastERPWindowComputed[i] = bandSize;
    }

    public double[] getLE(final int i, final int delta, final double epsilon) {
        if (lastLCSSWindowComputed[i] != delta) {
            LEs.set(i, new double[train.get(i).length()]);
            UEs.set(i, new double[train.get(i).length()]);
            computeLEandUE(i, delta, epsilon);
        }
        return LEs.get(i);
    }

    public double[] getUE(final int i, final int delta, final double epsilon) {
        if (lastLCSSWindowComputed[i] != delta) {
            LEs.set(i, new double[train.get(i).length()]);
            UEs.set(i, new double[train.get(i).length()]);
            computeLEandUE(i, delta, epsilon);
        }
        return UEs.get(i);
    }

    public void computeLEandUE(final int i, final int delta, final double epsilon) {
        lbLCSScomputer.fillUL(train.get(i), epsilon, delta, UEs.get(i), LEs.get(i));
        this.lastLCSSWindowComputed[i] = delta;
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
        lbDistances = new double[train.size()][train.size()];
    }

    public void setLbDistances(double lb, int qIndex, int cIndex) {
        lbDistances[qIndex][cIndex] = lb;
    }

    public double getLbDistances(int qIndex, int cIndex) {
        return lbDistances[qIndex][cIndex];
    }

    public boolean lbDistanceExist(int qIndex, int cIndex) {
        return lbDistances[qIndex][cIndex] != 0;
    }
}
