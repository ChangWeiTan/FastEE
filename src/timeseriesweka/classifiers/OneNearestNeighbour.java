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
package timeseriesweka.classifiers;

import timeseriesweka.elasticDistances.LazyAssessNN;
import timeseriesweka.fastWWS.PotentialNN;
import timeseriesweka.fastWWS.SequenceStatsCache;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.text.DecimalFormat;
import java.util.*;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 * <p>
 * Superclass for NN classifiers
 */
public abstract class OneNearestNeighbour extends AbstractClassifier {
    /*------------------------------------------------------------------------------------------------------------------
        Variables for FastWWS
     -----------------------------------------------------------------------------------------------------------------*/
    static final int nParams = 100;
    double queryMax, queryMin;
    int approxSamples = 10;
    private int trainResample = 0;
    protected Instances train;
    double[] U, L;
    protected String datasetName;
    int maxWindow;
    PotentialNN[][] nns;
    private ArrayList<Neighbour>[][] nnOrderings = null;
    private ArrayList<Neighbour> tmp = null;
    int[][][] classCounts;
    HashMap<String, ArrayList<Instance>> classedData;
    HashMap<String, ArrayList<Integer>> classedDataIndices;
    String[] classMap;
    boolean[] indexToTreat;
    protected int[] index;
    int[] reverseIndex;
    SequenceStatsCache trainCache;
    String classifierIdentifier;
    boolean allowLoocv = true;
    boolean allowFastWWS = false;
    boolean allowApprox = false;
    boolean singleParamCv = false;
    private double cvTime;
    ArrayList<LazyAssessNN>[][] unprocessed;
    PriorityQueue<LazyAssessNN>[][] queues;
    LazyAssessNN[][] nnsCrossLine;
    double[] UBs;
    static double timePrev;
    double timeLimit = 3.6e12;
    int instanceLimit = 10;

    private int bsfParamId;
    private DecimalFormat df = new DecimalFormat("##.###");
    private double confidence;

    protected OneNearestNeighbour() {
    }

    public abstract double distance(Instance first, Instance second);

    public double distance(Instance[] first, Instance[] second) {
        double sum = 0;
        double thisDist;
        for (int d = 0; d < first.length; d++) {
            thisDist = this.distance(first[d], second[d]);
            sum += thisDist;
        }

        return sum;
    }

    public abstract void setParamsFromParamId(Instances train, int paramId);

    public void buildClassifier(Instances train) throws Exception {
        this.train = train;
    }

    public void buildClassifier(Instances train, SequenceStatsCache cache) {
        this.train = train;
        this.trainCache = cache;
    }

    public void buildClassifier() {
        this.train = null;
    }

    public abstract double classifyWithLowerBound(Instance instance);

    @Override
    public double classifyInstance(Instance instance) {
        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.train.numClasses()];

        double thisDist;

        for (int j = 0; j < this.train.numInstances(); j++) {
            Instance i = train.instance(j);
            thisDist = distance(instance, i);
            if (thisDist < bsfDistance) {
                bsfDistance = thisDist;
                classCounts = new int[train.numClasses()];
                classCounts[(int) i.classValue()]++;
            } else if (thisDist == bsfDistance) {
                classCounts[(int) i.classValue()]++;
            }
        }

        double bsfClass = -1;
        double bsfCount = -1;
        for (int c = 0; c < classCounts.length; c++) {
            if (classCounts[c] > bsfCount) {
                bsfCount = classCounts[c];
                bsfClass = c;
            }
        }

        return bsfClass;
    }

    public double accuracy(Instances test, SequenceStatsCache testCache) {
        double a = 0;
        int size = test.numInstances();
        Instance d;
        double predictedClass, trueClass;
        for (int i = 0; i < size; i++) {
            d = test.instance(i);
            queryMax = testCache.getMax(i);
            queryMin = testCache.getMin(i);
            predictedClass = classifyInstance(d);
            trueClass = d.classValue();
            if (trueClass == predictedClass)
                a++;
        }
        return a / size;
    }

    public double accuracyWithLowerBound(Instances test, SequenceStatsCache testCache) {
        double a = 0;
        int size = test.numInstances();
        Instance d;
        double predictedClass, trueClass;
        for (int i = 0; i < size; i++) {
            d = test.instance(i);
            queryMax = testCache.getMax(i);
            queryMin = testCache.getMin(i);
            predictedClass = classifyWithLowerBound(d);
            trueClass = d.classValue();
            if (trueClass == predictedClass)
                a++;
        }
        return a / size;
    }

    String getClassifierIdentifier() {
        return classifierIdentifier;
    }

    public void setClassifierIdentifier(String classifierIdentifier) {
        this.classifierIdentifier = classifierIdentifier;
    }

    public void setFastWWS(boolean flag) {
        this.allowFastWWS = flag;
    }

    void setApprox(boolean flag) {
        this.allowApprox = flag;
    }

    public double[] loocv(Instances train) throws Exception {
        long start = System.nanoTime();
        double[] accAndPreds;

        bsfParamId = -1;
        double bsfAcc = -1;
        double[] bsfaccAndPreds = null;

        System.out.print("[1-NN] LOOCV for " + this.toString() + ", param ");
        for (int paramId = 0; paramId < nParams; paramId++) {
            System.out.print(".");
            accAndPreds = loocvAccAndPreds(train, paramId);
            if (accAndPreds[0] > bsfAcc) {
                bsfParamId = paramId;
                bsfAcc = accAndPreds[0];
                bsfaccAndPreds = accAndPreds;
            }
            if (!this.allowLoocv) {
                paramId = nParams + 1;
            }
        }
        long end = System.nanoTime();
        cvTime = 1.0 * (end - start) / 1e9;

        this.buildClassifier(train);
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, bsfParamId);
        } else {
            this.setParamsFromParamId(train, 100);
        }
        System.out.println(String.format(", bsfParamId %d - %s, bsfAcc %.5f, took %.5f s", bsfParamId,
                getParamInformationString(), bsfAcc, cvTime));

        return bsfaccAndPreds;
    }

    public double[] loocvWithLowerBound(Instances train) throws Exception {
        long start = System.nanoTime();
        double[] accAndPreds;

        bsfParamId = -1;
        double bsfAcc = -1;
        double[] bsfaccAndPreds = null;

        System.out.print("[1-NN] LOOCV with Lower Bound for " + this.toString() + ", param ");
        for (int paramId = 0; paramId < nParams; paramId++) {
            System.out.print(".");
            accAndPreds = loocvWithLowerBound(train, paramId);
            if (accAndPreds[0] > bsfAcc) {
                bsfParamId = paramId;
                bsfAcc = accAndPreds[0];
                bsfaccAndPreds = accAndPreds;
            }
            if (!this.allowLoocv) {
                paramId = nParams + 1;
            }
        }
        long end = System.nanoTime();
        cvTime = 1.0 * (end - start) / 1e9;

        this.buildClassifier(train);
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, bsfParamId);
        } else {
            this.setParamsFromParamId(train, 100);
        }
        System.out.println(String.format(", bsfParamId %d - %s, bsfAcc %.5f, took %.5f s", bsfParamId,
                getParamInformationString(), bsfAcc, cvTime));

        return bsfaccAndPreds;
    }

    private double[] loocvAccAndPreds(Instances train, int paramId) throws Exception {
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, paramId);
        } else {
            this.setParamsFromParamId(train, 100);
        }

        Instances trainLoocv;
        Instance testLoocv;

        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[train.numInstances() + 1];
        for (int i = 0; i < train.numInstances(); i++) {
            trainLoocv = new Instances(train);
            testLoocv = trainLoocv.remove(i);
            actual = testLoocv.classValue();
            this.buildClassifier(trainLoocv);
            pred = this.classifyInstance(testLoocv);
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = (double) correct / train.numInstances();

        return accAndPreds;
    }

    private double[] loocvWithLowerBound(Instances train, int paramId) throws Exception {
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, paramId);
        } else {
            this.setParamsFromParamId(train, 100);
        }

        Instances trainLoocv;
        Instance testLoocv;

        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[train.numInstances() + 1];
        for (int i = 0; i < train.numInstances(); i++) {
            queryMax = trainCache.getMax(i);
            queryMin = trainCache.getMin(i);
            trainLoocv = new Instances(train);
            testLoocv = trainLoocv.remove(i);
            actual = testLoocv.classValue();
            this.buildClassifier(trainLoocv);
            pred = this.classifyWithLowerBound(testLoocv);
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = (double) correct / train.numInstances();

        return accAndPreds;
    }

    public double[] loocvEstimate(Instances train, double timeLimit, int instanceLimit) throws Exception {
        long start = System.nanoTime();
        boolean OVERTIME = false;
        Instances trainLoocv;
        Instance testLoocv;

        int[] correct = new int[nParams];
        double pred, actual;

        double[][] accAndPreds = new double[nParams][train.numInstances() + 1];
        int i;
        System.out.print(String.format("[1-NN] Estimating LOOCV for %s with %d instances and %.3e s", this.toString(), instanceLimit, timeLimit));
        for (i = 0; i < train.numInstances(); i++) {
            trainLoocv = new Instances(train);
            testLoocv = trainLoocv.remove(i);
            actual = testLoocv.classValue();
            this.buildClassifier(trainLoocv);
            for (int paramId = 0; paramId < nParams; paramId++) {
                if (this.allowLoocv) {
                    this.setParamsFromParamId(train, paramId);
                } else {
                    this.setParamsFromParamId(train, 100);
                }
                pred = this.classifyInstance(testLoocv);
                if (pred == actual) {
                    correct[paramId]++;
                }
                accAndPreds[paramId][i + 1] = pred;
                if (!this.allowLoocv) {
                    break;
                }
            }
            if ((System.nanoTime() - start) >= timeLimit && i >= instanceLimit) {
                System.out.println("[1-NN] Overtime --> kill");
                OVERTIME = true;
                break;
            }
        }
        bsfParamId = -1;
        double bsfAcc = -1;
        double[] bsfaccAndPreds = null;

        for (int paramId = 0; paramId < nParams; paramId++) {
            accAndPreds[paramId][0] = (double) correct[paramId] / train.numInstances();
            if (accAndPreds[paramId][0] > bsfAcc) {
                bsfParamId = paramId;
                bsfAcc = accAndPreds[paramId][0];
                bsfaccAndPreds = accAndPreds[paramId];
            }
        }
        long end = System.nanoTime();
        cvTime = 1.0 * (end - start) / 1e9;
        if (OVERTIME) {
            cvTime *= train.numInstances() / (i + 1);
        }

        this.buildClassifier(train);
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, bsfParamId);
        } else {
            this.setParamsFromParamId(train, 100);
        }
        System.out.println(String.format(", bsfParamId %d - %s, bsfAcc %.5f, took %.5f s", bsfParamId,
                getParamInformationString(), bsfAcc, cvTime));

        return bsfaccAndPreds;
    }

    public double[] loocvEstimateWithLowerBound(Instances train, double timeLimit, int instanceLimit) throws Exception {
        long start = System.nanoTime();
        boolean OVERTIME = false;
        Instances trainLoocv;
        Instance testLoocv;

        int[] correct = new int[nParams];
        double pred, actual;

        double[][] accAndPreds = new double[nParams][train.numInstances() + 1];
        int i;
        System.out.print(String.format("[1-NN] Estimating LOOCV with Lower Bound for %s with %d instances and %.3e s", this.toString(), instanceLimit, timeLimit));
        for (i = 0; i < train.numInstances(); i++) {
            queryMax = trainCache.getMax(i);
            queryMin = trainCache.getMin(i);
            trainLoocv = new Instances(train);
            testLoocv = trainLoocv.remove(i);
            actual = testLoocv.classValue();
            this.buildClassifier(trainLoocv);
            for (int paramId = 0; paramId < nParams; paramId++) {
                if (this.allowLoocv) {
                    this.setParamsFromParamId(train, paramId);
                } else {
                    this.setParamsFromParamId(train, 100);
                }
                pred = this.classifyWithLowerBound(testLoocv);
                if (pred == actual) {
                    correct[paramId]++;
                }
                accAndPreds[paramId][i + 1] = pred;
                if (!this.allowLoocv) {
                    break;
                }
            }
            if ((System.nanoTime() - start) >= timeLimit && i >= instanceLimit) {
                System.out.println("[1-NN] Overtime --> kill");
                OVERTIME = true;
                break;
            }
        }
        bsfParamId = -1;
        double bsfAcc = -1;
        double[] bsfaccAndPreds = null;
        for (int paramId = 0; paramId < nParams; paramId++) {
            accAndPreds[paramId][0] = (double) correct[paramId] / train.numInstances();
            if (accAndPreds[paramId][0] > bsfAcc) {
                bsfParamId = paramId;
                bsfAcc = accAndPreds[paramId][0];
                bsfaccAndPreds = accAndPreds[paramId];
            }
        }
        long end = System.nanoTime();
        cvTime = 1.0 * (end - start) / 1e9;
        if (OVERTIME) {
            cvTime *= train.numInstances() / (i + 1);
        }

        this.buildClassifier(train);
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, bsfParamId);
        } else {
            this.setParamsFromParamId(train, 100);
        }
        System.out.println(String.format(", bsfParamId %d - %s, bsfAcc %.5f, took %.5f s", bsfParamId,
                getParamInformationString(), bsfAcc, cvTime));

        return bsfaccAndPreds;
    }

    public abstract void initFastWWS(Instances train, SequenceStatsCache cache);

    public abstract void initFastWWS(Instances train, SequenceStatsCache cache, int n);

    public abstract int initFastWWSEstimate(Instances train, SequenceStatsCache cache, long start, double timeLimit, int instanceLimit);

    public abstract void initFastWWSApproximate(Instances train, SequenceStatsCache cache, int nSamples);

    public double[] fastWWS(Instances train) throws Exception {
        long start = System.nanoTime();
        double[] accAndPreds;
        maxWindow = train.numAttributes() - 1;

        initFastWWS(train, trainCache);           // initialise nearest neighbour table

        bsfParamId = -1;                        // best so far parameter ID
        double bsfAcc = -1;                     // best so far accuracy
        double[] bsfaccAndPreds = null;         // best so far accuracy and class predictions
        System.out.print("[1-NN] FastWWS for " + this.toString() + ", param ");
        // go through all the parameters
        for (int paramId = 0; paramId < nParams; paramId++) {
            System.out.print(".");
            accAndPreds = fastWWSAccAndPred(train, paramId, train.numInstances());
            if (accAndPreds[0] > bsfAcc) {
                bsfAcc = accAndPreds[0];
                bsfParamId = paramId;
                bsfaccAndPreds = accAndPreds;
            }
            if (!this.allowLoocv) {
                paramId = nParams + 1;
            }
        }
        long end = System.nanoTime();
        cvTime = 1.0 * (end - start) / 1e9;

        // reset the classifier with the best parameters
        this.buildClassifier(train);
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, bsfParamId);
        } else {
            this.setParamsFromParamId(train, 100);
        }
        System.out.println(", bsfParamId " + bsfParamId + " - " + getParamInformationString() +
                ", bsfAcc " + df.format(bsfAcc) + ", took " + cvTime + "s");

        return bsfaccAndPreds;
    }

    public double[] fastWWS(Instances train, int n) throws Exception {
        long start = System.nanoTime();
        double[] accAndPreds;
        maxWindow = train.numAttributes() - 1;
        train.randomize(new Random(trainResample));
        initFastWWS(train, trainCache, n);           // initialise nearest neighbour table

        bsfParamId = -1;                        // best so far parameter ID
        double bsfAcc = -1;                     // best so far accuracy
        double[] bsfaccAndPreds = null;         // best so far accuracy and class predictions
        System.out.print("[1-NN] FastWWS for " + this.toString() + ", param ");
        // go through all the parameters
        for (int paramId = 0; paramId < nParams; paramId++) {
            System.out.print(".");
            accAndPreds = fastWWSAccAndPred(train, paramId, n);
            if (accAndPreds[0] > bsfAcc) {
                bsfAcc = accAndPreds[0];
                bsfParamId = paramId;
                bsfaccAndPreds = accAndPreds;
            }
            if (!this.allowLoocv) {
                paramId = nParams + 1;
            }
        }
        long end = System.nanoTime();
        cvTime = 1.0 * (end - start) / 1e9;

        // reset the classifier with the best parameters
        this.buildClassifier(train);
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, bsfParamId);
        } else {
            this.setParamsFromParamId(train, 100);
        }
        System.out.println(", bsfParamId " + bsfParamId + " - " + getParamInformationString() +
                ", bsfAcc " + df.format(bsfAcc) + ", took " + cvTime + "s");

        return bsfaccAndPreds;
    }

    public double[] fastWWSAccAndPred(Instances train, int paramId, int n) {
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, paramId);
        } else {
            this.setParamsFromParamId(train, 100);
        }
        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[n + 1];
        for (int i = 0; i < n; i++) {
            actual = train.instance(i).classValue();
            pred = train.instance(nns[paramId][i].index).classValue();
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = (double) correct / n;

        return accAndPreds;
    }

    public double[] fastWWSEstimate(Instances train, double timeLimit, int instanceLimit) throws Exception {
        long start = System.nanoTime();
        double[] accAndPreds;
        maxWindow = train.numAttributes() - 1;

        int completed = initFastWWSEstimate(train, trainCache, start, timeLimit, instanceLimit);           // initialise nearest neighbour table

        bsfParamId = -1;                        // best so far parameter ID
        double bsfAcc = -1;                     // best so far accuracy
        double[] bsfaccAndPreds = null;         // best so far accuracy and class predictions
        System.out.print("[1-NN] FastWWS for " + this.toString() + ", param ");
        // go through all the parameters
        for (int paramId = 0; paramId < nParams; paramId++) {
            System.out.print(".");
            accAndPreds = fastWWSAccAndPredEstimate(train, paramId);
            if (accAndPreds[0] > bsfAcc) {
                bsfAcc = accAndPreds[0];
                bsfParamId = paramId;
                bsfaccAndPreds = accAndPreds;
            }
            if (!this.allowLoocv) {
                paramId = nParams + 1;
            }
        }
        long end = System.nanoTime();
        cvTime += 1.0 * (end - start) / 1e9;
        if (completed != train.numInstances()) {
            double x1 = 10;
            double a = (x1 * cvTime - completed * timePrev) / (x1 * completed * completed - x1 * x1 * completed);
            double b = timePrev / x1 - (x1 * cvTime - completed * timePrev) / (completed * completed - completed * x1);

            cvTime = (train.numInstances() * train.numInstances() * a + b * train.numInstances()) / 1e9;
        }

        // reset the classifier with the best parameters
        this.buildClassifier(train);
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, bsfParamId);
        }
        System.out.println(String.format(", bsfParamId %d - %s, bsfAcc %.5f, took %.5f s", bsfParamId,
                getParamInformationString(), bsfAcc, cvTime));

        return bsfaccAndPreds;
    }

    private double[] fastWWSAccAndPredEstimate(Instances train, int paramId) {
        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[train.numInstances() + 1];
        for (int i = 0; i < train.numInstances(); i++) {
            if (nns[paramId][i].index >= 0) {
                actual = train.instance(i).classValue();
                pred = train.instance(nns[paramId][i].index).classValue();
                if (pred == actual) {
                    correct++;
                }
                accAndPreds[i + 1] = pred;
            }
        }
        accAndPreds[0] = (double) correct / train.numInstances();

        return accAndPreds;
    }

    public double[] fastWWSApproximate(Instances train, int nSamples) throws Exception {
        if (this.classifierIdentifier.contains("R1")) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
        }
        train.randomize(new Random(trainResample));
        long start = System.nanoTime();
        double[] accAndPreds;
        maxWindow = train.numAttributes() - 1;

        initFastWWSApproximate(train, trainCache, nSamples);           // initialise nearest neighbour table

        bsfParamId = -1;                        // best so far parameter ID
        double bsfAcc = -1;                     // best so far accuracy
        double[] bsfaccAndPreds = null;         // best so far accuracy and class predictions
        System.out.print("[1-NN] FastWWS for " + this.toString() + ", param ");
        // go through all the parameters
        for (int paramId = 0; paramId < nParams; paramId++) {
            System.out.print(".");
            accAndPreds = fastWWSAccAndPredApproximate(train, paramId, nSamples);
            if (accAndPreds[0] > bsfAcc) {
                bsfAcc = accAndPreds[0];
                bsfParamId = paramId;
                bsfaccAndPreds = accAndPreds;
            }
            if (!this.allowLoocv) {
                paramId = nParams + 1;
            }
        }
        long end = System.nanoTime();
        cvTime = 1.0 * (end - start) / 1e9;

        // reset the classifier with the best parameters
        this.buildClassifier(train);
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, bsfParamId);
        }
        System.out.println(", bsfParamId " + bsfParamId + " - " + getParamInformationString() +
                ", bsfAcc " + df.format(bsfAcc) + ", took " + cvTime + "s");

        return bsfaccAndPreds;
    }

    public double[] fastWWSAccAndPredApproximate(Instances train, int paramId, int nSamples) {
        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[nSamples + 1];
        for (int i = 0; i < nSamples; i++) {
            actual = train.instance(i).classValue();
            pred = train.instance(nns[paramId][i].index).classValue();
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = (double) correct / nSamples;

        return accAndPreds;
    }

    public abstract String getParamInformationString();

    @Override
    public String toString() {
        return this.classifierIdentifier;
    }

    public void setTrainResample(int resample) {
        trainResample = resample;
    }

    public void setApproxSamples(int nSamples) {
        approxSamples = nSamples;
    }

    public void setTrain(Instances train) {
        this.train = train;
        U = new double[train.numAttributes() - 1];
        L = new double[train.numAttributes() - 1];
    }

    public void setTrainCache(SequenceStatsCache cache) {
        this.trainCache = cache;
    }

    void setQueryMinMax(double max, double min) {
        queryMax = max;
        queryMin = min;
    }

    public int getBsfParamId() {
        return bsfParamId;
    }

    public void setBsfParamId(int a) {
        bsfParamId = a;
    }

    public double getCvTime() {
        return cvTime;
    }

    public PotentialNN[][] getNns() {
        return nns;
    }

    public void turnOffCV() {
        this.allowLoocv = false;
    }

    public void turnOnCV() {
        this.allowLoocv = true;
    }

    public void turnOffFastWWS() {
        this.allowFastWWS = false;
    }

    public void turnOnFastWWS() {
        this.allowFastWWS = true;
    }

    public class Neighbour implements Comparable<Neighbour> {
        int indexInTrain;
        double distanceToQuery;

        Neighbour(int i, double d) {
            indexInTrain = i;
            distanceToQuery = d;
        }

        @Override
        public String toString() {
            return indexInTrain + " - " + distanceToQuery;
        }

        @Override
        public int compareTo(Neighbour o) {
            return Double.compare(this.distanceToQuery, o.distanceToQuery);
        }
    }
}
