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

import timeseriesweka.elasticDistances.LCSS;
import timeseriesweka.elasticDistances.LazyAssessNN;
import timeseriesweka.fastWWS.PotentialNN;
import timeseriesweka.fastWWS.SequenceStatsCache;
import timeseriesweka.lowerBounds.LbLcss;
import utilities.Tools;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * NN-LCSS Classifier
 */
public class LCSS1NN extends OneNearestNeighbour {
    // parameters
    private int delta;                              // delta value
    private double epsilon;                         // epsilon (threshold)

    private double[] epsilons;                      // set of epsilon values
    private int[] deltas;                           // set of delta values
    private boolean epsilonsAndDeltasRefreshed;     // indicator if we refresh the params

    public LCSS1NN(int delta, double epsilon) {
        this.delta = delta;
        this.epsilon = epsilon;
        epsilonsAndDeltasRefreshed = false;
        this.classifierIdentifier = "LCSS_1NN";
        this.allowLoocv = false;
    }

    public LCSS1NN() {
        this.delta = 3;
        this.epsilon = 1;
        epsilonsAndDeltasRefreshed = false;
        this.classifierIdentifier = "LCSS_1NN";
    }

    public void setDelta(int delta) {
        this.delta = delta;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public void initFastWWS(Instances train, SequenceStatsCache cache) {
        if (train.numInstances() < 2) {
            System.err.println("Set is to small: " + train.numInstances() + " sequence. At least 2 sequences needed.");
        }

        nns = new PotentialNN[nParams][train.numInstances()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.numInstances(); ++len) {
                nns[paramId][len] = new PotentialNN();
            }
        }
        classCounts = new int[nParams][train.numInstances()][train.numClasses()];

        LazyAssessNN[] lazyAssessNNS = new LazyAssessNN[train.numInstances()];
        for (int i = 0; i < train.numInstances(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN(cache);
        }
        ArrayList<LazyAssessNN> challengers = new ArrayList<>(train.numInstances());

        // Iteration for all TS, starting with the second one (first is  reference)
        for (int current = 1; current < train.numInstances(); ++current) {
            // --- --- Get the data --- ---
            Instance sCurrent = train.instance(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                LazyAssessNN d = lazyAssessNNS[previous];
                d.setWoutKim(train.instance(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // --- --- For each, decreasing (positive) windows --- ---
            for (int paramId = nParams - 1; paramId > -1; --paramId) {

                setParamsFromParamId(train, paramId);

                // --- Get the data
                PotentialNN currPNN = nns[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have  NN for sure, but we still have to check if current is  new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        // --- Get the data
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Try to beat the previous best NN
                        double toBeat = prevNN.distance;
                        LazyAssessNN challenger = lazyAssessNNS[previous];
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatLCSS(toBeat, delta, epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(delta);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.numClasses()];
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            }
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have  NN yet.
                    // Sort the challengers so we have  better chance to organize  good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN challenger : challengers) {
                        // --- Get the data
                        int previous = challenger.indexQuery;

                        // --- First we want to beat the current best candidate of reference:
                        double toBeat = currPNN.distance;
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatLCSS(toBeat, delta, epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(delta);
                            currPNN.set(previous, r, d, PotentialNN.Status.BC);
                            if (d < toBeat) {
                                classCounts[paramId][current] = new int[train.numClasses()];
                                classCounts[paramId][current][(int) challenger.getQuery().classValue()]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][current][(int) challenger.getQuery().classValue()]++;
                            }
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        PotentialNN prevNN = nns[paramId][previous];
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatLCSS(toBeat, delta, epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(delta);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.numClasses()];
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            }
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int tmp = paramId;
//                    nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
                    double prevEpsilon = epsilon;
                    while (tmp > 0 && paramId % 10 > 0 && prevEpsilon == epsilon && delta >= r) {
                        nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();

                        tmp--;
                        this.setParamsFromParamId(train, tmp);
                    }
                }
            }
        }
    }

    @Override
    public void initFastWWS(Instances train, SequenceStatsCache cache, int n) {
        if (n < 2) {
            System.err.println("Set is to small: " + n + " sequence. At least 2 sequences needed.");
        }

        nns = new PotentialNN[nParams][n];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < n; ++len) {
                nns[paramId][len] = new PotentialNN();
            }
        }
        classCounts = new int[nParams][n][train.numClasses()];

        LazyAssessNN[] lazyAssessNNS = new LazyAssessNN[n];
        for (int i = 0; i < n; ++i) {
            lazyAssessNNS[i] = new LazyAssessNN(cache);
        }
        ArrayList<LazyAssessNN> challengers = new ArrayList<>(train.numInstances());

        // Iteration for all TS, starting with the second one (first is  reference)
        for (int current = 1; current < n; ++current) {
            // --- --- Get the data --- ---
            Instance sCurrent = train.instance(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                LazyAssessNN d = lazyAssessNNS[previous];
                d.setWoutKim(train.instance(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // --- --- For each, decreasing (positive) windows --- ---
            for (int paramId = nParams - 1; paramId > -1; --paramId) {

                setParamsFromParamId(train, paramId);

                // --- Get the data
                PotentialNN currPNN = nns[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have  NN for sure, but we still have to check if current is  new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        // --- Get the data
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Try to beat the previous best NN
                        double toBeat = prevNN.distance;
                        LazyAssessNN challenger = lazyAssessNNS[previous];
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatLCSS(toBeat, delta, epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(delta);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.numClasses()];
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            }
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have  NN yet.
                    // Sort the challengers so we have  better chance to organize  good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN challenger : challengers) {
                        // --- Get the data
                        int previous = challenger.indexQuery;

                        // --- First we want to beat the current best candidate of reference:
                        double toBeat = currPNN.distance;
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatLCSS(toBeat, delta, epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(delta);
                            currPNN.set(previous, r, d, PotentialNN.Status.BC);
                            if (d < toBeat) {
                                classCounts[paramId][current] = new int[train.numClasses()];
                                classCounts[paramId][current][(int) challenger.getQuery().classValue()]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][current][(int) challenger.getQuery().classValue()]++;
                            }
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        PotentialNN prevNN = nns[paramId][previous];
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatLCSS(toBeat, delta, epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(delta);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.numClasses()];
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            }
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int tmp = paramId;
//                    nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
                    double prevEpsilon = epsilon;
                    while (tmp > 0 && paramId % 10 > 0 && prevEpsilon == epsilon && delta >= r) {
                        nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();

                        tmp--;
                        this.setParamsFromParamId(train, tmp);
                    }
                }
            }
        }
    }

    @Override
    public int initFastWWSEstimate(Instances train, SequenceStatsCache cache, long start, double timeLimit, int instanceLimit) {
        return 0;
    }

    @Override
    public void initFastWWSApproximate(Instances train, SequenceStatsCache cache, int nSamples) {
        if (train.numInstances() < 2) {
            System.err.println("Set is to small: " + train.numInstances() + " sequence. At least 2 sequences needed.");
        }

        // We need  N*L storing area. We favorite an access per window size.
        // For each [Window Size][sequence], we store the nearest neighbour. See above.
        nns = new PotentialNN[nParams][nSamples];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < nSamples; ++len) {
                nns[paramId][len] = new PotentialNN();
            }
        }
        classCounts = new int[nParams][nSamples][train.numClasses()];

        LazyAssessNN[] lazyAssessNNS = new LazyAssessNN[train.numInstances()];
        for (int i = 0; i < train.numInstances(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN(cache);
        }
        ArrayList<LazyAssessNN> challengers = new ArrayList<>(train.numInstances());

        // Iteration for all TS, starting with the second one (first is  reference)
        for (int current = 0; current < nSamples; ++current) {
            // --- --- Get the data --- ---
            Instance sCurrent = train.instance(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                LazyAssessNN d = lazyAssessNNS[previous];
                d.setWoutKim(train.instance(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // --- --- For each, decreasing (positive) windows --- ---
            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);

                // --- Get the data
                PotentialNN currPNN = nns[paramId][current];

                Collections.sort(challengers);
                boolean newNN = false;
                for (LazyAssessNN challenger : challengers) {
                    // --- Get the data
                    int previous = challenger.indexQuery;
                    if (previous == current) previous = challenger.indexReference;
                    if (previous == currPNN.index) continue;

                    // --- First we want to beat the current best candidate of reference:
                    double toBeat = currPNN.distance;
                    LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatLCSS(toBeat, delta, epsilon);

                    // --- Check the result
                    if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                        int r = challenger.getMinWindowValidityForFullDistance();
                        double d = challenger.getDistance(delta);
                        currPNN.set(previous, r, d, PotentialNN.Status.BC);
                        if (d < toBeat) {
                            classCounts[paramId][current] = new int[train.numClasses()];
                            classCounts[paramId][current][(int) challenger.getQuery().classValue()]++;
                        } else if (d == toBeat) {
                            classCounts[paramId][current][(int) challenger.getQuery().classValue()]++;
                        }
                        newNN = true;
                    }

                    if (previous < nSamples) {
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatLCSS(toBeat, delta, epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(delta);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.numClasses()];
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            }
                        }
                    }
                }

                if (newNN) {
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int tmp = paramId;
                    nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
                    double prevEpsilon = epsilon;
                    while (tmp > 0 && paramId % 10 > 0 && prevEpsilon == epsilon && delta >= r) {
                        nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();

                        tmp--;
                        this.setParamsFromParamId(train, tmp);
                    }
                }
            }
        }
    }

    @Override
    public double[] fastWWSAccAndPred(Instances train, int paramId, int n) {
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, paramId);
        }
        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[n + 1];
        for (int i = 0; i < n; i++) {
            actual = train.instance(i).classValue();
            pred = -1;
            double bsfCount = -1;
            for (int c = 0; c < classCounts[paramId][i].length; c++) {
                if (classCounts[paramId][i][c] > bsfCount) {
                    bsfCount = classCounts[paramId][i][c];
                    pred = c;
                }
            }
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = (double) correct / train.numInstances();

        return accAndPreds;
    }

    @Override
    public double[] fastWWSAccAndPredApproximate(Instances train, int paramId, int nSamples) {
        if (this.allowLoocv) {
            this.setParamsFromParamId(train, paramId);
        }

        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[nSamples + 1];
        for (int i = 0; i < nSamples; i++) {
            actual = train.instance(i).classValue();
            pred = -1;
            double bsfCount = -1;
            for (int c = 0; c < classCounts[paramId][i].length; c++) {
                if (classCounts[paramId][i][c] > bsfCount) {
                    bsfCount = classCounts[paramId][i][c];
                    pred = c;
                }
            }
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = (double) correct / nSamples;

        return accAndPreds;
    }

    @Override
    public double[] loocv(Instances train) throws Exception {
        if (allowApprox)
            return super.fastWWSApproximate(train, approxSamples);
        else if (allowFastWWS)
            return super.fastWWS(train);
        else
            return super.loocv(train);
    }

    @Override
    public void buildClassifier(Instances train) throws Exception {
        super.buildClassifier(train);
        epsilonsAndDeltasRefreshed = false;
    }

    /*------------------------------------------------------------------------------------------------------------------
        Distances
     -----------------------------------------------------------------------------------------------------------------*/
    public double distance(Instance first, Instance second) {
        return LCSS.distance(first, second, this.epsilon, this.delta);
    }

    public double lowerbound(Instance q, Instance c) {
        double[] U = new double[q.numAttributes() - 1];
        double[] L = new double[q.numAttributes() - 1];
        LbLcss.fillUL(q, epsilon, delta, U, L);
        return lowerbound(c, U, L);
    }

    public double lowerbound(Instance c, double[] U, double[] L) {
        return LbLcss.distance(c, U, L);
    }

    @Override
    public double classifyInstance(Instance instance) {
        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.train.numClasses()];

        double thisDist;

        for (Instance i : this.train) {
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

    @Override
    public double classifyWithLowerBound(Instance instance) {
        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.train.numClasses()];
        double[] U = new double[instance.numAttributes() - 1];
        double[] L = new double[instance.numAttributes() - 1];

        double thisDist, lbDist;
        LbLcss.fillUL(instance, epsilon, delta, U, L);

        for (Instance i : this.train) {
            lbDist = LbLcss.distance(i, U, L);
            if (lbDist <= bsfDistance) {
                thisDist = distance(instance, i);
                if (thisDist < bsfDistance) {
                    bsfDistance = thisDist;
                    classCounts = new int[train.numClasses()];
                    classCounts[(int) i.classValue()]++;
                } else if (thisDist == bsfDistance) {
                    classCounts[(int) i.classValue()]++;
                }
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

    @Override
    public void setParamsFromParamId(Instances train, int paramId) {
        if (!epsilonsAndDeltasRefreshed) {
            double stdTrain = Tools.stdv_p(train);
            double stdFloor = stdTrain * 0.2;
            epsilons = Tools.getInclusive10(stdFloor, stdTrain);
            deltas = Tools.getInclusive10(0, (train.numAttributes() - 1) / 4);
            epsilonsAndDeltasRefreshed = true;
        }
        this.delta = deltas[paramId % 10];      // changed this for FastWWS, previously deltas[paramId / 10];
        this.epsilon = epsilons[paramId / 10];  // changed this for FastWWS, previously epsilons[paramId % 10];
    }

    @Override
    public String getParamInformationString() {
        return "delta=" + this.delta + ", epsilon=" + this.epsilon;
    }

    public String getParams() {
        return this.delta + "," + this.epsilon;
    }

    public double[] getEpsilons() {
        return epsilons;
    }

    public int[] getDeltas() {
        return deltas;
    }
}
