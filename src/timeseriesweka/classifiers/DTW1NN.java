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

import timeseriesweka.elasticDistances.DTW;
import timeseriesweka.elasticDistances.DistanceResults;
import timeseriesweka.elasticDistances.LazyAssessNN;
import timeseriesweka.fastWWS.PotentialNN;
import timeseriesweka.fastWWS.SequenceStatsCache;
import timeseriesweka.lowerBounds.LbKeogh;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * NN-DTW Classifier
 */
public class DTW1NN extends OneNearestNeighbour {
    private double r;   // warping window size in terms of percentage of sequence's length
    private int w;

    public DTW1NN(double r) {
        this.allowLoocv = false;
        this.r = r;
        if (r != 1) {
            this.classifierIdentifier = "DTW_Rn_1NN";
        } else {
            this.classifierIdentifier = "DTW_R1_1NN";
        }
    }

    public DTW1NN() {
        this.r = 1;
        this.w = 1;
        this.classifierIdentifier = "DTW_R1_1NN";
    }

    public DTW1NN(Instances train) {
        this.train = train;
        U = new double[train.numAttributes() - 1];
        L = new double[train.numAttributes() - 1];
        this.r = 1;
        this.w = train.numAttributes() - 1;
        this.classifierIdentifier = "DTW_R1_1NN";
    }

    public void setWindow(double w) {
        r = w;
    }

    private int getWindowSize2(int n) {
        w = (int) (r * n);
        return w;
    }

    @Override
    public void initFastWWS(Instances train, SequenceStatsCache cache) {
        if (train.numInstances() < 2) {
            System.err.println("Set is too small: " + train.numInstances() + " sequence. At least 2 sequences needed.");
        }

        nns = new PotentialNN[nParams][train.numInstances()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.numInstances(); ++len) {
                nns[paramId][len] = new PotentialNN();
            }
        }

        LazyAssessNN[] lazyAssessNNS = new LazyAssessNN[train.numInstances()];
        for (int i = 0; i < train.numInstances(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN(cache);
        }
        ArrayList<LazyAssessNN> challengers = new ArrayList<>(train.numInstances());

        // Iteration for all TS, starting with the second one (first is the reference)
        for (int current = 1; current < train.numInstances(); ++current) {
            Instance sCurrent = train.instance(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                LazyAssessNN d = lazyAssessNNS[previous];
                d.set(train.instance(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // --- --- For each, decreasing (positive) windows --- ---
            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);
                int win = getWindowSize2(maxWindow);
                PotentialNN currPNN = nns[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        // --- Get the data
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Try to beat the previous best NN
                        double toBeat = prevNN.distance;
                        LazyAssessNN challenger = lazyAssessNNS[previous];
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatDTW(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(win);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have the NN yet.
                    // Sort the challengers so we have the better chance to organize the good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN challenger : challengers) {
                        // --- Get the data
                        int previous = challenger.indexQuery;
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatDTW(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(win);
                            currPNN.set(previous, r, d, PotentialNN.Status.BC);
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatDTW(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(win);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int winEnd = getParamIdFromWindow(r, train.numAttributes() - 1);
                    for (int tmp = paramId; tmp >= winEnd; --tmp) {
                        nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
                    }
                }
            }
        }
    }

    public void initFastWWS(Instances train, SequenceStatsCache cache, int n) {
        if (n < 2) {
            System.err.println("Set is too small: " + n + " sequence. At least 2 sequences needed.");
        }

        nns = new PotentialNN[nParams][n];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < n; ++len) {
                nns[paramId][len] = new PotentialNN();
            }
        }

        LazyAssessNN[] lazyAssessNNS = new LazyAssessNN[n];
        for (int i = 0; i < n; ++i) {
            lazyAssessNNS[i] = new LazyAssessNN(cache);
        }
        ArrayList<LazyAssessNN> challengers = new ArrayList<>(n);

        // Iteration for all TS, starting with the second one (first is the reference)
        for (int current = 1; current < n; ++current) {
            Instance sCurrent = train.instance(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                LazyAssessNN d = lazyAssessNNS[previous];
                d.set(train.instance(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // --- --- For each, decreasing (positive) windows --- ---
            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);
                int win = getWindowSize2(maxWindow);
                PotentialNN currPNN = nns[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        // --- Get the data
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Try to beat the previous best NN
                        double toBeat = prevNN.distance;
                        LazyAssessNN challenger = lazyAssessNNS[previous];
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatDTW(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(win);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have the NN yet.
                    // Sort the challengers so we have the better chance to organize the good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN challenger : challengers) {
                        // --- Get the data
                        int previous = challenger.indexQuery;
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatDTW(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(win);
                            currPNN.set(previous, r, d, PotentialNN.Status.BC);
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatDTW(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(win);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int winEnd = getParamIdFromWindow(r, train.numAttributes() - 1);
                    for (int tmp = paramId; tmp >= winEnd; --tmp) {
                        nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
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
            System.err.println("Set is too small: " + train.numInstances() + " sequence. At least 2 sequences needed.");
        }

        nns = new PotentialNN[nParams][nSamples];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < nSamples; ++len) {
                nns[paramId][len] = new PotentialNN();
            }
        }

        LazyAssessNN[] lazyAssessNNS = new LazyAssessNN[train.numInstances()];
        for (int i = 0; i < train.numInstances(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN(cache);
        }
        ArrayList<LazyAssessNN> challengers = new ArrayList<>(train.numInstances());

        // Iteration for all TS, starting with the second one (first is the reference)
        for (int current = 0; current < nSamples; ++current) {
            Instance sCurrent = train.instance(current);

            challengers.clear();
            for (int previous = 0; previous < train.numInstances(); ++previous) {
                if (previous == current) continue;

                LazyAssessNN d = lazyAssessNNS[previous];
                d.set(train.instance(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // --- --- For each, decreasing (positive) windows --- ---
            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);
                int win = getWindowSize2(maxWindow);
                PotentialNN currPNN = nns[paramId][current];

                Collections.sort(challengers);
                boolean newNN = false;
                for (LazyAssessNN challenger : challengers) {
                    // --- Get the data
                    int previous = challenger.indexQuery;
                    if (previous == current) previous = challenger.indexReference;
                    if (previous == currPNN.index) continue;

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatDTW(toBeat, win);
                    // --- Check the result
                    if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                        int r = challenger.getMinWindowValidityForFullDistance();
                        double d = challenger.getDistance(win);
                        currPNN.set(previous, r, d, PotentialNN.Status.BC);
                        newNN = true;
                    }

                    if (previous < nSamples) {
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatDTW(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(win);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                        }
                    }
                }
                if (newNN) {
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int winEnd = getParamIdFromWindow(r, train.numAttributes() - 1);
                    for (int tmp = paramId; tmp >= winEnd; --tmp) {
                        nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
                    }
                }
            }
        }
    }

    @Override
    public double[] loocv(Instances train) throws Exception {
        if (this.allowLoocv && this.classifierIdentifier.contains("R1")) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
            if (allowApprox)
                return super.fastWWSApproximate(train, approxSamples);
            else if (allowFastWWS)
                return super.fastWWS(train);
            else
                return super.loocv(train);
        } else if (this.allowLoocv) {
            if (allowApprox)
                return super.fastWWSApproximate(train, approxSamples);
            else if (allowFastWWS)
                return super.fastWWS(train);
            else
                return super.loocv(train);
        }
        return super.loocv(train);
    }

    /*------------------------------------------------------------------------------------------------------------------
        Distances
     -----------------------------------------------------------------------------------------------------------------*/
    public final double distance(Instance first, Instance second, double cutOffValue) {
        return DTW.distance(first, second, w, cutOffValue);
    }

    public final double distance(Instance first, Instance second) {
        return DTW.distance(first, second, w);
    }

    private DistanceResults distanceExt(Instance first, Instance second) {
        return DTW.distanceExt(first, second, w);
    }

    private DistanceResults distanceExt(Instance first, Instance second, double cutOffValue) {
        return DTW.distanceExt(first, second, w, cutOffValue);
    }

    public final double lowerbound(Instance q, Instance c) {
        double[] U = new double[q.numAttributes() - 1];
        double[] L = new double[q.numAttributes() - 1];
        LbKeogh.fillUL(q, w, U, L);
        return lowerbound(c, U, L);
    }

    public final double lowerbound(Instance c, double[] U, double[] L) {
        return LbKeogh.distance(c, U, L);
    }

    @Override
    public double classifyInstance(Instance instance) {
        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.train.numClasses()];

        double thisDist;

        for (int j = 0; j < this.train.numInstances(); j++) {
            Instance i = train.instance(j);
            thisDist = distance(instance, i, bsfDistance);
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
        LbKeogh.fillUL(instance, w, U, L);

        for (int j = 0; j < this.train.numInstances(); j++) {
            Instance i = train.instance(j);
            lbDist = LbKeogh.distance(i, U, L, bsfDistance);
            if (lbDist < bsfDistance) {
                thisDist = distance(instance, i, bsfDistance);
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
        if (this.classifierIdentifier.contains("R1")) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
        }
        this.r = (double) paramId / 100;
        this.w = getWindowSize2(train.numAttributes() - 1);
    }

    @Override
    public String getParamInformationString() {
        return "r=" + this.r;
    }

    public String getParams() {
        return this.r + "";
    }

    private int getParamIdFromWindow(int w, int n) {
        double r = 1.0 * w / n;
        return (int) Math.ceil(r * 100);
    }
}
