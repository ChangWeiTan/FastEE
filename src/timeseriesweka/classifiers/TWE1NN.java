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

import development.DataSets;
import timeseriesweka.elasticDistances.LazyAssessNN;
import timeseriesweka.elasticDistances.TWED;
import timeseriesweka.fastWWS.PotentialNN;
import timeseriesweka.fastWWS.SequenceStatsCache;
import timeseriesweka.lowerBounds.LbTwed;
import utilities.ClassifierTools;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * NN-TWED Classifier
 */
public class TWE1NN extends OneNearestNeighbour {
    // parameters
    private static double[] twe_nuParams = TWED.twe_nuParams;       // set of nu values
    private static double[] twe_lamdaParams = TWED.twe_lamdaParams; // set of lambda values
    private double nu;                                          // nu value (stiffness)
    private double lambda;                                      // lambda value

    public TWE1NN(double nu, double lambda) {
        this.nu = nu;
        this.lambda = lambda;
        this.classifierIdentifier = "TWE_1NN";
        this.allowLoocv = false;
    }

    public TWE1NN() {
        this.nu = 0.005;
        this.lambda = 0.5;
        this.classifierIdentifier = "TWE_1NN";
    }

    public void setNu(double nu) {
        this.nu = nu;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    @Override
    public void initFastWWS(Instances train, SequenceStatsCache cache) {
        if (train.numInstances() < 2) {
            System.err.println("Set is to small: " + train.numInstances() + " sequence. At least 2 sequences needed.");
        }

        // We need N*L storing area. We favorite an access per window size.
        // For each [Window Size][sequence], we store the nearest neighbour. See above.
        nns = new PotentialNN[nParams][train.numInstances()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.numInstances(); ++len) {
                nns[paramId][len] = new PotentialNN();
            }
        }

        // Vector of lazyAssessNNS lbKeogh, propagating bound info "horizontally"
        LazyAssessNN[] lazyAssessNNS = new LazyAssessNN[train.numInstances()];
        for (int i = 0; i < train.numInstances(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN(cache);
        }
        // "Challengers" that compete with each other to be the NN of query
        ArrayList<LazyAssessNN> challengers = new ArrayList<>(train.numInstances());

        // Iteration for all TS, starting with the second one (first is reference)
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
            for (int paramId = 0; paramId < nParams; ++paramId) {

                setParamsFromParamId(train, paramId);

                // --- Get the data
                PotentialNN currPNN = nns[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have NN for sure, but we still have to check if current is new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        // --- Get the data
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Try to beat the previous best NN
                        double toBeat = prevNN.distance;
                        LazyAssessNN challenger = lazyAssessNNS[previous];
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatTWED(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            prevNN.set(current, d, PotentialNN.Status.NN);
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have NN yet.
                    // Sort the challengers so we have better chance to organize good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN challenger : challengers) {
                        // --- Get the data
                        int previous = challenger.indexQuery;
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatTWED(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            currPNN.set(previous, d, PotentialNN.Status.BC);
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatTWED(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            prevNN.set(current, d, PotentialNN.Status.NN);
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    nns[paramId][current].set(index, d, PotentialNN.Status.NN);
                }
            }
        }
    }

    @Override
    public void initFastWWS(Instances train, SequenceStatsCache cache, int n) {
        if (n < 2) {
            System.err.println("Set is to small: " + n + " sequence. At least 2 sequences needed.");
        }

        // We need N*L storing area. We favorite an access per window size.
        // For each [Window Size][sequence], we store the nearest neighbour. See above.
        nns = new PotentialNN[nParams][n];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < n; ++len) {
                nns[paramId][len] = new PotentialNN();
            }
        }

        // Vector of lazyAssessNNS lbKeogh, propagating bound info "horizontally"
        LazyAssessNN[] lazyAssessNNS = new LazyAssessNN[n];
        for (int i = 0; i < n; ++i) {
            lazyAssessNNS[i] = new LazyAssessNN(cache);
        }
        // "Challengers" that compete with each other to be the NN of query
        ArrayList<LazyAssessNN> challengers = new ArrayList<>(train.numInstances());

        // Iteration for all TS, starting with the second one (first is reference)
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
            for (int paramId = 0; paramId < nParams; ++paramId) {

                setParamsFromParamId(train, paramId);

                // --- Get the data
                PotentialNN currPNN = nns[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have NN for sure, but we still have to check if current is new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        // --- Get the data
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Try to beat the previous best NN
                        double toBeat = prevNN.distance;
                        LazyAssessNN challenger = lazyAssessNNS[previous];
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatTWED(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            prevNN.set(current, d, PotentialNN.Status.NN);
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have NN yet.
                    // Sort the challengers so we have better chance to organize good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN challenger : challengers) {
                        // --- Get the data
                        int previous = challenger.indexQuery;
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatTWED(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            currPNN.set(previous, d, PotentialNN.Status.BC);
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatTWED(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            prevNN.set(current, d, PotentialNN.Status.NN);
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    nns[paramId][current].set(index, d, PotentialNN.Status.NN);
                }
            }
        }
    }

    @Override
    public int initFastWWSEstimate(Instances train, SequenceStatsCache cache, long start, double timeLimit, int instanceLimit) {
        if (train.numInstances() < 2) {
            System.err.println("Set is to small: " + train.numInstances() + " sequence. At least 2 sequences needed.");
        }

        // We need N*L storing area. We favorite an access per window size.
        // For each [Window Size][sequence], we store the nearest neighbour. See above.
        nns = new PotentialNN[nParams][train.numInstances()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.numInstances(); ++len) {
                nns[paramId][len] = new PotentialNN();
            }
        }

        // Vector of lazyAssessNNS lbKeogh, propagating bound info "horizontally"
        LazyAssessNN[] lazyAssessNNS = new LazyAssessNN[train.numInstances()];
        for (int i = 0; i < train.numInstances(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN(cache);
        }
        // "Challengers" that compete with each other to be the NN of query
        ArrayList<LazyAssessNN> challengers = new ArrayList<>(train.numInstances());

        int current;
        // Iteration for all TS, starting with the second one (first is reference)
        for (current = 1; current < train.numInstances(); ++current) {
            // --- --- Get the data --- ---
            Instance sCurrent = train.instance(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                LazyAssessNN d = lazyAssessNNS[previous];
                d.setForTWED(train.instance(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // --- --- For each, decreasing (positive) windows --- ---
            for (int paramId = 0; paramId < nParams; ++paramId) {

                setParamsFromParamId(train, paramId);

                // --- Get the data
                PotentialNN currPNN = nns[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have NN for sure, but we still have to check if current is new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        // --- Get the data
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Try to beat the previous best NN
                        double toBeat = prevNN.distance;
                        LazyAssessNN challenger = lazyAssessNNS[previous];
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatTWED(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            prevNN.set(current, d, PotentialNN.Status.NN);
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have NN yet.
                    // Sort the challengers so we have better chance to organize good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN challenger : challengers) {
                        // --- Get the data
                        int previous = challenger.indexQuery;
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatTWED(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            currPNN.set(previous, d, PotentialNN.Status.BC);
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatTWED(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            prevNN.set(current, d, PotentialNN.Status.NN);
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    nns[paramId][current].set(index, d, PotentialNN.Status.NN);
                }
            }
            double timeNow = (System.nanoTime() - start);
            if (timeNow >= timeLimit && current >= instanceLimit) {
                System.out.println("Overtime");
                break;
            }
            if (current == 10) timePrev = timeNow;
        }
        return current;
    }

    @Override
    public void initFastWWSApproximate(Instances train, SequenceStatsCache cache, int nSamples) {
        if (train.numInstances() < 2) {
            System.err.println("Set is to small: " + train.numInstances() + " sequence. At least 2 sequences needed.");
        }

        // We need N*L storing area. We favorite an access per window size.
        // For each [Window Size][sequence], we store the nearest neighbour. See above.
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

        // Iteration for all TS, starting with the second one (first is reference)
        for (int current = 0; current < nSamples; ++current) {
            // --- --- Get the data --- ---
            Instance sCurrent = train.instance(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < train.numInstances(); ++previous) {
                if (previous == current) continue;

                LazyAssessNN d = lazyAssessNNS[previous];
                d.setWoutKim(train.instance(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // --- --- For each, decreasing (positive) windows --- ---
            for (int paramId = 0; paramId < nParams; ++paramId) {
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

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatTWED(toBeat, nu, lambda);

                    // --- Check the result
                    if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                        double d = challenger.getDistance();
                        currPNN.set(previous, d, PotentialNN.Status.BC);
                        newNN = true;
                    }

                    if (previous < nSamples) {
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatTWED(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            prevNN.set(current, d, PotentialNN.Status.NN);
                        }
                    }
                }

                if (newNN) {
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    nns[paramId][current].set(index, d, PotentialNN.Status.NN);
                }
            }
        }
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
    public double[] loocvEstimate(Instances train, double timeLimit, int instanceLimit) throws Exception {
        if (allowFastWWS)
            return super.fastWWSEstimate(train, timeLimit, instanceLimit);
        else
            return super.loocvEstimate(train, timeLimit, instanceLimit);
    }

    /*------------------------------------------------------------------------------------------------------------------
        Distances
     -----------------------------------------------------------------------------------------------------------------*/
    public final double distance(Instance first, Instance second) {
        return TWED.distance(first, second, nu, lambda);
    }

    public final double lowerbound(Instance q, Instance c, double queryMax, double queryMin, double nu, double lambda) {
        return LbTwed.distance(q, c, queryMax, queryMin, nu, lambda);
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

        double thisDist, lbDist;

        for (Instance i : this.train) {
            lbDist = LbTwed.distance(instance, i, this.queryMax, this.queryMin, this.nu, this.lambda, bsfDistance);
            if (lbDist < bsfDistance) {
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
        this.nu = twe_nuParams[paramId / 10];
        this.lambda = twe_lamdaParams[paramId % 10];
    }

    @Override
    public String getParamInformationString() {
        return "nu=" + this.nu + ", lambda=" + this.lambda;
    }

    public String getParams() {
        return this.nu + "," + this.lambda;
    }

    public double[] getTwe_nuParams() {
        return twe_nuParams;
    }

    public double[] getTwe_lamdaParams() {
        return twe_lamdaParams;
    }
}
