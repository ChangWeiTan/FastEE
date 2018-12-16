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
import timeseriesweka.elasticDistances.ERP;
import timeseriesweka.elasticDistances.LazyAssessNN;
import timeseriesweka.fastWWS.PotentialNN;
import timeseriesweka.fastWWS.SequenceStatsCache;
import timeseriesweka.lowerBounds.LbErp;
import utilities.ClassifierTools;
import utilities.Tools;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * NN-DTW Classifier
 */
public class ERP1NN extends OneNearestNeighbour {
    // parameters
    private double g;                               // g value
    private double bandSize;                        // band size in terms of percentage of sequence's length

    private double[] gValues;                       // set of g values
    private double[] bandSizes;                     // set of band sizes
    private boolean gAndWindowsRefreshed = false;   // indicator if we refresh the params

    public ERP1NN(double g, double bandSize) {
        this.g = g;
        this.bandSize = bandSize;
        this.gAndWindowsRefreshed = false;
        this.classifierIdentifier = "ERP_1NN";
        this.allowLoocv = false;
    }

    public ERP1NN() {
        // note: default params probably won't suit the majority of problems.
        // Should set through cv or prior knowledge
        this.g = 0.5;
        this.bandSize = 5;
        this.gAndWindowsRefreshed = false;
        this.classifierIdentifier = "ERP_1NN";
    }

    public void setG(double g) {
        this.g = g;
    }

    public void setBandSize(double bandSize) {
        this.bandSize = bandSize;
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

        // Vector of LazyUCR lbKeogh, propagating bound info "horizontally"
        LazyAssessNN[] lazyAssessNNS = new LazyAssessNN[train.numInstances()];
        for (int i = 0; i < train.numInstances(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN(cache);
        }
        // "Challengers" that compete with each other to be the NN of query
        ArrayList<LazyAssessNN> challengers = new ArrayList<>(train.numInstances());

        // Iteration for all TS, starting with the second one (first is the reference)
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
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        // --- Get the data
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Try to beat the previous best NN
                        double toBeat = prevNN.distance;
                        LazyAssessNN challenger = lazyAssessNNS[previous];
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatERP(toBeat, g, bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(ERP.getWindowSize(bandSize, train.numAttributes() - 1));
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
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatERP(toBeat, g, bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(ERP.getWindowSize(bandSize, train.numAttributes() - 1));
                            currPNN.set(previous, r, d, PotentialNN.Status.BC);
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatERP(toBeat, g, bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(ERP.getWindowSize(bandSize, train.numAttributes() - 1));
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int w = ERP.getWindowSize(bandSize, train.numAttributes() - 1);
                    int tmp = paramId;
                    double prevG = g;
                    while (tmp > 0 && paramId % 10 > 0 && prevG == g && w >= r) {
                        nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
                        tmp--;

                        this.setParamsFromParamId(train, tmp);
                        w = ERP.getWindowSize(bandSize, train.numAttributes() - 1);
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

        // Vector of LazyUCR lbKeogh, propagating bound info "horizontally"
        LazyAssessNN[] lazyAssessNNS = new LazyAssessNN[n];
        for (int i = 0; i < n; ++i) {
            lazyAssessNNS[i] = new LazyAssessNN(cache);
        }
        // "Challengers" that compete with each other to be the NN of query
        ArrayList<LazyAssessNN> challengers = new ArrayList<>(train.numInstances());

        // Iteration for all TS, starting with the second one (first is the reference)
        for (int current = 1; current < n; ++current) {
            // --- --- Get the data --- ---
            Instance sCurrent = train.instance(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                LazyAssessNN d = lazyAssessNNS[previous];
                d.set(train.instance(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // --- --- For each, decreasing (positive) windows --- ---
            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);

                // --- Get the data
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
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatERP(toBeat, g, bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(ERP.getWindowSize(bandSize, train.numAttributes() - 1));
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
                        LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatERP(toBeat, g, bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(ERP.getWindowSize(bandSize, train.numAttributes() - 1));
                            currPNN.set(previous, r, d, PotentialNN.Status.BC);
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatERP(toBeat, g, bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(ERP.getWindowSize(bandSize, train.numAttributes() - 1));
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int w = ERP.getWindowSize(bandSize, train.numAttributes() - 1);
                    int tmp = paramId;
                    double prevG = g;
                    while (tmp >= 0 && prevG == g && w >= r) {
                        prevG = g;
                        nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
                        tmp--;
                        if (tmp >= 0) {
                            this.setParamsFromParamId(train, tmp);
                            w = ERP.getWindowSize(bandSize, train.numAttributes() - 1);
                        }
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
            // --- --- Get the data --- ---
            Instance sCurrent = train.instance(current);

            // Clear off the previous challengers and add all the previous sequences
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
                    LazyAssessNN.RefineReturnType rrt = challenger.tryToBeatERP(toBeat, g, bandSize);

                    // --- Check the result
                    if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                        int r = challenger.getMinWindowValidityForFullDistance();
                        double d = challenger.getDistance(ERP.getWindowSize(bandSize, train.numAttributes() - 1));
                        currPNN.set(previous, r, d, PotentialNN.Status.BC);
                        newNN = true;
                    }

                    if (previous < nSamples) {
                        PotentialNN prevNN = nns[paramId][previous];

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeatERP(toBeat, g, bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNN.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(ERP.getWindowSize(bandSize, train.numAttributes() - 1));
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                        }
                    }
                }

                if (newNN) {
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int w = ERP.getWindowSize(bandSize, train.numAttributes() - 1);
                    int tmp = paramId;
                    double prevG = g;
                    while (tmp >= 0 && prevG == g && w >= r) {
                        prevG = g;
                        nns[tmp][current].set(index, r, d, PotentialNN.Status.NN);
                        tmp--;
                        if (tmp >= 0) {
                            this.setParamsFromParamId(train, tmp);
                            w = ERP.getWindowSize(bandSize, train.numAttributes() - 1);
                        }
                    }
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
    public void buildClassifier(Instances train) throws Exception {
        super.buildClassifier(train);
        this.gAndWindowsRefreshed = false;
    }

    /*------------------------------------------------------------------------------------------------------------------
        Distances
     -----------------------------------------------------------------------------------------------------------------*/
    public final double distance(Instance first, Instance second) {
        return ERP.distance(first, second, this.g, this.bandSize);
    }

    public final double lowerbound(Instance q, Instance c) {
        double[] U = new double[q.numAttributes() - 1];
        double[] L = new double[q.numAttributes() - 1];
        LbErp.fillUL(q, g, bandSize, U, L);
        return lowerbound(c, U, L);
    }

    public final double lowerbound(Instance c, double[] U, double[] L) {
        return LbErp.distance(c, U, L);
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
        LbErp.fillUL(instance, g, bandSize, U, L);

        for (Instance i : this.train) {
            lbDist = LbErp.distance(i, U, L, bsfDistance);
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
        if (!this.gAndWindowsRefreshed) {
            double stdv = Tools.stdv_p(train);
            bandSizes = Tools.getInclusive10(0, 0.25);
            gValues = Tools.getInclusive10(0.2 * stdv, stdv);
            this.gAndWindowsRefreshed = true;
        }
        this.g = gValues[paramId / 10];
        this.bandSize = bandSizes[paramId % 10];
    }

    @Override
    public String getParamInformationString() {
        return "g=" + this.g + ", bandSize=" + this.bandSize;
    }

    public String getParams() {
        return this.g + "," + this.bandSize;
    }

    public double[] getgValues() {
        return gValues;
    }

    public double[] getBandSizes() {
        return bandSizes;
    }
}
