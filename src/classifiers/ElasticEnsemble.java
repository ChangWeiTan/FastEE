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
package classifiers;


import application.Application;
import classifiers.nearestNeighbour.*;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.SequenceStatsCache;
import filters.DerivativeFilter;
import results.ClassificationResults;
import results.TrainingClassificationResults;

import java.util.ArrayList;
import java.util.Random;

import static classifiers.TimeSeriesClassifier.TrainOpts.LOOCV;
import static classifiers.TimeSeriesClassifier.TrainOpts.LOOCV0;

/**
 * Original code from http://www.timeseriesclassification.com/code.php
 * https://bitbucket.org/TonyBagnall/time-series-classification/src/f47c300b937af1d06aac86bdbe228cb9a9d5e00d?at=default
 */
public class ElasticEnsemble extends TimeSeriesClassifier {

    // classifiers in EE
    public enum ConstituentClassifiers {
        Euclidean_1NN,
        DTW_R1_1NN,
        DTW_Rn_1NN,
        WDTW_1NN,
        DDTW_R1_1NN,
        DDTW_Rn_1NN,
        WDDTW_1NN,
        LCSS_1NN,
        MSM_1NN,
        TWE_1NN,
        ERP_1NN
    }

    protected String shortName = "EE";

    ConstituentClassifiers[] classifiersToUse;
    OneNearestNeighbour[] classifiers = null;

    SequenceStatsCache testCache;
    SequenceStatsCache derTestCache;

    boolean usesDer = false;

    private double ensembleCvAcc = -1;
    private double[] ensembleCvPreds = null;

    int queryIndex;

    int[] cvParamId;
    double[] cvTime;
    double[] cvAccs;
    double[][] cvPreds;

    public ElasticEnsemble() {
        this.classifierIdentifier = "ElasticEnsemble";
        this.classifiersToUse = ConstituentClassifiers.values();
        this.trainingOptions = LOOCV;
    }

    static boolean isDerivative(ConstituentClassifiers classifier) {
        return (classifier == ConstituentClassifiers.DDTW_R1_1NN ||
                classifier == ConstituentClassifiers.DDTW_Rn_1NN ||
                classifier == ConstituentClassifiers.WDDTW_1NN);
    }

    public String[] getIndividualClassifierNames() {
        String[] names = new String[this.classifiersToUse.length];
        for (int i = 0; i < classifiersToUse.length; i++) {
            names[i] = classifiersToUse[i].toString();
        }
        return names;
    }

    // get CV accuracy of the ensemble
    public void getEnsembleCvAcc() {
        if (this.ensembleCvAcc != -1 && this.ensembleCvPreds != null)
            return;

        this.getEnsembleCvPreds();
    }

    // get CV predictions of the ensemble
    private void getEnsembleCvPreds() {
        if (this.ensembleCvPreds != null)
            return;

        this.ensembleCvPreds = new double[trainData.size()];

        double actual, pred;
        double bsfWeight;
        int correct = 0;
        ArrayList<Double> bsfClassVals;
        double[] weightByClass;
        for (int i = 0; i < trainData.size(); i++) {
            actual = trainData.get(i).classificationLabel;
            bsfClassVals = null;
            bsfWeight = -1;
            weightByClass = new double[trainData.getNumClasses()];

            for (int c = 0; c < classifiers.length; c++) {
                weightByClass[(int) this.cvPreds[c][i]] += this.cvAccs[c];

                if (weightByClass[(int) this.cvPreds[c][i]] > bsfWeight) {
                    bsfWeight = weightByClass[(int) this.cvPreds[c][i]];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(this.cvPreds[c][i]);
                } else if (weightByClass[(int) this.cvPreds[c][i]] == bsfWeight) {
                    assert bsfClassVals != null;
                    bsfClassVals.add(this.cvPreds[c][i]);
                }
            }

            assert bsfClassVals != null;
            if (bsfClassVals.size() > 1) {
                pred = bsfClassVals.get(new Random().nextInt(bsfClassVals.size()));
            } else {
                pred = bsfClassVals.get(0);
            }

            if (pred == actual) {
                correct++;
            }
            this.ensembleCvPreds[i] = pred;
        }

        this.ensembleCvAcc = (double) correct / trainData.size();
    }

    @Override
    public void summary() {
        System.out.println(toString());
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] Members: " + classifiersToUse[0].toString());
        for (int i = 1; i < classifiersToUse.length; i++) {
            str.append(",").append(classifiersToUse[i].toString());
        }
        return str.toString();
    }


    @Override
    public TrainingClassificationResults fit(Sequences train) throws Exception {
        this.trainData = train;
        this.trainData1st = null;
        usesDer = false;

        this.classifiers = new OneNearestNeighbour[this.classifiersToUse.length];
        this.cvAccs = new double[classifiersToUse.length];
        this.cvParamId = new int[classifiersToUse.length];
        this.cvTime = new double[classifiersToUse.length];
        this.cvPreds = new double[classifiersToUse.length][this.trainData.size()];

        for (int c = 0; c < classifiersToUse.length; c++) {
            if (isDerivative(this.classifiersToUse[c])) {
                usesDer = true;
                // do this once
                if (trainData1st == null)
                    this.trainData1st = DerivativeFilter.getFirstDerivative(train);
            }
            classifiers[c] = getClassifier(this.classifiersToUse[c]);
        }

        long buildTime = 0;
        TrainingClassificationResults clfRes;
        for (int c = 0; c < classifiers.length; c++) {
            if (Application.verbose > 1)
                System.out.println("[" + shortName + "] Building " + classifiers[c].classifierIdentifier);

            if (isDerivative(classifiersToUse[c])) {
                clfRes = classifiers[c].fit(trainData1st);
            } else {
                clfRes = classifiers[c].fit(train);
            }
            cvParamId[c] = clfRes.paramId;
            cvAccs[c] = clfRes.accuracy;
            cvTime[c] = clfRes.elapsedTimeNanoSeconds;
            cvPreds[c] = clfRes.predictions;
            buildTime += cvTime[c];
        }
        this.getEnsembleCvPreds();
        this.getEnsembleCvAcc();

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier,
                ensembleCvAcc,
                buildTime,
                ensembleCvPreds);
        results.paramId = bestParamId;
        results.cvAcc = cvAccs;
        results.cvTime = cvTime;
        results.cvPreds = cvPreds;

        return results;
    }


    OneNearestNeighbour getClassifier(ConstituentClassifiers classifier) throws Exception {
        OneNearestNeighbour baseClf;
        switch (classifier) {
            case Euclidean_1NN:
                baseClf = new ED1NN(LOOCV0);
                break;
            case DTW_R1_1NN:
                baseClf = new DTW1NN(100, LOOCV0);
                break;
            case DDTW_R1_1NN:
                baseClf = new DTW1NN(100, LOOCV0);
                baseClf.classifierIdentifier = baseClf.classifierIdentifier.replace("DTW", "DDTW");
                break;
            case DTW_Rn_1NN:
                baseClf = new DTW1NN(-1, this.trainingOptions);
                break;
            case DDTW_Rn_1NN:
                baseClf = new DTW1NN(-1, this.trainingOptions);
                baseClf.classifierIdentifier = baseClf.classifierIdentifier.replace("DTW", "DDTW");
                break;
            case WDTW_1NN:
                baseClf = new WDTW1NN(-1, this.trainingOptions);
                break;
            case WDDTW_1NN:
                baseClf = new WDTW1NN(-1, this.trainingOptions);
                baseClf.classifierIdentifier = baseClf.classifierIdentifier.replace("DTW", "DDTW");
                break;
            case LCSS_1NN:
                baseClf = new LCSS1NN(-1, this.trainingOptions);
                break;
            case ERP_1NN:
                baseClf = new ERP1NN(-1, this.trainingOptions);
                break;
            case MSM_1NN:
                baseClf = new MSM1NN(-1, this.trainingOptions);
                break;
            case TWE_1NN:
                baseClf = new TWE1NN(-1, this.trainingOptions);
                break;
            default:
                throw new Exception("Unsupported classifier type");
        }

        baseClf.useLBAtPrediction = true;
        return baseClf;
    }

    @Override
    public ClassificationResults evaluate(Sequences testData) throws Exception {
        testCache = new SequenceStatsCache(testData, testData.get(0).length());
        Sequences derTestData = null;
        if (this.usesDer) {
            derTestData = DerivativeFilter.getFirstDerivative(testData);
            derTestCache = new SequenceStatsCache(derTestData, derTestData.get(0).length());
        }

        final int testSize = testData.size();
        int nCorrect = 0;
        int numClass = trainData.getNumClasses();
        int[][] confMat = new int[numClass][numClass];
        double[] predictions = new double[testSize];

        final long startTime = System.nanoTime();

        for (queryIndex = 0; queryIndex < testData.size(); queryIndex++) {
            Sequence query = testData.get(queryIndex);
            Sequence derQuery = null;
            if (usesDer) {
                assert derTestData != null;
                derQuery = derTestData.get(queryIndex);
            }
            int predictClass = predict(query, derQuery);
            if (predictClass == query.classificationLabel) nCorrect++;
            confMat[query.classificationLabel][predictClass]++;
            predictions[queryIndex] = predictClass;
        }

        final long stopTime = System.nanoTime();

        classificationResults = new ClassificationResults(
                this.classifierIdentifier,
                this.bestParamId,
                nCorrect,
                testSize,
                startTime,
                stopTime,
                confMat,
                predictions
        );
        return classificationResults;
    }

    public int predict(Sequence instance) throws  Exception{
        Sequence derIns = null;
        if (usesDer){
            derIns = DerivativeFilter.getFirstDerivative(instance);
        }
        return predict(instance, derIns);
    }

    public int predict(Sequence instance, Sequence derIns) throws Exception {
        if (classifiers == null) {
            throw new Exception("Error: classifier not built");
        }

        double bsfVote = -1;
        double[] classTotals = new double[trainData.getNumClasses()];
        ArrayList<Integer> bsfClassVal = null;

        int pred;

        for (int c = 0; c < classifiers.length; c++) {
            if (isDerivative(classifiersToUse[c])) {
                classifiers[c].testCache = derTestCache;
                if (classifiers[c].useLBAtPrediction) pred = classifiers[c].predictWithLb(derIns);
                else pred = classifiers[c].predict(derIns);
            } else {
                classifiers[c].testCache = testCache;
                if (classifiers[c].useLBAtPrediction) pred = classifiers[c].predictWithLb(instance);
                else pred = classifiers[c].predict(instance);
            }
            classTotals[pred] += cvAccs[c];

            if (classTotals[pred] > bsfVote) {
                bsfClassVal = new ArrayList<>();
                bsfClassVal.add(pred);
                bsfVote = classTotals[pred];
            } else if (classTotals[pred] == bsfVote) {
                assert bsfClassVal != null;
                bsfClassVal.add(pred);
            }
        }

        assert bsfClassVal != null;
        if (bsfClassVal.size() > 1) {
            return bsfClassVal.get(new Random(46).nextInt(bsfClassVal.size()));
        }
        return bsfClassVal.get(0);
    }

    @Override
    public String getParamInformationString() {
        return null;
    }

    public double predictWithLb(final Sequence instance) throws Exception {
        if (classifiers == null) {
            throw new Exception("Error: classifier not built");
        }
        Sequence derIns = null;
        if (this.usesDer) {
            derIns = DerivativeFilter.getFirstDerivative(instance);
        }

        double bsfVote = -1;
        double[] classTotals = new double[trainData.getNumClasses()];
        ArrayList<Double> bsfClassVal = null;

        double pred;

        for (int c = 0; c < classifiers.length; c++) {
            if (isDerivative(classifiersToUse[c])) {
                pred = classifiers[c].predictWithLb(derIns);
            } else {
                pred = classifiers[c].predictWithLb(instance);
            }

            try {
                classTotals[(int) pred] += cvAccs[c];
            } catch (Exception e) {
                System.out.println("cv accs " + cvAccs.length);
                System.out.println(pred);
                throw e;
            }

            if (classTotals[(int) pred] > bsfVote) {
                bsfClassVal = new ArrayList<>();
                bsfClassVal.add(pred);
                bsfVote = classTotals[(int) pred];
            } else if (classTotals[(int) pred] == bsfVote) {
                assert bsfClassVal != null;
                bsfClassVal.add(pred);
            }
        }

        assert bsfClassVal != null;
        if (bsfClassVal.size() > 1) {
            return bsfClassVal.get(new Random(46).nextInt(bsfClassVal.size()));
        }
        return bsfClassVal.get(0);
    }


    public double[] getCVAccs() throws Exception {
        if (this.cvAccs == null) {
            throw new Exception("Error: classifier not built yet");
        }
        return this.cvAccs;
    }

    public String getParameters() {
        StringBuilder params = new StringBuilder();
        for (OneNearestNeighbour classifier : classifiers) {
            params.append(classifier.classifierIdentifier).append(",").append(classifier.getParamInformationString()).append(",");
        }
        return params.toString();
    }

    public void setTestCache(SequenceStatsCache cache) {
        this.testCache = cache;
    }

    public void setTestCache(Sequences test) {
        this.testCache = new SequenceStatsCache(test, test.length());
        if (this.usesDer) {
            Sequences derTest = DerivativeFilter.getFirstDerivative(test);
            this.derTestCache = new SequenceStatsCache(derTest, derTest.length());
        }
    }

    public void setTestCache(Sequences test, SequenceStatsCache cache) {
        this.testCache = cache;
        if (this.usesDer) {
            Sequences derTest = DerivativeFilter.getFirstDerivative(test);
            this.derTestCache = new SequenceStatsCache(derTest, derTest.length());
        }
    }

    public void setQueryIndex(int i) {
        queryIndex = i;
    }

    public int[] getCVParams() throws Exception {
        if (this.cvParamId == null) {
            throw new Exception("Error: classifier not built yet");
        }
        return this.cvParamId;
    }

    public double[] getCvTime() throws Exception {
        if (this.cvTime == null) {
            throw new Exception("Error: classifier not built yet");
        }
        return this.cvTime;
    }

    public String doTime(long start, long now) {
        double duration = 1.0 * (now - start) / 1e9;
        return "" + (int) (duration) + " s " + (int) (duration % 1 * 1000) + " ms";
    }
}

