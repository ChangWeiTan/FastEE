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

import timeseriesweka.fastWWS.SequenceStatsCache;
import timeseriesweka.filters.DerivativeFilter;
import utilities.ClassifierResults;
import utilities.ClassifierTools;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

import java.util.ArrayList;
import java.util.Random;

/**
 * Original code from http://www.timeseriesclassification.com/code.php
 * https://bitbucket.org/TonyBagnall/time-series-classification/src/f47c300b937af1d06aac86bdbe228cb9a9d5e00d?at=default
 */
public class ElasticEnsemble extends AbstractClassifierWithTrainingData {
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "J. Lines and A. Bagnall");
        result.setValue(TechnicalInformation.Field.TITLE, "Time Series Classification with Ensembles of Elastic Distance Measures");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.VOLUME, "29");
        result.setValue(TechnicalInformation.Field.NUMBER, "3");

        result.setValue(TechnicalInformation.Field.PAGES, "565-592");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        return result;
    }

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

    Instances train;
    Instances derTrain;
    ConstituentClassifiers[] classifiersToUse;
    OneNearestNeighbour[] classifiers = null;

    SequenceStatsCache trainCache;
    SequenceStatsCache derTrainCache;
    SequenceStatsCache testCache;
    SequenceStatsCache derTestCache;

    boolean usesDer = false;
    DerivativeFilter df = new DerivativeFilter();

    private double ensembleCvAcc = -1;
    private double[] ensembleCvPreds = null;

    static double queryMax, queryMin;
    int queryIndex;
    boolean allowFastWWS = false;

    int[] cvParamId;
    double[] cvTime;
    double[] cvAccs;
    double[][] cvPreds;

    public ElasticEnsemble() {
        this.classifiersToUse = ConstituentClassifiers.values();
    }

    public static void main(String[] args) throws Exception {
        String datasetName = "ArrowHead";
        Instances train = ClassifierTools.loadData("C:\\Users\\cwtan\\workspace\\Dataset\\TSC_Problems\\" +
                datasetName + "\\" + datasetName + "_TRAIN");
        Instances test = ClassifierTools.loadData("C:\\Users\\cwtan\\workspace\\Dataset\\TSC_Problems\\" +
                datasetName + "\\" + datasetName + "_TEST");

        ElasticEnsemble ee = new ElasticEnsemble();
        ee.buildClassifier(train);
        int correct = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            if (test.instance(i).classValue() == ee.classifyInstance(test.instance(i))) {
                correct++;
            }
        }
        System.out.println("correct: " + correct + "/" + test.numInstances());
        System.out.println((double) correct / test.numInstances());
        System.out.println(ee.getEnsembleCvAcc());
    }

    static boolean isDerivative(ConstituentClassifiers classifier) {
        return (classifier == ConstituentClassifiers.DDTW_R1_1NN ||
                classifier == ConstituentClassifiers.DDTW_Rn_1NN ||
                classifier == ConstituentClassifiers.WDDTW_1NN);
    }

    static boolean isDTW(ConstituentClassifiers classifier) {
        return (classifier == ConstituentClassifiers.DTW_R1_1NN ||
                classifier == ConstituentClassifiers.DDTW_Rn_1NN);
    }

    static boolean isApproxClassifier(ConstituentClassifiers classifier) {
        return (classifier == ConstituentClassifiers.WDTW_1NN ||
                classifier == ConstituentClassifiers.TWE_1NN ||
                classifier == ConstituentClassifiers.WDDTW_1NN);
    }

    static boolean isFixedParam(ConstituentClassifiers classifier) {
        return (classifier == ConstituentClassifiers.DDTW_R1_1NN ||
                classifier == ConstituentClassifiers.DTW_R1_1NN ||
                classifier == ConstituentClassifiers.Euclidean_1NN);
    }

    public String[] getIndividualClassifierNames() {
        String[] names = new String[this.classifiersToUse.length];
        for (int i = 0; i < classifiersToUse.length; i++) {
            names[i] = classifiersToUse[i].toString();
        }
        return names;
    }

    public double getEnsembleCvAcc() {
        if (this.ensembleCvAcc != -1 && this.ensembleCvPreds != null) {
            return this.ensembleCvAcc;
        }

        this.getEnsembleCvPreds();
        return this.ensembleCvAcc;
    }

    private void getEnsembleCvPreds() {
        if (this.ensembleCvPreds != null) {
            return;
        }

        this.ensembleCvPreds = new double[train.numInstances()];

        double actual, pred;
        double bsfWeight;
        int correct = 0;
        ArrayList<Double> bsfClassVals;
        double[] weightByClass;
        for (int i = 0; i < train.numInstances(); i++) {
            actual = train.instance(i).classValue();
            bsfClassVals = null;
            bsfWeight = -1;
            weightByClass = new double[train.numClasses()];

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

        this.ensembleCvAcc = (double) correct / train.numInstances();
    }

    public ClassifierResults getTrainResults() {
        trainResults.acc = getEnsembleCvAcc();
        return trainResults;
    }

    public double accuracy(Instances test) throws Exception {
        double a = 0;
        int size = test.numInstances();
        Instance d;
        double predictedClass, trueClass;
        for (int i = 0; i < size; i++) {
            d = test.instance(i);
            predictedClass = classifyInstance(d);
            trueClass = d.classValue();
            if (trueClass == predictedClass)
                a++;
        }
        return a / size;
    }

    public double accuracyWithLowerBound(Instances test) throws Exception {
        double a = 0;
        int size = test.numInstances();
        Instance d;
        double predictedClass, trueClass;
        for (int i = 0; i < size; i++) {
            d = test.instance(i);
            predictedClass = classifyWithLowerBound(d);
            trueClass = d.classValue();
            if (trueClass == predictedClass)
                a++;
        }
        return a / size;
    }

    @Override
    public void buildClassifier(Instances train) throws Exception {
        this.train = train;
        this.derTrain = null;
        usesDer = false;

        this.classifiers = new OneNearestNeighbour[this.classifiersToUse.length];
        this.cvAccs = new double[classifiers.length];
        this.cvParamId = new int[classifiers.length];
        this.cvTime = new double[classifiers.length];
        this.cvPreds = new double[classifiers.length][this.train.numInstances()];

        for (int c = 0; c < classifiers.length; c++) {
            classifiers[c] = getClassifier(this.classifiersToUse[c]);
            classifiers[c].setFastWWS(false);
            if (isDerivative(this.classifiersToUse[c])) {
                usesDer = true;
            }
        }

        if (usesDer) {
            this.derTrain = df.process(train);
        }

        trainResults.buildTime = 0;
        double[] cvAccAndPreds;
        for (int c = 0; c < classifiers.length; c++) {
            System.out.println("[EE] Building " + classifiers[c].getClassifierIdentifier());
            if (isDerivative(classifiersToUse[c])) {
                cvAccAndPreds = classifiers[c].loocv(derTrain);
            } else {
                cvAccAndPreds = classifiers[c].loocv(train);
            }
            cvParamId[c] = classifiers[c].getBsfParamId();
            cvAccs[c] = cvAccAndPreds[0];
            cvTime[c] = classifiers[c].getCvTime();
            System.arraycopy(cvAccAndPreds, 1, this.cvPreds[c], 0, cvAccAndPreds.length - 1);
            trainResults.buildTime += cvTime[c];
        }
    }

    public void buildClassifierEstimate(Instances train, double timeLimit, int instanceLimit) throws Exception {
        this.train = train;
        this.derTrain = null;
        usesDer = false;

        this.classifiers = new OneNearestNeighbour[this.classifiersToUse.length];
        this.cvAccs = new double[classifiers.length];
        this.cvParamId = new int[classifiers.length];
        this.cvTime = new double[classifiers.length];
        this.cvPreds = new double[classifiers.length][this.train.numInstances()];

        for (int c = 0; c < classifiers.length; c++) {
            classifiers[c] = getClassifier(this.classifiersToUse[c]);
            classifiers[c].setFastWWS(false);
            if (isDerivative(this.classifiersToUse[c])) {
                usesDer = true;
            }
        }

        if (usesDer) {
            this.derTrain = df.process(train);
        }

        trainResults.buildTime = 0;
        double[] cvAccAndPreds;
        for (int c = 0; c < classifiers.length; c++) {
            System.out.println("[EE] Building " + classifiers[c].getClassifierIdentifier());
            if (isDerivative(classifiersToUse[c])) {
                cvAccAndPreds = classifiers[c].loocvEstimate(derTrain, timeLimit, instanceLimit);
            } else {
                cvAccAndPreds = classifiers[c].loocvEstimate(train, timeLimit, instanceLimit);
            }

            cvParamId[c] = classifiers[c].getBsfParamId();
            cvAccs[c] = cvAccAndPreds[0];
            cvTime[c] = classifiers[c].getCvTime();
            System.arraycopy(cvAccAndPreds, 1, this.cvPreds[c], 0, cvAccAndPreds.length - 1);
            trainResults.buildTime += cvTime[c];
        }
    }

    public void buildClassifierWithParams(Instances train, int[] bestParamId, double[] bestCvAcc) throws Exception {
        this.train = train;
        this.derTrain = null;
        usesDer = false;

        this.classifiers = new OneNearestNeighbour[this.classifiersToUse.length];
        this.cvTime = new double[classifiers.length];
        this.cvAccs = new double[classifiers.length];
        this.cvParamId = new int[classifiers.length];
        this.cvPreds = new double[classifiers.length][this.train.numInstances()];

        for (int c = 0; c < classifiers.length; c++) {
            classifiers[c] = getClassifier(this.classifiersToUse[c]);
            if (isDerivative(this.classifiersToUse[c])) {
                usesDer = true;
            }
        }

        if (usesDer) {
            this.derTrain = df.process(train);
        }

        for (int c = 0; c < classifiers.length; c++) {
            System.out.println("[EE] Building " + classifiers[c].getClassifierIdentifier());
            classifiers[c].buildClassifier(train);
            if (isDTW(classifiersToUse[c])) {
                classifiers[c].setParamsFromParamId(train, 100);
            } else if (classifiersToUse[c] != ConstituentClassifiers.Euclidean_1NN) {
                classifiers[c].setParamsFromParamId(train, bestParamId[c]);
            }
        }
        cvAccs = bestCvAcc;
    }

    static OneNearestNeighbour getClassifier(ConstituentClassifiers classifier) throws Exception {
        OneNearestNeighbour knn;
        switch (classifier) {
            case Euclidean_1NN:
                return new ED1NN();
            case DTW_R1_1NN:
                return new DTW1NN(1);
            case DDTW_R1_1NN:
                knn = new DTW1NN(1);
                knn.setClassifierIdentifier(classifier.toString());
                return knn;
            case DTW_Rn_1NN:
                return new DTW1NN();
            case DDTW_Rn_1NN:
                knn = new DTW1NN();
                knn.setClassifierIdentifier(classifier.toString());
                return knn;
            case WDTW_1NN:
                return new WDTW1NN();
            case WDDTW_1NN:
                knn = new WDTW1NN();
                knn.setClassifierIdentifier(classifier.toString());
                return knn;
            case LCSS_1NN:
                return new LCSS1NN();
            case ERP_1NN:
                return new ERP1NN();
            case MSM_1NN:
                return new MSM1NN();
            case TWE_1NN:
                return new TWE1NN();
            default:
                throw new Exception("Unsupported classifier type");
        }
    }

    public double classifyInstance(Instance instance) throws Exception {
        if (classifiers == null) {
            throw new Exception("Error: classifier not built");
        }
        Instance derIns = null;
        if (this.usesDer) {
            Instances temp = new Instances(derTrain, 1);
            temp.add(instance);
            temp = df.process(temp);
            derIns = temp.instance(0);
        }

        double bsfVote = -1;
        double[] classTotals = new double[train.numClasses()];
        ArrayList<Double> bsfClassVal = null;

        double pred;

        for (int c = 0; c < classifiers.length; c++) {
            if (isDerivative(classifiersToUse[c])) {
                pred = classifiers[c].classifyInstance(derIns);
            } else {
                pred = classifiers[c].classifyInstance(instance);
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

    public double classifyWithLowerBound(Instance instance) throws Exception {
        if (classifiers == null) {
            throw new Exception("Error: classifier not built");
        }
        Instance derIns = null;
        if (this.usesDer) {
            Instances temp = new Instances(derTrain, 1);
            temp.add(instance);
            temp = df.process(temp);
            derIns = temp.instance(0);
        }

        double bsfVote = -1;
        double[] classTotals = new double[train.numClasses()];
        ArrayList<Double> bsfClassVal = null;

        double pred;

        for (int c = 0; c < classifiers.length; c++) {
            if (isDerivative(classifiersToUse[c])) {
                pred = classifiers[c].classifyWithLowerBound(derIns);
            } else {
                pred = classifiers[c].classifyWithLowerBound(instance);
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

    public double[] classifyInstanceByConstituents(Instance instance) throws Exception {
        double[] predsByClassifier = new double[this.classifiers.length];

        for (int i = 0; i < classifiers.length; i++) {
            predsByClassifier[i] = classifiers[i].classifyInstance(instance);
        }

        return predsByClassifier;
    }

    public double[] getCVAccs() throws Exception {
        if (this.cvAccs == null) {
            throw new Exception("Error: classifier not built yet");
        }
        return this.cvAccs;
    }

    public String getClassifierInfo() {
        StringBuilder st = new StringBuilder();
        st.append("EE using:\n");
        st.append("=====================\n");
        for (int c = 0; c < classifiers.length; c++) {
            st.append(classifiersToUse[c]).append(" ").append(classifiers[c].getClassifierIdentifier()).append(" ").append(cvAccs[c]).append("\n");
        }
        return st.toString();
    }

    public String getParameters() {
        StringBuilder params = new StringBuilder();
        params.append(super.getParameters()).append(",");
        for (OneNearestNeighbour classifier : classifiers) {
            params.append(classifier.getClassifierIdentifier()).append(",").append(classifier.getParamInformationString()).append(",");
        }
        return params.toString();
    }

    public void setTestCache(SequenceStatsCache cache) {
        this.testCache = cache;
    }

    public void setTestCache(Instances test) {
        this.testCache = new SequenceStatsCache(test, test.numAttributes() - 1);
        if (this.usesDer) {
            Instances derTest = df.process(test);
            this.derTestCache = new SequenceStatsCache(derTest, derTest.numAttributes() - 1);
        }
    }

    public void setTestCache(Instances test, SequenceStatsCache cache) throws Exception {
        this.testCache = cache;
        if (this.usesDer) {
            Instances derTest = df.process(test);
            this.derTestCache = new SequenceStatsCache(derTest, derTest.numAttributes() - 1);
        }
    }

    public void setQueryIndex(int i) {
        queryIndex = i;
    }

    @Override
    public String toString() {
        return super.toString() + "\n" + this.getClassifierInfo();
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

