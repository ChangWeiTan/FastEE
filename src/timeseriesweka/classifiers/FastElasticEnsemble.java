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
import utilities.ClassifierTools;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

import java.util.ArrayList;
import java.util.Random;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * FastEE Classifier
 */
public class FastElasticEnsemble extends ElasticEnsemble {
    @Override
    public TechnicalInformation getTechnicalInformation() {
        //todo: change here when publish
        TechnicalInformation result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "CW. Tan and F. Petitjeaan and G. Webb");
        result.setValue(TechnicalInformation.Field.TITLE, "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");

        return result;
    }

    public FastElasticEnsemble() {
        classifiersToUse = ConstituentClassifiers.values();
        turnOnFastWWS();
    }

    public static void main(String[] args) throws Exception {
        String datasetName = "ArrowHead";
        Instances train = ClassifierTools.loadData("C:\\Users\\cwtan\\workspace\\Dataset\\TSC_Problems\\" +
                datasetName + "\\" + datasetName + "_TRAIN");
        Instances test = ClassifierTools.loadData("C:\\Users\\cwtan\\workspace\\Dataset\\TSC_Problems\\" +
                datasetName + "\\" + datasetName + "_TEST");

        SequenceStatsCache testCache = new SequenceStatsCache(test, test.numAttributes());

        compare(train, test);

        FastElasticEnsemble fastEE = new FastElasticEnsemble();
        fastEE.buildClassifier(train);
        fastEE.setTestCache(test, testCache);
        int correct = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            fastEE.setQueryIndex(i);
            if (test.instance(i).classValue() == fastEE.classifyInstance(test.instance(i))) {
                correct++;
            }
        }
        System.out.println("correct: " + correct + "/" + test.numInstances());
        System.out.println((double) correct / test.numInstances());
        System.out.println(fastEE.getEnsembleCvAcc());
    }

    private static void compare(Instances train, Instances test) throws Exception {
        SequenceStatsCache testCache = new SequenceStatsCache(test, test.numAttributes());
        FastElasticEnsemble fastEE = new FastElasticEnsemble();
        ElasticEnsemble ee = new ElasticEnsemble();

        long startTime, endTime;

        System.out.println("-------------------- FAST EE --------------------");
        startTime = System.nanoTime();
        fastEE.buildClassifier(train);
        endTime = System.nanoTime();
        double fastEETime = 1.0 * (endTime - startTime) / 1e9;
        System.out.println("FastEE Train Time: " + fastEETime + " s");
        System.out.println("-------------------- FAST EE --------------------");
        System.out.println();

        System.out.println("-------------------- EE --------------------");
        startTime = System.nanoTime();
        ee.buildClassifier(train);
        endTime = System.nanoTime();
        double eeTime = 1.0 * (endTime - startTime) / 1e9;
        System.out.println("EE Train Time: " + eeTime + " s");
        System.out.println("-------------------- EE --------------------");
        System.out.println();

        System.out.println("FastEE Train Time:  " + fastEETime + " s");
        System.out.println("EE Train Time:      " + eeTime + " s");
        System.out.println("FastEE CV Acc:      " + fastEE.getEnsembleCvAcc());
        System.out.println("EE CV Acc:          " + ee.getEnsembleCvAcc());


        int correct = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            if (test.instance(i).classValue() == ee.classifyInstance(test.instance(i))) {
                correct++;
            }
        }
        double eeAcc = 1.0 * correct / test.numInstances();

        fastEE.setTestCache(test, testCache);
        correct = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            fastEE.setQueryIndex(i);
            if (test.instance(i).classValue() == fastEE.classifyInstance(test.instance(i))) {
                correct++;
            }
        }
        double fastEEAcc = 1.0 * correct / test.numInstances();

        System.out.println("FastEE Acc:         " + fastEEAcc);
        System.out.println("EE Acc:             " + eeAcc);
        System.out.println("Comparison completed.");
    }

    @Override
    public double accuracy(Instances test) throws Exception {
        double a = 0;
        int size = test.numInstances();
        Instance d;
        double predictedClass, trueClass;
        for (int i = 0; i < size; i++) {
            d = test.instance(i);
            queryIndex = i;
            queryMax = testCache.getMax(i);
            queryMin = testCache.getMin(i);
            predictedClass = classifyInstance(d);
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
        this.trainCache = new SequenceStatsCache(train, train.numAttributes() - 1);
        usesDer = false;

        this.classifiers = new OneNearestNeighbour[this.classifiersToUse.length];
        this.cvAccs = new double[classifiers.length];
        this.cvParamId = new int[classifiers.length];
        this.cvTime = new double[classifiers.length];
        this.cvPreds = new double[classifiers.length][this.train.numInstances()];

        for (int c = 0; c < classifiers.length; c++) {
            classifiers[c] = getClassifier(this.classifiersToUse[c]);
            classifiers[c].setFastWWS(allowFastWWS);
            if (isDerivative(this.classifiersToUse[c])) {
                usesDer = true;
            }
        }

        if (usesDer) {
            this.derTrain = df.process(train);
            this.derTrainCache = new SequenceStatsCache(derTrain, derTrain.numAttributes() - 1);
        }

        trainResults.buildTime = 0;
        double[] cvAccAndPreds;
        for (int c = 0; c < classifiers.length; c++) {
            System.out.println("[FastEE] Building " + classifiers[c].getClassifierIdentifier());
            if (isDerivative(classifiersToUse[c])) {
                classifiers[c].setTrainCache(derTrainCache);
                cvAccAndPreds = classifiers[c].loocv(derTrain);
            } else {
                classifiers[c].setTrainCache(trainCache);
                cvAccAndPreds = classifiers[c].loocv(train);
            }

            cvParamId[c] = classifiers[c].getBsfParamId();
            cvAccs[c] = cvAccAndPreds[0];
            cvTime[c] = classifiers[c].getCvTime();
            System.arraycopy(cvAccAndPreds, 1, this.cvPreds[c], 0, cvAccAndPreds.length - 1);
            trainResults.buildTime += cvTime[c];
        }
    }

    @Override
    public void buildClassifierEstimate(Instances train, double timeLimit, int instanceLimit) throws Exception {
        this.train = train;
        this.derTrain = null;
        this.trainCache = new SequenceStatsCache(train, train.numAttributes() - 1);
        usesDer = false;

        this.classifiers = new OneNearestNeighbour[this.classifiersToUse.length];
        this.cvAccs = new double[classifiers.length];
        this.cvParamId = new int[classifiers.length];
        this.cvTime = new double[classifiers.length];
        this.cvPreds = new double[classifiers.length][this.train.numInstances()];

        for (int c = 0; c < classifiers.length; c++) {
            classifiers[c] = getClassifier(this.classifiersToUse[c]);
            classifiers[c].setFastWWS(allowFastWWS);
            if (isDerivative(this.classifiersToUse[c])) {
                usesDer = true;
            }
        }

        if (usesDer) {
            this.derTrain = df.process(train);
            this.derTrainCache = new SequenceStatsCache(derTrain, derTrain.numAttributes() - 1);
        }

        trainResults.buildTime = 0;
        double[] cvAccAndPreds;
        for (int c = 0; c < classifiers.length; c++) {
            System.out.println("[FastEE] Building " + classifiers[c].getClassifierIdentifier());
            if (isDerivative(classifiersToUse[c])) {
                classifiers[c].setTrainCache(derTrainCache);
                cvAccAndPreds = classifiers[c].loocvEstimate(derTrain, timeLimit, instanceLimit);
            } else {
                classifiers[c].setTrainCache(trainCache);
                cvAccAndPreds = classifiers[c].loocvEstimate(train, timeLimit, instanceLimit);
            }

            cvParamId[c] = classifiers[c].getBsfParamId();
            cvAccs[c] = cvAccAndPreds[0];
            cvTime[c] = classifiers[c].getCvTime();
            System.arraycopy(cvAccAndPreds, 1, this.cvPreds[c], 0, cvAccAndPreds.length - 1);
            trainResults.buildTime += cvTime[c];
        }
    }

    @Override
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
                queryMax = derTestCache.getMax(queryIndex);
                queryMin = derTestCache.getMin(queryIndex);
                classifiers[c].setQueryMinMax(queryMax, queryMin);
                pred = classifiers[c].classifyWithLowerBound(derIns);
            } else {
                queryMax = testCache.getMax(queryIndex);
                queryMin = testCache.getMin(queryIndex);
                classifiers[c].setQueryMinMax(queryMax, queryMin);
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

    @Override
    public String getClassifierInfo() {
        StringBuilder st = new StringBuilder();
        st.append("FastEE using:\n");
        st.append("=====================\n");
        for (int c = 0; c < classifiers.length; c++) {
            st.append(classifiersToUse[c]).append(" ").append(classifiers[c].getClassifierIdentifier()).append(" ").append(cvAccs[c]).append("\n");
        }
        return st.toString();
    }

    public void turnOffFastWWS() {
        this.allowFastWWS = false;
    }

    public void turnOnFastWWS() {
        this.allowFastWWS = true;
    }

    public void setQueryIndex(int i) {
        queryIndex = i;
    }
}

