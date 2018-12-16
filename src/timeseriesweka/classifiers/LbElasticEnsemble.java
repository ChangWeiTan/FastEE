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
 * LbEE Classifier
 */
public class LbElasticEnsemble extends ElasticEnsemble {
    public LbElasticEnsemble() {
        classifiersToUse = ConstituentClassifiers.values();
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
            System.out.println("[LbEE] Building " + classifiers[c].getClassifierIdentifier());
            if (isDerivative(classifiersToUse[c])) {
                classifiers[c].setTrainCache(derTrainCache);
                cvAccAndPreds = classifiers[c].loocvWithLowerBound(derTrain);
            } else {
                classifiers[c].setTrainCache(trainCache);
                cvAccAndPreds = classifiers[c].loocvWithLowerBound(train);
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
        st.append("LbEE using:\n");
        st.append("=====================\n");
        for (int c = 0; c < classifiers.length; c++) {
            st.append(classifiersToUse[c]).append(" ").append(classifiers[c].getClassifierIdentifier()).append(" ").append(cvAccs[c]).append("\n");
        }
        return st.toString();
    }
}

