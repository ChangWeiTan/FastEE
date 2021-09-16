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
package experiments;

import fileIO.OutFile;
import timeseriesweka.classifiers.*;
import timeseriesweka.fastWWS.SequenceStatsCache;
import utilities.ClassifierTools;
import weka.core.Instances;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Training an indiviual classifier using LbEE
 */
public class IndividualClassifierLbEE extends Experiments {
    private static String datasetName = "Phoneme";
    private static String distanceMeasure = "DTW_Rn";

    public static void main(String[] args) throws Exception {
        // retrieve input arguments
        if (args.length > 0) outputPath = args[0];
        if (args.length > 1) datasetPath = args[1];
        if (args.length > 2) datasetName = args[2];
        if (args.length > 3) distanceMeasure = args[3];

        // set output path
        setOutputPath(outputPath, datasetName);

        // display them
        System.out.println("[LB-EE] Input arguments:");
        System.out.println(String.format("[LB-EE] Output path     : %s", outputPath));
        System.out.println(String.format("[LB-EE] Dataset path    : %s", datasetPath));
        System.out.println(String.format("[LB-EE] Dataset name    : %s", datasetName));
        System.out.println(String.format("[LB-EE] Distances       : %s", distanceMeasure));

        // load data
        System.out.println(String.format("[LB-EE] Loading %s", datasetName));
        Instances train = ClassifierTools.loadTrain(datasetPath, datasetName);
        Instances test = ClassifierTools.loadTest(datasetPath, datasetName);
        System.out.println(String.format("[LB-EE] %s loaded", datasetName));

        // initialise classifier
        FastElasticEnsemble.ConstituentClassifiers classifierType;
        switch (distanceMeasure) {
            case "classifiers.distance.Euclidean":
                classifierType = FastElasticEnsemble.ConstituentClassifiers.Euclidean_1NN;
                break;
            case "DTW_Rn":
                classifierType = FastElasticEnsemble.ConstituentClassifiers.DTW_Rn_1NN;
                break;
            case "DTW_R1":
                classifierType = FastElasticEnsemble.ConstituentClassifiers.DTW_R1_1NN;
                break;
            case "WDTW":
                classifierType = FastElasticEnsemble.ConstituentClassifiers.WDTW_1NN;
                break;
            case "DDTW_Rn":
                classifierType = FastElasticEnsemble.ConstituentClassifiers.DDTW_Rn_1NN;
                train = df.process(train);
                test = df.process(test);
                break;
            case "DDTW_R1":
                classifierType = FastElasticEnsemble.ConstituentClassifiers.DDTW_R1_1NN;
                train = df.process(train);
                test = df.process(test);
                break;
            case "WDDTW":
                classifierType = FastElasticEnsemble.ConstituentClassifiers.WDDTW_1NN;
                train = df.process(train);
                test = df.process(test);
                break;
            case "LCSS":
                classifierType = FastElasticEnsemble.ConstituentClassifiers.LCSS_1NN;
                break;
            case "MSM":
                classifierType = FastElasticEnsemble.ConstituentClassifiers.MSM_1NN;
                break;
            case "TWE":
                classifierType = FastElasticEnsemble.ConstituentClassifiers.TWE_1NN;
                break;
            case "ERP":
                classifierType = FastElasticEnsemble.ConstituentClassifiers.ERP_1NN;
                break;
            default:
                throw new RuntimeException("[LB-EE] Undefined distance measure for EE");
        }

        // initialise cache
        System.out.println(String.format("[LB-EE] Caching %s", datasetName));
        SequenceStatsCache trainCache = new SequenceStatsCache(train, test.numAttributes());
        SequenceStatsCache testCache = new SequenceStatsCache(test, test.numAttributes());
        System.out.println(String.format("[LB-EE] %s cached", datasetName));

        // start building classifier
        System.out.println(String.format("LB-EE] Building NN-%s classifier", distanceMeasure));
        OneNearestNeighbour classifier = getClassifier(classifierType);
        classifier.setFastWWS(false);
        classifier.buildClassifier(train, trainCache);
        System.out.println(String.format("[EE] Training NN-%s classifier", distanceMeasure));
        double[] cvAccAndPred;
        if (train.numInstances() > instanceLimit)
            cvAccAndPred = classifier.loocvEstimateWithLowerBound(train, timeLimit, instanceLimit);
        else
            cvAccAndPred = classifier.loocvWithLowerBound(train);
        trainTime = classifier.getCvTime();
        System.out.println(String.format("[LB-EE] Train Time: %.5f s, Accuracy: %.5f", trainTime, cvAccAndPred[0]));

        // save training results
        String trainRes = trainTime + "," + cvAccAndPred[0] + "," + classifier.getBsfParamId();
        OutFile of = new OutFile(outputPath + datasetName + "/" + datasetName + "_LbEE_" + distanceMeasure + "_TRAIN.csv");
        of.writeLine(trainRes);
        of.closeFile();

        // start classification
        long startTime = System.nanoTime();
        double accuracy = classifier.accuracyWithLowerBound(test, testCache);
        long endTime = System.nanoTime();
        testTime = 1.0 * (endTime - startTime) / 1e9;
        System.out.println(String.format("[LB-EE] Test Time: %.5f s, Accuracy: %.5f", testTime, accuracy));

        // save classification results
        String testRes = testTime + "," + accuracy;
        of = new OutFile(outputPath + datasetName + "/" + datasetName + "_LbEE_" + distanceMeasure + "_TEST.csv");
        of.writeLine(testRes);
        of.closeFile();
        System.out.println();
    }
}
