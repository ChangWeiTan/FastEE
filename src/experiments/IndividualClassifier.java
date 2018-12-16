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
 * Training an indiviual classifier
 */
public class IndividualClassifier extends Experiments {
    private static String datasetName = "ArrowHead";
    private static String distanceMeasure = "DTW_Rn";
    private static String type = "FastEE";

    public static void main(String[] args) throws Exception {
        // retrieve input arguments
        if (args.length > 0) outputPath = args[0];
        if (args.length > 1) datasetPath = args[1];
        if (args.length > 2) datasetName = args[2];
        if (args.length > 3) distanceMeasure = args[3];
        if (args.length > 4) type = args[4];

        // set output path
        setOutputPath(outputPath, datasetName);

        // display them
        System.out.println("[Individual] Input arguments:");
        System.out.println(String.format("[Individual] Output path     : %s", outputPath));
        System.out.println(String.format("[Individual] Dataset path    : %s", datasetPath));
        System.out.println(String.format("[Individual] Dataset name    : %s", datasetName));
        System.out.println(String.format("[Individual] Distances       : %s", distanceMeasure));
        System.out.println(String.format("[Individual] Type of EE      : %s", type));

        // load data
        System.out.println(String.format("[Individual] Loading %s", datasetName));
        Instances train = ClassifierTools.loadTrain(datasetPath, datasetName);
        Instances test = ClassifierTools.loadTest(datasetPath, datasetName);
        System.out.println(String.format("[Individual] %s loaded", datasetName));

        // initialise classifier
        ElasticEnsemble.ConstituentClassifiers classifierType;
        switch (distanceMeasure) {
            case "Euclidean":
                classifierType = ElasticEnsemble.ConstituentClassifiers.Euclidean_1NN;
                break;
            case "DTW_Rn":
                classifierType = ElasticEnsemble.ConstituentClassifiers.DTW_Rn_1NN;
                break;
            case "DTW_R1":
                classifierType = ElasticEnsemble.ConstituentClassifiers.DTW_R1_1NN;
                break;
            case "WDTW":
                classifierType = ElasticEnsemble.ConstituentClassifiers.WDTW_1NN;
                break;
            case "DDTW_Rn":
                classifierType = ElasticEnsemble.ConstituentClassifiers.DDTW_Rn_1NN;
                train = df.process(train);
                test = df.process(test);
                break;
            case "DDTW_R1":
                classifierType = ElasticEnsemble.ConstituentClassifiers.DDTW_R1_1NN;
                train = df.process(train);
                test = df.process(test);
                break;
            case "WDDTW":
                classifierType = ElasticEnsemble.ConstituentClassifiers.WDDTW_1NN;
                train = df.process(train);
                test = df.process(test);
                break;
            case "LCSS":
                classifierType = ElasticEnsemble.ConstituentClassifiers.LCSS_1NN;
                break;
            case "MSM":
                classifierType = ElasticEnsemble.ConstituentClassifiers.MSM_1NN;
                break;
            case "TWE":
                classifierType = ElasticEnsemble.ConstituentClassifiers.TWE_1NN;
                break;
            case "ERP":
                classifierType = ElasticEnsemble.ConstituentClassifiers.ERP_1NN;
                break;
            default:
                throw new RuntimeException("[Individual] Undefined distance measure for EE");
        }

        System.out.println(String.format("[Individual] Building %s-NN-%s classifier", type, distanceMeasure));
        OneNearestNeighbour classifier = getClassifier(classifierType);
        double[] cvAccAndPred;

        // initialise cache
        System.out.println(String.format("[Individual] Caching %s", datasetName));
        SequenceStatsCache trainCache = new SequenceStatsCache(train, test.numAttributes());
        SequenceStatsCache testCache = new SequenceStatsCache(test, test.numAttributes());
        System.out.println(String.format("[Individual] %s cached", datasetName));

        // start building classifier
        if (type.equals("FastEE") || type.equals("ApproxEE")) {
            classifier.setFastWWS(true);
            classifier.buildClassifier(train, trainCache);
            System.out.println(String.format("[Individual] Training %s-NN-%s classifier", type, distanceMeasure));
            cvAccAndPred = classifier.loocv(train);
        } else {
            classifier.setFastWWS(false);
            classifier.buildClassifier(train, trainCache);
            System.out.println("[Individual] Building classifier");
            System.out.println(String.format("[Individual] Training %s-NN-%s classifier", type, distanceMeasure));
            cvAccAndPred = classifier.loocv(train);  // comment if dataset too large
//            cvAccAndPred = classifier.loocvEstimate(train, timeLimit, instanceLimit); // uncomment if dataset too large
        }

        // save training results
        trainTime = classifier.getCvTime();
        System.out.println(String.format("[Individual] Train Time: %.5f s, Accuracy: %.5f", trainTime, cvAccAndPred[0]));
        String trainRes = trainTime + "," + cvAccAndPred[0] + "," + classifier.getBsfParamId();
        OutFile of = new OutFile(outputPath + datasetName + "/" + datasetName + "_" + type + "_" + distanceMeasure + "_TRAIN.csv");
        of.writeLine(trainRes);
        of.closeFile();

        // start classification
        long startTime, endTime;
        double accuracy;
        if (type.equals("EE")) {
            startTime = System.nanoTime();
            accuracy = ClassifierTools.accuracy(test, classifier);
            endTime = System.nanoTime();
        } else {
            startTime = System.nanoTime();
            accuracy = classifier.accuracyWithLowerBound(test, testCache);
            endTime = System.nanoTime();
        }
        testTime = 1.0 * (endTime - startTime) / 1e9;
        System.out.println(String.format("[Individual] Test Time: %.5f s, Accuracy: %.5f", testTime, accuracy));

        // save classification results
        String testRes = testTime + "," + accuracy;
        of = new OutFile(outputPath + datasetName + "/" + datasetName + "_" + type + "_" + distanceMeasure + "_TEST.csv");
        of.writeLine(testRes);
        of.closeFile();
        System.out.println();
    }
}
