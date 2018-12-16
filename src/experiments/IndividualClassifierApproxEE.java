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
import timeseriesweka.classifiers.FastElasticEnsemble;
import timeseriesweka.classifiers.OneNearestNeighbour;
import timeseriesweka.fastWWS.SequenceStatsCache;
import utilities.ClassifierTools;
import weka.core.Instances;

import java.util.Random;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Training an indiviual classifier using ApproxEE
 */
public class IndividualClassifierApproxEE extends Experiments {
    private static String datasetName = "ChlorineConcentration";
    private static String distanceMeasure = "WDTW";
    private static int nSamples = 2;
    private static int runs = 0;

    public static void main(String[] args) throws Exception {
        // retrieve input arguments
        if (args.length > 0) outputPath = args[0];
        if (args.length > 1) datasetPath = args[1];
        if (args.length > 2) datasetName = args[2];
        if (args.length > 3) distanceMeasure = args[3];
        if (args.length > 4) nSamples = Integer.parseInt(args[4]);
        if (args.length > 5) runs = Integer.parseInt(args[5]);
        boolean append = runs > 0;

        // set output path
        setOutputPath(outputPath, datasetName);

        // display them
        System.out.println("[APPROX-EE] Input arguments:");
        System.out.println(String.format("[APPROX-EE] Output path     : %s", outputPath));
        System.out.println(String.format("[APPROX-EE] Dataset path    : %s", datasetPath));
        System.out.println(String.format("[APPROX-EE] Dataset name    : %s", datasetName));
        System.out.println(String.format("[APPROX-EE] Distances       : %s", distanceMeasure));
        System.out.println(String.format("[APPROX-EE] # Samples       : %s", nSamples));
        System.out.println(String.format("[APPROX-EE] Runs            : %s", runs));

        // load data
        System.out.println(String.format("[APPROX-EE] Loading %s", datasetName));
        Instances train = ClassifierTools.loadTrain(datasetPath, datasetName);
        Instances test = ClassifierTools.loadTest(datasetPath, datasetName);
        System.out.println(String.format("[APPROX-EE] %s loaded", datasetName));
        nSamples = Math.min(train.numInstances(), nSamples);

        // initialise classifier
        FastElasticEnsemble.ConstituentClassifiers classifierType;
        switch (distanceMeasure) {
            case "Euclidean":
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
                throw new RuntimeException("[APPROX-EE] Undefined distance measure for EE");
        }
        // initialise cache
        System.out.println(String.format("[APPROX-EE] Caching %s", datasetName));
        SequenceStatsCache trainCache = new SequenceStatsCache(train, test.numAttributes());
        SequenceStatsCache testCache = new SequenceStatsCache(test, test.numAttributes());
        System.out.println(String.format("[APPROX-EE] %s cached", datasetName));

        // start building classifier
        System.out.println(String.format("[APPROX-EE] Building NN-%s classifier with run %d", distanceMeasure, runs));
        OneNearestNeighbour classifier = getClassifier(classifierType);
        classifier.setFastWWS(true);
        classifier.setTrainResample(runs);
        classifier.buildClassifier(train, trainCache);
        if (distanceMeasure.equals("WDTW") || distanceMeasure.equals("WDDTW") || distanceMeasure.equals("TWE")) {
            classifier.setApproxSamples(nSamples);
        }
        System.out.println(String.format("[APPROX-EE] Training NN-%s classifier with run %d", distanceMeasure, runs));
        double[] cvAccAndPred = classifier.fastWWSApproximate(train, nSamples);
        trainTime = classifier.getCvTime();
        System.out.println(String.format("[APPROX-EE] Train Time: %.5f s, Accuracy: %.5f", trainTime, cvAccAndPred[0]));

        // save training results
        String trainRes = trainTime + "," + cvAccAndPred[0] + "," + classifier.getBsfParamId();
        OutFile of = new OutFile(outputPath + datasetName + "/" + datasetName + "_ApproxEE" + nSamples + "_" + distanceMeasure + "_TRAIN.csv", append);
        of.writeLine(trainRes);
        of.closeFile();

        // start classification
        long startTime = System.nanoTime();
        double accuracy = classifier.accuracyWithLowerBound(test, testCache);
        long endTime = System.nanoTime();
        testTime = 1.0 * (endTime - startTime) / 1e9;
        System.out.println(String.format("[APPROX-EE] Test Time: %.5f s, Accuracy: %.5f", testTime, accuracy));

        // save classification results
        String testRes = testTime + "," + accuracy;
        of = new OutFile(outputPath + datasetName + "/" + datasetName + "_ApproxEE" + nSamples + "_" + distanceMeasure + "_TEST.csv", append);
        of.writeLine(testRes);
        of.closeFile();
        System.out.println();
    }
}
