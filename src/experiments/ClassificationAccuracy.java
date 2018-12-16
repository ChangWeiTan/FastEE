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

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Get classification accuracy for all 100 parameters
 */
public class ClassificationAccuracy extends Experiments {
    private static String datasetName = "ArrowHead";
    private static String distanceMeasure = "TWE";

    public static void main(String[] args) throws Exception {
        outputPath = System.getProperty("user.dir") + "/output/ClassificationAccuracy/";

        // retrieve input arguments
        if (args.length > 0) outputPath = args[0];
        if (args.length > 1) datasetPath = args[1];
        if (args.length > 2) datasetName = args[2];
        if (args.length > 3) distanceMeasure = args[3];

        setOutputPath(outputPath, datasetName);

        // display them
        System.out.println("[CLASSIFICATION-ACC] Input arguments:");
        System.out.println(String.format("[CLASSIFICATION-ACC] Output path     : %s", outputPath));
        System.out.println(String.format("[CLASSIFICATION-ACC] Dataset path    : %s", datasetPath));
        System.out.println(String.format("[CLASSIFICATION-ACC] Dataset name    : %s", datasetName));
        System.out.println(String.format("[CLASSIFICATION-ACC] Distances       : %s", distanceMeasure));

        System.out.println(String.format("[CLASSIFICATION-ACC] Loading %s", datasetName));
        Instances train = ClassifierTools.loadTrain(datasetPath, datasetName);
        Instances test = ClassifierTools.loadTest(datasetPath, datasetName);
        System.out.println(String.format("[CLASSIFICATION-ACC] %s loaded", datasetName));

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
                throw new RuntimeException("[CLASSIFICATION-ACC] Undefined distance measure for EE");
        }
        // initialise cache
        System.out.println(String.format("[CLASSIFICATION-ACC] Caching %s", datasetName));
        SequenceStatsCache trainCache = new SequenceStatsCache(train, test.numAttributes());
        SequenceStatsCache testCache = new SequenceStatsCache(test, test.numAttributes());
        System.out.println(String.format("[CLASSIFICATION-ACC] %s cached", datasetName));

        // start building the classifier
        System.out.println(String.format("[CLASSIFICATION-ACC] Building NN-%s classifier", distanceMeasure));
        OneNearestNeighbour classifier = getClassifier(classifierType);
        classifier.setFastWWS(true);
        classifier.buildClassifier(train, trainCache);

        for (int i = 0; i < 100; i++) {
            boolean append = i > 0;
            classifier.setParamsFromParamId(train, i);
            long startTime = System.nanoTime();
            double accuracy = classifier.accuracyWithLowerBound(test, testCache);
            long endTime = System.nanoTime();
            testTime = 1.0 * (endTime - startTime) / 1e9;
            System.out.println(String.format("[CLASSIFICATION-ACC] %d -- Test Time: %.5f s, Accuracy: %.5f", i, testTime, accuracy));
            String testRes = i + "," + testTime + "," + accuracy;
            OutFile of = new OutFile(outputPath + datasetName + "/" + datasetName + "_" + distanceMeasure + "_TEST.csv", append);
            of.writeLine(testRes);
            of.closeFile();
        }
        System.out.println();
    }
}
