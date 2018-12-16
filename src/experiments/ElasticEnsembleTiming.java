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
import utilities.ClassifierResults;
import utilities.ClassifierTools;
import weka.core.Instances;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Time the full training process
 */
public class ElasticEnsembleTiming extends Experiments {
    private static String datasetName = "ArrowHead";
    private static String classifierType = "FastEE";
    private static int nSamples = 1;
    
    public static void main(String[] args) throws Exception {
        // retrieve input arguments
        if (args.length > 0) outputPath = args[0];
        else setOutputPath("ensembles");
        if (args.length > 1) datasetPath = args[1];
        if (args.length > 2) datasetName = args[2];
        if (args.length > 3) classifierType = args[3];
        if (args.length > 4) nSamples = Integer.parseInt(args[4]);
        
        setOutputPath(outputPath, datasetName);

        // display them
        System.out.println("[ENSEMBLE] Input arguments:");
        System.out.println(String.format("[ENSEMBLE] Output path     : %s", outputPath));
        System.out.println(String.format("[ENSEMBLE] Dataset path    : %s", datasetPath));
        System.out.println(String.format("[ENSEMBLE] Dataset name    : %s", datasetName));
        System.out.println(String.format("[ENSEMBLE] Distances       : %s", classifierType));
        System.out.println(String.format("[ENSEMBLE] # Samples       : %d", nSamples));

        // load data
        System.out.println(String.format("[ENSEMBLE] Loading %s", datasetName));
        Instances train = ClassifierTools.loadTrain(datasetPath, datasetName);
        Instances test = ClassifierTools.loadTest(datasetPath, datasetName);
        System.out.println(String.format("[ENSEMBLE] %s loaded", datasetName));

        // initialise classifier
        ElasticEnsemble elasticEnsemble;
        switch (classifierType) {
            case "EE":
                elasticEnsemble = new ElasticEnsemble();
                break;
            case "FastEE":
                elasticEnsemble = new FastElasticEnsemble();
                break;
            case "LbEE":
                elasticEnsemble = new LbElasticEnsemble();
                break;
            case "ApproxEE":
                elasticEnsemble = new ApproxElasticEnsemble(nSamples);
                break;
            default:
                throw new RuntimeException("[ENSEMBLE] Undefined EE classifier");
        }

        // start building the classifier
        System.out.println(String.format("[ENSEMBLE] Building %s classifier", classifierType));
        elasticEnsemble.buildClassifier(train);
        ClassifierResults classifierResults = elasticEnsemble.getTrainResults();
        trainTime = classifierResults.buildTime;

        // save training results
        System.out.println(String.format("[ENSEMBLE] Train Time: %.5f s, Accuracy: %.5f", trainTime, classifierResults.acc));
        String trainRes = trainTime + "," + classifierResults.acc;
        OutFile of = new OutFile(outputPath + datasetName + "/" + datasetName + "_" + classifierType + "_TRAIN.csv");
        of.writeLine(trainRes);
        of.closeFile();

        // initialise classification
        System.out.println(String.format("[ENSEMBLE] Caching %s", datasetName));
        elasticEnsemble.setTestCache(test);
        System.out.println(String.format("[ENSEMBLE] %s cached", datasetName));

        // start classification
        long startTime = System.nanoTime();
        double accuracy = elasticEnsemble.accuracy(test);
        long endTime = System.nanoTime();
        testTime = 1.0 * (endTime - startTime) / 1e9;
        System.out.println(String.format("[ENSEMBLE] Test Time: %.5f s, Accuracy: %.5f", testTime, accuracy));

        // save classification results
        String testRes = testTime + "," + accuracy;
        of = new OutFile(outputPath + datasetName + "/" + datasetName + "_" + classifierType + "_TEST.csv");
        of.writeLine(testRes);
        of.closeFile();
        System.out.println();
    }
}
