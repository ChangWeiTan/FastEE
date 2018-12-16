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
import timeseriesweka.classifiers.ElasticEnsemble;
import utilities.ClassifierTools;
import weka.core.Instances;

import java.text.DecimalFormat;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Classification done with the best parameter.
 * Requires the folder: System.getProperty("user.dir") + best_" + method + "_param/
 */
public class EEClassificationBestParam extends Experiments {
    private static String method = "FastEE";
    private static String problem = "ArrowHead";

    public static void main(String[] args) throws Exception {
        DecimalFormat df = new DecimalFormat("##.###");

        // retrieve input arguments
        if (args.length > 0) outputPath = args[0];
        if (args.length > 1) datasetPath = args[1];
        if (args.length > 2) problem = args[2];
        if (args.length > 3) method = args[3];

        setOutputPath(outputPath, problem);

        // display them
        System.out.println("[CLASSIFICATION] Input arguments:");
        System.out.println(String.format("[CLASSIFICATION] Output path     : %s", outputPath));
        System.out.println(String.format("[CLASSIFICATION] Dataset path    : %s", datasetPath));
        System.out.println(String.format("[CLASSIFICATION] Dataset name    : %s", problem));
        System.out.println(String.format("[CLASSIFICATION] Ensemble type   : %s", method));

        // initialise classifier
        ElasticEnsemble ee = new ElasticEnsemble();

        // load data
        Instances train = ClassifierTools.loadTrain(datasetPath, problem);
        Instances test = ClassifierTools.loadTest(datasetPath, problem);

        // load parameters and loocv accuracy
        int[] bestParamId = loadBestParam(problem, method);
        double[] bestCvAcc = loadBestCVAcc(problem, method);

        // set parameters for the classifier
        System.out.println("[CLASSIFICATION] Setting best param for " + method + " on " + problem + " problem.");
        ee.buildClassifierWithParams(train, bestParamId, bestCvAcc);

        // start classification
        System.out.println("[CLASSIFICATION] Testing " + method + " on " + problem + " problem.");
        long startTime = System.nanoTime();
        double a = ee.accuracyWithLowerBound(test);
        long endTime = System.nanoTime();
        testTime = 1.0 * (endTime - startTime) / 1e9;

        // save
        OutFile of = new OutFile(outputPath + problem + "/" + problem + "_" + method + "_TEST.csv");
        String res = testTime + csvDelimiter + a;
        of.writeLine(res);
        of.closeFile();

        a *= 100;
        System.out.println("[CLASSIFICATION] Completed Testing on " + problem + " problem with accuracy of " +
                df.format(a) + "% -- Test Time: " + df.format(testTime) + "s");
    }
}
