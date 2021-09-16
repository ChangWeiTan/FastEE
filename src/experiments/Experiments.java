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

import timeseriesweka.classifiers.*;
import timeseriesweka.filters.DerivativeFilter;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Experiments
 */
public class Experiments {
    private final static String osName = System.getProperty("os.name");         // get OS Name
    private final static String userName = System.getProperty("user.name");     // get User Name

    protected static int instanceLimit = 500;                       // size limit to estimate the training time
    protected static double timeLimit = 3.6e12;                     // time limit to estimate the training time

    public static String outputPath = setOutputPath();              // output path
    public static String datasetPath = setDatasetPath();            // dataset path

    public static DerivativeFilter df = new DerivativeFilter();     // derivative converter

    public static double trainTime, testTime;                       // training and testing time

    final static String csvDelimiter = ",";                         // csv delimiter

    // load the best parameter
    static int[] loadBestParam(String datasetName, String method) {
        String path = System.getProperty("user.dir") + "/best_" + method.toLowerCase() + "_param/";
        int[] bestParamId = new int[11];
        //              0           1       2       3       4       5       6       7    8    9   10
        // BestParamId: classifiers.distance.Euclidean, DTW_R1, DTW_Rn, WDTW, DDTW_R1, DDTW_Rn, WDDTW, LCSS, MSM, TWE, ERP
        int[] params = readCSVtoInt(path + datasetName + "_best_" + method.toLowerCase() + "_param.csv");
        //          0           1       2       3       4       5       6       7    8    9   10
        // Params: classifiers.distance.Euclidean, DTW_R1, DTW_Rn, DDTW_R1, DDTW_Rn, WDTW, WDDTW, TWE, LCSS, MSM, ERP

        // classifiers.distance.Euclidean
        bestParamId[0] = params[0];
        // DTW_R1
        bestParamId[1] = params[1];
        // DTW_Rn
        bestParamId[2] = params[2];
        // WDTW
        bestParamId[3] = params[5];
        // DDTW_R1
        bestParamId[4] = params[3];
        // DDTW_Rn
        bestParamId[5] = params[4];
        // WDDTW
        bestParamId[6] = params[6];
        // LCSS
        bestParamId[7] = params[8];
        // MSM
        bestParamId[8] = params[9];
        // TWE
        bestParamId[9] = params[7];
        // ERP
        bestParamId[10] = params[10];

        System.out.println("[EXPERIMENT] Best ParamId for " + datasetName + ":");
        System.out.println("[EXPERIMENT] \tEuclidean_1NN: " + bestParamId[0]);
        System.out.println("[EXPERIMENT] \tDTW_R1_1NN: " + bestParamId[1]);
        System.out.println("[EXPERIMENT] \tDTW_Rn_1NN: " + bestParamId[2]);
        System.out.println("[EXPERIMENT] \tWDTW_1NN: " + bestParamId[3]);
        System.out.println("[EXPERIMENT] \tDDTW_R1_1NN: " + bestParamId[4]);
        System.out.println("[EXPERIMENT] \tDDTW_Rn_1NN: " + bestParamId[5]);
        System.out.println("[EXPERIMENT] \tWDDTW_1NN: " + bestParamId[6]);
        System.out.println("[EXPERIMENT] \tLCSS_1NN: " + bestParamId[7]);
        System.out.println("[EXPERIMENT] \tMSM_1NN: " + bestParamId[8]);
        System.out.println("[EXPERIMENT] \tTWE_1NN: " + bestParamId[9]);
        System.out.println("[EXPERIMENT] \tERP_1NN: " + bestParamId[10]);

        return bestParamId;
    }

    // load the best parameter for approx EE
    static int[] loadBestParam(String datasetName, String method, int nSamples, int runs) {
        if (nSamples < 0) return loadBestParam(datasetName, method);

        String path = System.getProperty("user.dir") + "/best_" + method.toLowerCase() + nSamples + "_param/";
        int[] bestParamId = new int[11];
        //              0           1       2       3       4       5       6       7    8    9   10
        // BestParamId: classifiers.distance.Euclidean, DTW_R1, DTW_Rn, WDTW, DDTW_R1, DDTW_Rn, WDDTW, LCSS, MSM, TWE, ERP
        ArrayList<int[]> params = readCSVtoInts(path + datasetName + "_best_" + method.toLowerCase() + nSamples + "_param.csv");
        //          0           1       2       3       4       5       6       7    8    9   10
        // Params: classifiers.distance.Euclidean, DTW_R1, DTW_Rn, DDTW_R1, DDTW_Rn, WDTW, WDDTW, TWE, LCSS, MSM, ERP

        // classifiers.distance.Euclidean
        bestParamId[0] = params.get(runs)[0];
        // DTW_R1
        bestParamId[1] = params.get(runs)[1];
        // DTW_Rn
        bestParamId[2] = params.get(runs)[2];
        // WDTW
        bestParamId[3] = params.get(runs)[5];
        // DDTW_R1
        bestParamId[4] = params.get(runs)[3];
        // DDTW_Rn
        bestParamId[5] = params.get(runs)[4];
        // WDDTW
        bestParamId[6] = params.get(runs)[6];
        // LCSS
        bestParamId[7] = params.get(runs)[8];
        // MSM
        bestParamId[8] = params.get(runs)[9];
        // TWE
        bestParamId[9] = params.get(runs)[7];
        // ERP
        bestParamId[10] = params.get(runs)[10];

        System.out.println("[EXPERIMENT] Best ParamId for " + datasetName + ":");
        System.out.println("[EXPERIMENT] \tEuclidean_1NN: " + bestParamId[0]);
        System.out.println("[EXPERIMENT] \tDTW_R1_1NN: " + bestParamId[1]);
        System.out.println("[EXPERIMENT] \tDTW_Rn_1NN: " + bestParamId[2]);
        System.out.println("[EXPERIMENT] \tWDTW_1NN: " + bestParamId[3]);
        System.out.println("[EXPERIMENT] \tDDTW_R1_1NN: " + bestParamId[4]);
        System.out.println("[EXPERIMENT] \tDDTW_Rn_1NN: " + bestParamId[5]);
        System.out.println("[EXPERIMENT] \tWDDTW_1NN: " + bestParamId[6]);
        System.out.println("[EXPERIMENT] \tLCSS_1NN: " + bestParamId[7]);
        System.out.println("[EXPERIMENT] \tMSM_1NN: " + bestParamId[8]);
        System.out.println("[EXPERIMENT] \tTWE_1NN: " + bestParamId[9]);
        System.out.println("[EXPERIMENT] \tERP_1NN: " + bestParamId[10]);

        return bestParamId;
    }

    // load the best cv accuracy
    static double[] loadBestCVAcc(String datasetName, String method) {
        String path = System.getProperty("user.dir") + "/best_" + method.toLowerCase() + "_param/";
        double[] cvAcc = new double[11];

        double[] acc = readCSVtoDouble(path + datasetName + "_best_" + method.toLowerCase() + "_cv_acc.csv");

        // classifiers.distance.Euclidean
        cvAcc[0] = acc[0];
        // DTW_R1
        cvAcc[1] = acc[1];
        // DTW_Rn
        cvAcc[2] = acc[2];
        // WDTW
        cvAcc[3] = acc[5];
        // DDTW_R1
        cvAcc[4] = acc[3];
        // DDTW_Rn
        cvAcc[5] = acc[4];
        // WDDTW
        cvAcc[6] = acc[6];
        // LCSS
        cvAcc[7] = acc[8];
        // MSM
        cvAcc[8] = acc[9];
        // TWE
        cvAcc[9] = acc[7];
        // ERP
        cvAcc[10] = acc[10];

        System.out.println("[EXPERIMENT] Best CV Acc for " + datasetName + ":");
        System.out.println("[EXPERIMENT] \tEuclidean_1NN: " + cvAcc[0]);
        System.out.println("[EXPERIMENT] \tDTW_R1_1NN: " + cvAcc[1]);
        System.out.println("[EXPERIMENT] \tDTW_Rn_1NN: " + cvAcc[2]);
        System.out.println("[EXPERIMENT] \tWDTW_1NN: " + cvAcc[3]);
        System.out.println("[EXPERIMENT] \tDDTW_R1_1NN: " + cvAcc[4]);
        System.out.println("[EXPERIMENT] \tDDTW_Rn_1NN: " + cvAcc[5]);
        System.out.println("[EXPERIMENT] \tWDDTW_1NN: " + cvAcc[6]);
        System.out.println("[EXPERIMENT] \tLCSS_1NN: " + cvAcc[7]);
        System.out.println("[EXPERIMENT] \tMSM_1NN: " + cvAcc[8]);
        System.out.println("[EXPERIMENT] \tTWE_1NN: " + cvAcc[9]);
        System.out.println("[EXPERIMENT] \tERP_1NN: " + cvAcc[10]);

        return cvAcc;
    }

    // load the best cv accuracy for approx EE
    static double[] loadBestCVAcc(String datasetName, String method, int nSamples, int runs) {
        if (nSamples < 0) return loadBestCVAcc(datasetName, method);

        String path = System.getProperty("user.dir") + "/best_" + method.toLowerCase() + nSamples + "_param/";
        double[] cvAcc = new double[11];

        ArrayList<double[]> acc = readCSVtoDoubles(path + datasetName + "_best_" + method.toLowerCase() + nSamples + "_cv_acc.csv");

        // classifiers.distance.Euclidean
        cvAcc[0] = acc.get(runs)[0];
        // DTW_R1
        cvAcc[1] = acc.get(runs)[1];
        // DTW_Rn
        cvAcc[2] = acc.get(runs)[2];
        // WDTW
        cvAcc[3] = acc.get(runs)[5];
        // DDTW_R1
        cvAcc[4] = acc.get(runs)[3];
        // DDTW_Rn
        cvAcc[5] = acc.get(runs)[4];
        // WDDTW
        cvAcc[6] = acc.get(runs)[6];
        // LCSS
        cvAcc[7] = acc.get(runs)[8];
        // MSM
        cvAcc[8] = acc.get(runs)[9];
        // TWE
        cvAcc[9] = acc.get(runs)[7];
        // ERP
        cvAcc[10] = acc.get(runs)[10];

        System.out.println("[EXPERIMENT] Best CV Acc for " + datasetName + ":");
        System.out.println("[EXPERIMENT] \tEuclidean_1NN: " + cvAcc[0]);
        System.out.println("[EXPERIMENT] \tDTW_R1_1NN: " + cvAcc[1]);
        System.out.println("[EXPERIMENT] \tDTW_Rn_1NN: " + cvAcc[2]);
        System.out.println("[EXPERIMENT] \tWDTW_1NN: " + cvAcc[3]);
        System.out.println("[EXPERIMENT] \tDDTW_R1_1NN: " + cvAcc[4]);
        System.out.println("[EXPERIMENT] \tDDTW_Rn_1NN: " + cvAcc[5]);
        System.out.println("[EXPERIMENT] \tWDDTW_1NN: " + cvAcc[6]);
        System.out.println("[EXPERIMENT] \tLCSS_1NN: " + cvAcc[7]);
        System.out.println("[EXPERIMENT] \tMSM_1NN: " + cvAcc[8]);
        System.out.println("[EXPERIMENT] \tTWE_1NN: " + cvAcc[9]);
        System.out.println("[EXPERIMENT] \tERP_1NN: " + cvAcc[10]);

        return cvAcc;
    }

    // generate random parameter
    static int[] genRandomParams(String datasetName) {
        final int rangeMin = 0, rangeMax = 99;
        int[] randomParamId = new int[11];
        for (int i = 0; i < randomParamId.length; i++) {
            randomParamId[i] = new Random().nextInt((rangeMax - rangeMin) + 1);
        }

        System.out.println("[EXPERIMENT] Random ParamId for " + datasetName + ":");
        System.out.println("[EXPERIMENT] \tEuclidean_1NN: " + randomParamId[0]);
        System.out.println("[EXPERIMENT] \tDTW_R1_1NN: " + randomParamId[1]);
        System.out.println("[EXPERIMENT] \tDTW_Rn_1NN: " + randomParamId[2]);
        System.out.println("[EXPERIMENT] \tWDTW_1NN: " + randomParamId[3]);
        System.out.println("[EXPERIMENT] \tDDTW_R1_1NN: " + randomParamId[4]);
        System.out.println("[EXPERIMENT] \tDDTW_Rn_1NN: " + randomParamId[5]);
        System.out.println("[EXPERIMENT] \tWDDTW_1NN: " + randomParamId[6]);
        System.out.println("[EXPERIMENT] \tLCSS_1NN: " + randomParamId[7]);
        System.out.println("[EXPERIMENT] \tMSM_1NN: " + randomParamId[8]);
        System.out.println("[EXPERIMENT] \tTWE_1NN: " + randomParamId[9]);
        System.out.println("[EXPERIMENT] \tERP_1NN: " + randomParamId[10]);

        return randomParamId;
    }

    // generate random weights (loocv accuracy)
    static double[] genRandomWeights(String datasetName) {
        final double rangeMin = 0, rangeMax = 1;
        double[] randomWeights = new double[11];
        for (int i = 0; i < randomWeights.length; i++) {
            double r = new Random().nextDouble();
            randomWeights[i] = rangeMin + (rangeMax - rangeMin) * r;
        }

        System.out.println("[EXPERIMENT] Random Weights for " + datasetName + ":");
        System.out.println("[EXPERIMENT] \tEuclidean_1NN: " + randomWeights[0]);
        System.out.println("[EXPERIMENT] \tDTW_R1_1NN: " + randomWeights[1]);
        System.out.println("[EXPERIMENT] \tDTW_Rn_1NN: " + randomWeights[2]);
        System.out.println("[EXPERIMENT] \tWDTW_1NN: " + randomWeights[3]);
        System.out.println("[EXPERIMENT] \tDDTW_R1_1NN: " + randomWeights[4]);
        System.out.println("[EXPERIMENT] \tDDTW_Rn_1NN: " + randomWeights[5]);
        System.out.println("[EXPERIMENT] \tWDDTW_1NN: " + randomWeights[6]);
        System.out.println("[EXPERIMENT] \tLCSS_1NN: " + randomWeights[7]);
        System.out.println("[EXPERIMENT] \tMSM_1NN: " + randomWeights[8]);
        System.out.println("[EXPERIMENT] \tTWE_1NN: " + randomWeights[9]);
        System.out.println("[EXPERIMENT] \tERP_1NN: " + randomWeights[10]);

        return randomWeights;
    }

    // generate equal weights (loocv accuracy)
    static double[] genEqualWeights(String datasetName) {
        double[] equalWeights = new double[11];
        for (int i = 0; i < equalWeights.length; i++) {
            equalWeights[i] = 1;
        }

        System.out.println("[EXPERIMENT] Random Weights for " + datasetName + ":");
        System.out.println("[EXPERIMENT] \tEuclidean_1NN: " + equalWeights[0]);
        System.out.println("[EXPERIMENT] \tDTW_R1_1NN: " + equalWeights[1]);
        System.out.println("[EXPERIMENT] \tDTW_Rn_1NN: " + equalWeights[2]);
        System.out.println("[EXPERIMENT] \tWDTW_1NN: " + equalWeights[3]);
        System.out.println("[EXPERIMENT] \tDDTW_R1_1NN: " + equalWeights[4]);
        System.out.println("[EXPERIMENT] \tDDTW_Rn_1NN: " + equalWeights[5]);
        System.out.println("[EXPERIMENT] \tWDDTW_1NN: " + equalWeights[6]);
        System.out.println("[EXPERIMENT] \tLCSS_1NN: " + equalWeights[7]);
        System.out.println("[EXPERIMENT] \tMSM_1NN: " + equalWeights[8]);
        System.out.println("[EXPERIMENT] \tTWE_1NN: " + equalWeights[9]);
        System.out.println("[EXPERIMENT] \tERP_1NN: " + equalWeights[10]);

        return equalWeights;
    }

    // read csv file
    private static double[] readCSVtoDouble(String filename) {
        BufferedReader br = null;
        String line;
        double[] data = null;

        try {
            br = new BufferedReader(new FileReader(filename));
            line = br.readLine();
            String[] lines = line.split(csvDelimiter);
            data = new double[lines.length];
            for (int i = 0; i < data.length; i++) {
                data[i] = Double.valueOf(lines[i]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return data;
    }

    // read csv file
    private static ArrayList<double[]> readCSVtoDoubles(String filename) {
        BufferedReader br = null;
        String line;
        ArrayList<double[]> data = new ArrayList<>();

        try {
            br = new BufferedReader(new FileReader(filename));
            while ((line = br.readLine()) != null) {
                String[] lines = line.split(csvDelimiter);
                double[] tmp = new double[lines.length];
                for (int i = 0; i < tmp.length; i++) {
                    tmp[i] = Double.valueOf(lines[i]);
                }
                data.add(tmp);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return data;
    }

    // read csv file
    private static int[] readCSVtoInt(String filename) {
        BufferedReader br = null;
        String line;
        int[] data = null;

        try {
            br = new BufferedReader(new FileReader(filename));
            line = br.readLine();
            String[] lines = line.split(csvDelimiter);
            data = new int[lines.length];
            for (int i = 0; i < data.length; i++) {
                data[i] = Integer.valueOf(lines[i]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return data;
    }

    // read csv file
    private static ArrayList<int[]> readCSVtoInts(String filename) {
        BufferedReader br = null;
        String line;
        ArrayList<int[]> data = new ArrayList<>();

        try {
            br = new BufferedReader(new FileReader(filename));
            while ((line = br.readLine()) != null) {
                String[] lines = line.split(csvDelimiter);
                int[] tmp = new int[lines.length];
                for (int i = 0; i < tmp.length; i++) {
                    tmp[i] = Integer.valueOf(lines[i]);
                }
                data.add(tmp);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return data;
    }

    // get EE classifier
    public static OneNearestNeighbour getClassifier(ElasticEnsemble.ConstituentClassifiers classifier) throws Exception {
        OneNearestNeighbour oneNearestNeighbour;
        switch (classifier) {
            case Euclidean_1NN:
                return new ED1NN();
            case DTW_R1_1NN:
                return new DTW1NN(1);
            case DDTW_R1_1NN:
                oneNearestNeighbour = new DTW1NN(1);
                oneNearestNeighbour.setClassifierIdentifier(classifier.toString());
                return oneNearestNeighbour;
            case DTW_Rn_1NN:
                return new DTW1NN();
            case DDTW_Rn_1NN:
                oneNearestNeighbour = new DTW1NN();
                oneNearestNeighbour.setClassifierIdentifier(classifier.toString());
                return oneNearestNeighbour;
            case WDTW_1NN:
                return new WDTW1NN();
            case WDDTW_1NN:
                oneNearestNeighbour = new WDTW1NN();
                oneNearestNeighbour.setClassifierIdentifier(classifier.toString());
                return oneNearestNeighbour;
            case LCSS_1NN:
                return new LCSS1NN();
            case ERP_1NN:
                return new ERP1NN();
            case MSM_1NN:
                return new MSM1NN();
            case TWE_1NN:
                return new TWE1NN();
            default:
                throw new Exception("[EXPERIMENTS] Unsupported classifier type");
        }
    }

    // set output path
    public static String setOutputPath(String folder) {
        outputPath = System.getProperty("user.dir") + "/output/" + folder + "/";

        File dir = new File(outputPath);
        if (!dir.exists()) dir.mkdirs();
        return outputPath;
    }

    // set output path
    public static String setOutputPath(String outputPath, String folder) {
        outputPath = outputPath + folder + "/";

        File dir = new File(outputPath);
        if (!dir.exists()) dir.mkdirs();
        return outputPath;
    }

    // set output path
    private static String setOutputPath() {
        outputPath = System.getProperty("user.dir") + "/output/individual/";

        File dir = new File(outputPath);
        if (!dir.exists()) dir.mkdirs();
        return outputPath;
    }

    // set dataset path
    private static String setDatasetPath() {
        if (osName.contains("Window")) {
            datasetPath = "C:/Users/" + userName + "/workspace/Dataset/TSC_Problems/";
        } else {
            datasetPath = "/home/" + userName + "/workspace/Dataset/TSC_Problems/";
        }

        return datasetPath;
    }

    // print time
    public String doTime(long start) {
        long duration = System.nanoTime() - start;
        return "" + (duration / 1e9) + " s " + (duration % 1e9) + " ns";
    }

    // print time
    public String doTime(long start, long now) {
        long duration = now - start;
        return "" + (duration / 1e9) + " s " + (duration % 1e9) + " ns";
    }

}
