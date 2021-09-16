package application;

import classifiers.*;
import classifiers.nearestNeighbour.*;
import datasets.Sequences;
import fileIO.OutFile;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import results.ClassificationResults;
import results.TrainingClassificationResults;

import java.io.*;
import java.util.Iterator;

public class Application {
    public static Runtime runtime = Runtime.getRuntime();   // get the run time of the application
    public static final String optionsSep = "=";
    public static String outputPath;
    public static String datasetPath;
    public static String problem = "";
    public static String classifierName = "DTW-1NN";
    public static String machine = "windows";
    public static int paramId = 0;
    public static int verbose = 0;
    public static boolean znorm = true;
    public static int numThreads = 0;
    public static int iteration = 0;
    public static double scalabilityTrainRatio = 0;
    public static double scalabilityLengthRatio = 0;
    public static boolean doEvaluation = true;
    public static TimeSeriesClassifier.TrainOpts trainOpts = TimeSeriesClassifier.TrainOpts.FastCV;
    private final static String defaultSaveFilename = "results.csv";
    private final static String defaultSaveFilenameJSON = "results.json";

    public static void extractArguments(final String[] args) throws Exception {
        System.out.print("[APP] Input arguments:");
        for (String arg : args) {
            final String[] options = arg.trim().split(optionsSep);
            System.out.print(" " + arg);
            if (options.length >= 2)
                switch (options[0]) {
                    case "-out":
                        outputPath = options[1];
                        break;
                    case "-machine":
                        machine = options[1];
                        break;
                    case "-data":
                        datasetPath = options[1];
                        break;
                    case "-problem":
                        problem = options[1];
                        break;
                    case "-paramId":
                        paramId = Integer.parseInt(options[1]);
                        break;
                    case "-znorm":
                        znorm = Boolean.parseBoolean(options[1]);
                        break;
                    case "-trainOpts":
                        int a = Integer.parseInt(options[1]);
                        if (a == 0) trainOpts = TimeSeriesClassifier.TrainOpts.LOOCV;
                        else if (a == 1) trainOpts = TimeSeriesClassifier.TrainOpts.LOOCVLB;
                        else trainOpts = TimeSeriesClassifier.TrainOpts.FastCV;
                        break;
                    case "-classifier":
                        classifierName = options[1];
                        break;
                    case "-cpu":
                        numThreads = Integer.parseInt(options[1]);
                        break;
                    case "-iter":
                        iteration = Integer.parseInt(options[1]);
                        break;
                    case "-trainSize":
                        scalabilityTrainRatio = Double.parseDouble(options[1]);
                        break;
                    case "-length":
                        scalabilityLengthRatio = Double.parseDouble(options[1]);
                        break;
                    case "-verbose":
                        verbose = Integer.parseInt(options[1]);
                        break;
                    case "-eval":
                        doEvaluation = Boolean.parseBoolean(options[1]);
                        break;
                    default:
                        throw new Exception("Try -out <output_path>, -data <dataset_path>, -problem <problem>, -paramId <paramId>");
                }
            else
                throw new Exception("Try -out <output_path>, -data <dataset_path>, -problem <problem>, -paramId <paramId>");
        }

        System.out.println();

        if (classifierName.toLowerCase().contains("euclidean") || classifierName.toLowerCase().contains("ed"))
            paramId = -1;

        if (Application.datasetPath == null) {
            String username = System.getProperty("user.name");
            if (machine.equals("windows")) {
                Application.datasetPath = "C:/Users/" + username + "/workspace/Dataset/UCRArchive_2018/";
            } else if (machine.equals("m3")) {
                Application.datasetPath = "/projects/nc23/changwei/Dataset/UCRArchive_2018/";
            } else {
                Application.datasetPath = "/home/" + username + "/workspace/Dataset/UCRArchive_2018/";
            }
        }
    }

    public static TimeSeriesClassifier initTSC(final Sequences trainData) {
        TimeSeriesClassifier classifier;
        switch (classifierName) {
            // Elastic Ensemble
            case "ElasticEnsemble":
            case "EE":
                classifier = new ElasticEnsemble();
                break;
            // LB Elastic Ensemble
            case "LbElasticEnsemble":
            case "LbEE":
                classifier = new LbElasticEnsemble();
                break;
            // Fast Elastic Ensemble
            case "FastElasticEnsemble":
            case "FastEE":
                classifier = new FastElasticEnsemble();
                break;
            // Euclidean and Derivative Euclidean 1NN
            case "ED-1NN":
            case "ED1NN":
                classifier = new ED1NN(trainOpts);
                break;
            case "DED-1NN":
            case "DED1NN":
                classifier = new ED1NN(trainOpts, 1);
                break;
            // TWE 1NN
            case "TWE-1NN":
            case "TWE1NN":
                classifier = new TWE1NN(paramId, trainOpts);
                break;
            case "DTWE-1NN":
            case "DTWE1NN":
                classifier = new TWE1NN(paramId, trainOpts, 1);
                break;
            // LCSS and Derivative LCSS 1NN
            case "LCSS-1NN":
            case "LCSS1NN":
                classifier = new LCSS1NN(paramId, trainOpts);
                break;
            case "DLCSS-1NN":
            case "DLCSS1NN":
                classifier = new LCSS1NN(paramId, trainOpts, 1);
                break;
            // ERP and Derivative ERP 1NN
            case "ERP-1NN":
            case "ERP1NN":
                classifier = new ERP1NN(paramId, trainOpts);
                break;
            case "DERP-1NN":
            case "DERP1NN":
                classifier = new ERP1NN(paramId, trainOpts, 1);
                break;
            // MSM and Derivative MSM 1NN
            case "MSM-1NN":
            case "MSM1NN":
                classifier = new MSM1NN(paramId, trainOpts);
                break;
            case "DMSM-1NN":
            case "DMSM1NN":
                classifier = new MSM1NN(paramId, trainOpts, 1);
                break;
            // WDTW and Derivative WDTW 1NN
            case "WDTW-1NN":
            case "WDTW1NN":
                classifier = new WDTW1NN(paramId, trainOpts);
                break;
            case "WDDTW-1NN":
            case "WDDTW1NN":
                classifier = new WDTW1NN(paramId, trainOpts, 1);
                break;
            // Derivative DTW 1NN
            case "DDTW-1NN":
            case "DDTW1NN":
                classifier = new DTW1NN(paramId, trainOpts, 1);
                break;
            default:
                // DTW-1NN
                classifier = new DTW1NN(paramId, trainOpts);

                break;
        }

        return classifier;
    }

    public static void saveResults(String outputPath,
                                   TrainingClassificationResults trainingClassificationResults,
                                   ClassificationResults classificationResults) throws Exception {
        saveResults(outputPath,
                trainingClassificationResults,
                classificationResults,
                defaultSaveFilename);
    }

    public static void saveResults(String outputPath,
                                   TrainingClassificationResults trainingClassificationResults) throws Exception {
        saveResults(outputPath,
                trainingClassificationResults,
                defaultSaveFilename);
    }

    public static void saveResults(String outputPath,
                                   TrainingClassificationResults trainingClassificationResults,
                                   ClassificationResults classificationResults,
                                   String filename) throws Exception {
        OutFile outFile = new OutFile(outputPath, filename);
        outFile.writeLine("problem," + trainingClassificationResults.problem);
        outFile.writeLine("classifier," + trainingClassificationResults.classifier);
        if (trainingClassificationResults.paramId >= 0)
            outFile.writeLine("paramId," + trainingClassificationResults.paramId);

        outFile.writeLine("train_acc," + trainingClassificationResults.accuracy);
        outFile.writeLine("train_correct," + trainingClassificationResults.nbCorrect);
        outFile.writeLine("train_size," + trainingClassificationResults.trainSize);
        outFile.writeLine("train_time," + trainingClassificationResults.doTimeNs());
        outFile.writeLine("train_time_ns," + trainingClassificationResults.elapsedTimeNanoSeconds);

        outFile.writeLine("test_acc," + classificationResults.accuracy);
        outFile.writeLine("test_correct," + classificationResults.nbCorrect);
        outFile.writeLine("test_size," + classificationResults.testSize);
        outFile.writeLine("test_time," + classificationResults.doTimeNs());
        outFile.writeLine("test_time_ns," + classificationResults.elapsedTimeNanoSeconds);

        StringBuilder str = new StringBuilder("[");
        for (int i = 0; i < classificationResults.confMat.length; i++) {
            str.append("[").append(classificationResults.confMat[i][0]);
            for (int j = 1; j < classificationResults.confMat[i].length; j++) {
                str.append(":").append(classificationResults.confMat[i][j]);
            }
            str.append("]");
        }
        str.append("]");
        outFile.writeLine("test_conf_mat," + str);

        if (trainingClassificationResults.cvParams != null) {
            str = new StringBuilder("[" + trainingClassificationResults.cvParams[0]);
            for (int i = 1; i < trainingClassificationResults.cvParams.length; i++) {
                str.append(":").append(trainingClassificationResults.cvParams[i]);
            }
            str.append("]");
            outFile.writeLine("cv_param," + str);

            str = new StringBuilder("[" + trainingClassificationResults.cvAcc[0]);
            for (int i = 1; i < trainingClassificationResults.cvAcc.length; i++) {
                str.append(":").append(trainingClassificationResults.cvAcc[i]);
            }
            str.append("]");
            outFile.writeLine("cv_acc," + str);
        }
        if (trainingClassificationResults.cvPreds != null) {
            str = new StringBuilder("[");
            for (int j = 1; j < trainingClassificationResults.cvPreds.length; j++) {
                str.append("[").append(trainingClassificationResults.cvPreds[j][0]);
                for (int i = 1; i < trainingClassificationResults.cvPreds[j].length; i++) {
                    str.append(":").append(trainingClassificationResults.cvPreds[j][i]);
                }
                str.append("]");
            }
            outFile.writeLine("cvPreds," + str);
        }

        if (trainingClassificationResults.predictions != null) {
            str = new StringBuilder("[" + trainingClassificationResults.predictions[0]);
            for (int i = 1; i < trainingClassificationResults.predictions.length; i++) {
                str.append(":").append(trainingClassificationResults.predictions[i]);
            }
            str.append("]");
            outFile.writeLine("train_predictions," + str);
        }

        if (classificationResults.predictions != null) {
            str = new StringBuilder("[" + classificationResults.predictions[0]);
            for (int i = 1; i < classificationResults.predictions.length; i++) {
                str.append(":").append(classificationResults.predictions[i]);
            }
            str.append("]");
            outFile.writeLine("test_predictions," + str);
        }

        outFile.closeFile();
    }

    public static void saveResults(String outputPath,
                                   TrainingClassificationResults trainingClassificationResults,
                                   String filename) throws Exception {
        OutFile outFile = new OutFile(outputPath, filename);
        outFile.writeLine("problem," + trainingClassificationResults.problem);
        outFile.writeLine("classifier," + trainingClassificationResults.classifier);
        if (trainingClassificationResults.paramId >= 0)
            outFile.writeLine("paramId," + trainingClassificationResults.paramId);

        outFile.writeLine("train_acc," + trainingClassificationResults.accuracy);
        outFile.writeLine("train_correct," + trainingClassificationResults.nbCorrect);
        outFile.writeLine("train_size," + trainingClassificationResults.trainSize);
        outFile.writeLine("train_time," + trainingClassificationResults.doTimeNs());
        outFile.writeLine("train_time_ns," + trainingClassificationResults.elapsedTimeNanoSeconds);

        if (trainingClassificationResults.cvParams != null) {
            StringBuilder str = new StringBuilder("[" + trainingClassificationResults.cvParams[0]);
            for (int i = 1; i < trainingClassificationResults.cvParams.length; i++) {
                str.append(":").append(trainingClassificationResults.cvParams[i]);
            }
            str.append("]");
            outFile.writeLine("cv_param," + str);

            str = new StringBuilder("[" + trainingClassificationResults.cvAcc[0]);
            for (int i = 1; i < trainingClassificationResults.cvAcc.length; i++) {
                str.append(":").append(trainingClassificationResults.cvAcc[i]);
            }
            str.append("]");
            outFile.writeLine("cv_acc," + str);
        }

        StringBuilder str = new StringBuilder("[" + trainingClassificationResults.predictions[0]);
        for (int i = 1; i < trainingClassificationResults.predictions.length; i++) {
            str.append(":").append(trainingClassificationResults.predictions[i]);
        }
        str.append("]");
        outFile.writeLine("train_predictions," + str);


        outFile.closeFile();
    }


    public static void saveResultsToJSON(String outputPath,
                                         TrainingClassificationResults trainingClassificationResults,
                                         ClassificationResults classificationResults) {
        outputPath = outputPath + defaultSaveFilenameJSON;

        JSONObject tmp = new JSONObject();
        tmp.put("problem", trainingClassificationResults.problem);
        tmp.put("train_size", trainingClassificationResults.trainSize);
        tmp.put("test_size", classificationResults.testSize);
        JSONObject datasetJSON = new JSONObject();
        datasetJSON.put("dataset", tmp);

        tmp = new JSONObject();
        tmp.put("name", trainingClassificationResults.classifier);
        if (trainingClassificationResults.paramId >= 0) {
            tmp.put("paramId", trainingClassificationResults.paramId);
        }
        JSONObject classifierJSON = new JSONObject();
        classifierJSON.put("classifier", tmp);

        tmp = new JSONObject();

        tmp.put("train_acc", trainingClassificationResults.accuracy);
        tmp.put("train_correct", trainingClassificationResults.nbCorrect);
        tmp.put("test_acc", classificationResults.accuracy);
        tmp.put("test_correct", classificationResults.nbCorrect);
        StringBuilder str = new StringBuilder("\"[");
        for (int i = 0; i < classificationResults.confMat.length; i++) {
            str.append("[").append(classificationResults.confMat[i][0]);
            for (int j = 1; j < classificationResults.confMat[i].length; j++) {
                str.append(",").append(classificationResults.confMat[i][j]);
            }
            str.append("]");
        }
        str.append("]\"");
        tmp.put("test_conf_mat", str);
        if (trainingClassificationResults.cvParams != null) {
            str = new StringBuilder("[" + trainingClassificationResults.cvParams[0]);
            for (int i = 1; i < trainingClassificationResults.cvParams.length; i++) {
                str.append(",").append(trainingClassificationResults.cvParams[i]);
            }
            str.append("]");
            tmp.put("cv_param", "\"" + str + "\"");

            str = new StringBuilder("[" + trainingClassificationResults.cvAcc[0]);
            for (int i = 1; i < trainingClassificationResults.cvAcc.length; i++) {
                str.append(",").append(trainingClassificationResults.cvAcc[i]);
            }
            str.append("]");
            tmp.put("cv_acc", "\"" + str + "\"");
        }
        if (trainingClassificationResults.cvPreds != null) {
            str = new StringBuilder("[");
            for (int j = 1; j < trainingClassificationResults.cvPreds.length; j++) {
                str.append("[").append(trainingClassificationResults.cvPreds[j][0]);
                for (int i = 1; i < trainingClassificationResults.cvPreds[j].length; i++) {
                    str.append(",").append(trainingClassificationResults.cvPreds[j][i]);
                }
                str.append("]");
            }
            tmp.put("cvPreds", "\"" + str + "\"");
        }
        if (trainingClassificationResults.predictions != null) {
            str = new StringBuilder("[" + trainingClassificationResults.predictions[0]);
            for (int i = 1; i < trainingClassificationResults.predictions.length; i++) {
                str.append(",").append(trainingClassificationResults.predictions[i]);
            }
            str.append("]");
            tmp.put("train_predictions", "\"" + str + "\"");
        }
        if (classificationResults.predictions != null) {
            str = new StringBuilder("[" + classificationResults.predictions[0]);
            for (int i = 1; i < classificationResults.predictions.length; i++) {
                str.append(",").append(classificationResults.predictions[i]);
            }
            str.append("]");
            tmp.put("test_predictions", "\"" + str + "\"");
        }
        JSONObject accuracyJSON = new JSONObject();
        accuracyJSON.put("accuracy", tmp);

        tmp = new JSONObject();
        tmp.put("train_time", trainingClassificationResults.doTimeNs());
        tmp.put("train_time_ns", trainingClassificationResults.elapsedTimeNanoSeconds);
        tmp.put("test_time", classificationResults.doTimeNs());
        tmp.put("test_time_ns", classificationResults.elapsedTimeNanoSeconds);
        JSONObject durationJSON = new JSONObject();
        durationJSON.put("duration", tmp);

        JSONArray outputList = new JSONArray();
        outputList.add(datasetJSON);
        outputList.add(classifierJSON);
        outputList.add(accuracyJSON);
        outputList.add(durationJSON);

        //Write JSON file
        try (FileWriter file = new FileWriter(outputPath)) {
            //We can write any JSONArray or JSONObject instance to the file
            file.write(outputList.toJSONString());
            file.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveResultsToJSON(String outputPath,
                                         TrainingClassificationResults trainingClassificationResults) throws Exception {
        outputPath = outputPath + defaultSaveFilenameJSON;

        JSONObject tmp = new JSONObject();
        tmp.put("problem", trainingClassificationResults.problem);
        tmp.put("train_size", trainingClassificationResults.trainSize);
        JSONObject datasetJSON = new JSONObject();
        datasetJSON.put("dataset", tmp);

        tmp = new JSONObject();
        tmp.put("name", trainingClassificationResults.classifier);
        if (trainingClassificationResults.paramId >= 0) {
            tmp.put("paramId", trainingClassificationResults.paramId);
        }
        JSONObject classifierJSON = new JSONObject();
        classifierJSON.put("classifier", tmp);

        tmp = new JSONObject();

        tmp.put("train_acc", trainingClassificationResults.accuracy);
        tmp.put("train_correct", trainingClassificationResults.nbCorrect);

        if (trainingClassificationResults.cvParams != null) {
            StringBuilder str = new StringBuilder("[" + trainingClassificationResults.cvParams[0]);
            for (int i = 1; i < trainingClassificationResults.cvParams.length; i++) {
                str.append(",").append(trainingClassificationResults.cvParams[i]);
            }
            str.append("]");
            tmp.put("cv_param", "\"" + str + "\"");

            str = new StringBuilder("[" + trainingClassificationResults.cvAcc[0]);
            for (int i = 1; i < trainingClassificationResults.cvAcc.length; i++) {
                str.append(",").append(trainingClassificationResults.cvAcc[i]);
            }
            str.append("]");
            tmp.put("cv_acc", "\"" + str + "\"");
        }
        if (trainingClassificationResults.cvPreds != null) {
            StringBuilder str = new StringBuilder("[");
            for (int j = 1; j < trainingClassificationResults.cvPreds.length; j++) {
                str.append("[").append(trainingClassificationResults.cvPreds[j][0]);
                for (int i = 1; i < trainingClassificationResults.cvPreds[j].length; i++) {
                    str.append(",").append(trainingClassificationResults.cvPreds[j][i]);
                }
                str.append("]");
            }
            tmp.put("cvPreds", "\"" + str + "\"");
        }
        if (trainingClassificationResults.predictions != null) {
            StringBuilder str = new StringBuilder("[" + trainingClassificationResults.predictions[0]);
            for (int i = 1; i < trainingClassificationResults.predictions.length; i++) {
                str.append(",").append(trainingClassificationResults.predictions[i]);
            }
            str.append("]");
            tmp.put("train_predictions", "\"" + str + "\"");
        }
        JSONObject accuracyJSON = new JSONObject();
        accuracyJSON.put("accuracy", tmp);

        tmp = new JSONObject();
        tmp.put("train_time", trainingClassificationResults.doTimeNs());
        tmp.put("train_time_ns", trainingClassificationResults.elapsedTimeNanoSeconds);
        JSONObject durationJSON = new JSONObject();
        durationJSON.put("duration", tmp);

        JSONArray outputList = new JSONArray();
        outputList.add(datasetJSON);
        outputList.add(classifierJSON);
        outputList.add(accuracyJSON);
        outputList.add(durationJSON);

        //Write JSON file
        try (FileWriter file = new FileWriter(outputPath)) {
            //We can write any JSONArray or JSONObject instance to the file
            file.write(outputList.toJSONString());
            file.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static boolean isDatasetDone(String outputPath) {
        File f1 = new File(outputPath + defaultSaveFilename);
        File f2 = new File(outputPath + defaultSaveFilenameJSON);
        return f1.exists() && !f1.isDirectory() && f2.exists() && !f2.isDirectory();
    }

    public static void loadResults() {
        JSONParser jsonParser = new JSONParser();
        try (FileReader reader = new FileReader(outputPath + defaultSaveFilenameJSON)) {
            //Read JSON file
            Object obj = jsonParser.parse(reader);

            JSONArray results = (JSONArray) obj;

            for (JSONObject result : (Iterable<JSONObject>) results) {
                if (result.containsKey("accuracy")) {
                    JSONObject alist = (JSONObject) result.get("accuracy");
                    System.out.println("Train Acc: " + alist.get("train_acc"));
                    System.out.println("Test Acc: " + alist.get("test_acc"));
                } else if (result.containsKey("duration")) {
                    JSONObject alist = (JSONObject) result.get("duration");
                    System.out.println("Train Time: " + alist.get("train_time"));
                    System.out.println("Test Time: " + alist.get("test_time"));
                }
            }

        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
    }

    public static void printSummary(String moduleName) {
        System.out.println("[" + moduleName + "] DatasetPath: " + Application.datasetPath);
        System.out.println("[" + moduleName + "] Problem: " + Application.problem);
        System.out.println("[" + moduleName + "] Classifier: " + Application.classifierName);
        System.out.println("[" + moduleName + "] ParamId: " + Application.paramId);
        System.out.println("[" + moduleName + "] ZNorm: " + Application.znorm);
        System.out.println();
    }
}
