package experiments;

import application.Application;
import classifiers.TimeSeriesClassifier;
import datasets.DatasetLoader;
import datasets.Sequences;
import datasets.TimeSeriesDatasets;
import multiThreading.BenchmarkTask;
import multiThreading.MultiThreadedTask;
import results.ClassificationResults;
import results.TrainingClassificationResults;
import utils.StrLong;

import java.util.*;
import java.util.concurrent.Callable;

import static application.Application.extractArguments;
import static utils.GenericTools.doTimeNs;
import static utils.GenericTools.println;

public class TrainingTimeBenchmark {
    static String moduleName = "TrainingTimeBenchmark";
    private static final String[] testArgs = new String[]{
            "-machine=windows",
            "-problem=small",
            "-classifier=FastEE", // see classifiers in TimeSeriesClassifier.java
            "-paramId=-1",
            "-cpu=-1",
            "-verbose=0",
            "-iter=0",
            "-trainOpts=2",
    };

    public static void main(String[] args) throws Exception {
        final long startTime = System.nanoTime();
//        args = testArgs;
        extractArguments(args);

        if (Application.problem.equals(""))
            Application.problem = "Trace";

        Application.printSummary(moduleName);

        switch (Application.problem) {
            case "all":
            case "small": {
                String[] datasets;
                StrLong[] datasetOps;
                if (Application.problem.equals("small")) {
                    datasets = TimeSeriesDatasets.smallDatasets;
                    datasetOps = TimeSeriesDatasets.smallDatasetOperations;
                } else {
                    datasets = TimeSeriesDatasets.allDatasets;
                    datasetOps = TimeSeriesDatasets.allDatasetOperations;
                }
                Arrays.sort(datasetOps);
                long totalOp = 0;
                for (StrLong s : datasetOps) {
                    totalOp += s.value;
                }

                println("[" + moduleName + "] Number of datasets: " + datasets.length);
                println("[" + moduleName + "] Total operations: " + totalOp);

                ArrayList<String> myList = new ArrayList<>();
                Collections.addAll(myList, datasets);
                Collections.shuffle(myList, new Random(42));

                // Setup parallel training tasks
                int numThreads = Application.numThreads;
                if (numThreads <= 0) numThreads = Runtime.getRuntime().availableProcessors();
                numThreads = Math.min(numThreads, Runtime.getRuntime().availableProcessors());

                long operationPerThread = totalOp / numThreads;

                println("[" + moduleName + "] Number of threads: " + numThreads);
                println("[" + moduleName + "] Operations per thread: " + operationPerThread);

                final MultiThreadedTask parallelTasks = new MultiThreadedTask(numThreads);

                List<Callable<Integer>> tasks = new ArrayList<>();
                ArrayList<String>[] subset = new ArrayList[numThreads];
                for (int i = 0; i < numThreads; i++)
                    subset[i] = new ArrayList<>();
                int threadCount = 0;
                for (StrLong s : datasetOps) {
                    subset[threadCount].add(s.str);
                    threadCount++;
                    if (threadCount == numThreads) threadCount = 0;
                }
                for (int i = 0; i < numThreads; i++) {
                    String[] tmp = new String[subset[i].size()];
                    for (int j = 0; j < subset[i].size(); j++) {
                        tmp[j] = subset[i].get(subset[i].size() - j - 1);
                    }
                    tasks.add(new BenchmarkTask(tmp, i));
                }
                MultiThreadedTask.invokeParallelTasks(tasks, parallelTasks);
                parallelTasks.getExecutor().shutdown();
            }
            break;
            default:
                singleRun(Application.problem);
                break;
        }
        final long endTime = System.nanoTime();
        println("[" + moduleName + "] Total time taken " + doTimeNs(endTime - startTime));
    }

    /**
     * Single run of the experiments
     */
    private static void singleRun(String problem) throws Exception {
        DatasetLoader loader = new DatasetLoader();
        Sequences trainData = loader.readUCRTrain(problem, Application.datasetPath, Application.znorm).reorderClassLabels(null);
        if (Application.verbose > 1) trainData.summary();

        TimeSeriesClassifier classifier = Application.initTSC(trainData);
        if (Application.verbose > 1) println(classifier);

        if (Application.outputPath == null) {
            if (Application.paramId > 0)
                Application.outputPath = System.getProperty("user.dir") +
                        "/outputs/benchmark/" +
                        classifier.classifierIdentifier + "_" +
                        Application.paramId + "/" +
                        Application.iteration + "/" +
                        problem + "/";
            else
                Application.outputPath = System.getProperty("user.dir") +
                        "/outputs/benchmark/" +
                        classifier.classifierIdentifier + "/" +
                        Application.iteration + "/" +
                        problem + "/";
        }
        if (Application.isDatasetDone(Application.outputPath)) {
            Application.loadResults();
            return;
        }

        TrainingClassificationResults trainingResults = classifier.fit(trainData);
        trainingResults.problem = problem;
        if (Application.verbose > 1)
            println("[" + moduleName + "]" + trainingResults);

        double totalTime = trainingResults.elapsedTimeNanoSeconds;
        if (Application.doEvaluation) {
            Sequences testData = loader.readUCRTest(problem, Application.datasetPath, Application.znorm).reorderClassLabels(trainData.getInitialClassLabels());
            if (Application.verbose > 1) testData.summary();

            ClassificationResults classificationResults = classifier.evaluate(testData);
            classificationResults.problem = problem;
            if (Application.verbose == 0)
                println("[" + moduleName + "]" + classificationResults);
            else if (Application.verbose == 1)
                println("[" + moduleName + "] Problem: " + problem + ", Train Time: " + trainingResults.doTimeNs() +
                        ", Train Acc: " + trainingResults.accuracy + ", Test Acc: " + classificationResults.accuracy);
            totalTime += classificationResults.elapsedTimeNanoSeconds;
            if (Application.verbose > 1)
                println("[" + moduleName + "] Total time taken " + totalTime);

            Application.saveResults(
                    Application.outputPath,
                    trainingResults,
                    classificationResults);
            Application.saveResultsToJSON(
                    Application.outputPath,
                    trainingResults,
                    classificationResults
            );
        } else {
            if (Application.verbose == 1)
                println("[" + moduleName + "] Problem: " + problem + ", Time: " + trainingResults.doTimeNs());
            Application.saveResults(
                    Application.outputPath,
                    trainingResults);
            Application.saveResultsToJSON(
                    Application.outputPath,
                    trainingResults);
        }
    }
}
