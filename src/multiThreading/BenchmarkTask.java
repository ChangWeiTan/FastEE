package multiThreading;

import application.Application;
import classifiers.TimeSeriesClassifier;
import datasets.DatasetLoader;
import datasets.Sequences;
import results.ClassificationResults;
import results.TrainingClassificationResults;

import java.util.Objects;
import java.util.concurrent.Callable;

import static utils.GenericTools.println;

public class BenchmarkTask implements Callable<Integer> {
    String[] datasets;
    int threadCount;

    public BenchmarkTask(String[] datasets, int threadCount) {
        this.datasets = datasets;
        this.threadCount = threadCount;
    }

    private void singleRun(String problem) throws Exception {
        DatasetLoader loader = new DatasetLoader();
        Sequences trainData = loader.readUCRTrain(problem, Application.datasetPath, Application.znorm).reorderClassLabels(null);
        if (Application.verbose > 1) trainData.summary();

        TimeSeriesClassifier classifier = Application.initTSC(trainData);
        if (Application.verbose > 1) println(classifier);
        String outputPath;

        if (Application.paramId > 0)
            outputPath = System.getProperty("user.dir") +
                    "/outputs/benchmark/" +
                    classifier.classifierIdentifier + "_" +
                    Application.paramId + "/" +
                    Application.iteration + "/" +
                    problem + "/";
        else
            outputPath = System.getProperty("user.dir") +
                    "/outputs/benchmark/" +
                    classifier.classifierIdentifier + "/" +
                    Application.iteration + "/" +
                    problem + "/";


        if (Application.isDatasetDone(outputPath))
            return;

        TrainingClassificationResults trainingResults = classifier.fit(trainData);
        trainingResults.problem = problem;
        if (Application.verbose > 1)
            println("[Thread_" + threadCount + "]" + trainingResults);

        double totalTime = trainingResults.elapsedTimeNanoSeconds;
        if (Application.doEvaluation) {
            Sequences testData = loader.readUCRTest(problem, Application.datasetPath, Application.znorm).reorderClassLabels(trainData.getInitialClassLabels());
            if (Application.verbose > 1) testData.summary();

            ClassificationResults classificationResults = classifier.evaluate(testData);
            classificationResults.problem = problem;
            if (Application.verbose > 1)
                println("[Thread_" + threadCount + "]" + classificationResults);
            else if (Application.verbose == 0)
                println("[Thread_" + threadCount + "] Problem: " + problem + ", Train Time: " + trainingResults.doTimeNs() +
                        ", Train Acc: " + trainingResults.accuracy + ", Test Acc: " + classificationResults.accuracy);
            totalTime += classificationResults.elapsedTimeNanoSeconds;
            if (Application.verbose > 1)
                println("[Thread_" + threadCount + "] Total time taken " + totalTime);

            Application.saveResults(
                    outputPath,
                    trainingResults,
                    classificationResults);
            Application.saveResultsToJSON(
                    outputPath,
                    trainingResults,
                    classificationResults);
        } else {
            if (Application.verbose == 1)
                println("[Thread_" + threadCount + "] Problem: " + problem + ", Time: " + trainingResults.doTimeNs());
            Application.saveResults(
                    outputPath,
                    trainingResults);
            Application.saveResultsToJSON(
                    outputPath,
                    trainingResults);
        }
    }

    @Override
    public Integer call() throws Exception {
        println("[Thread_" + threadCount + "] Datasets: " + datasets.length);
        for (String dataset : datasets) {
            singleRun(dataset);
        }
        println("[Thread_" + threadCount + "] Completed all datasets");
        return null;
    }
}
