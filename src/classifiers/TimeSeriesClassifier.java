package classifiers;

import datasets.Sequence;
import datasets.Sequences;
import results.ClassificationResults;
import results.TrainingClassificationResults;

public abstract class TimeSeriesClassifier {
    public static String[] classifiers = new String[]{
            "DTW1NN",
            "DDTW1NN",
    };

    public enum TrainOpts {
        LOOCV0,
        LOOCV0LB,
        LOOCV,
        LOOCVLB,
        FastCV,
        ApproxCV,
    }

    public TrainingClassificationResults trainingClassificationResults;
    public ClassificationResults classificationResults;
    public Sequences trainData;
    public Sequences trainData1st;

    public String classifierIdentifier;
    public double trainingTime;

    public TrainOpts trainingOptions = TrainOpts.LOOCV0;
    public int bestParamId;
    protected int paramId;

    protected int queryIndex;

    public abstract void summary();

    public abstract TrainingClassificationResults fit(final Sequences trainData) throws Exception;

    public abstract int predict(final Sequence sequence) throws Exception;

    public ClassificationResults evaluate(Sequences testData) throws Exception {
        final int testSize = testData.size();
        int nCorrect = 0;
        int numClass = trainData.getNumClasses();
        int[][] confMat = new int[numClass][numClass];
        double[] predictions = new double[testSize];

        final long startTime = System.nanoTime();

        for (queryIndex = 0; queryIndex < testData.size(); queryIndex++) {
            Sequence query = testData.get(queryIndex);
            final int predictClass = predict(query);
            if (predictClass == query.classificationLabel) nCorrect++;
            confMat[query.classificationLabel][predictClass]++;
            predictions[queryIndex] = predictClass;
        }

        final long stopTime = System.nanoTime();

        classificationResults = new ClassificationResults(
                this.classifierIdentifier,
                this.bestParamId,
                nCorrect,
                testSize,
                startTime,
                stopTime,
                confMat,
                predictions
        );
        return classificationResults;
    }

    public void setTrainingData(final Sequences trainData) {
        this.trainData = trainData;
    }

    public void setParamsFromParamId(final int paramId) {
        this.paramId = paramId;
    }

    public abstract String getParamInformationString();
}
