package classifiers.nearestNeighbour;

import application.Application;
import classifiers.TimeSeriesClassifier;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import filters.DerivativeFilter;
import results.ClassificationResults;
import results.TrainingClassificationResults;

import static classifiers.TimeSeriesClassifier.TrainOpts.LOOCV;
import static classifiers.TimeSeriesClassifier.TrainOpts.LOOCV0;

public abstract class OneNearestNeighbour extends TimeSeriesClassifier {
    public SequenceStatsCache trainCache; // cache for training set
    public SequenceStatsCache testCache; // cache for test set

    final int nParams = 100;
    int maxWindow;
    int[][][] classCounts;
    int useDerivative = 0;

    public boolean trainDer = false;
    public boolean useLBAtPrediction = true;

    CandidateNN[][] candidateNNS;

    protected void init(final int paramId, TrainOpts trainOpts) {
        if (paramId < 0) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
            this.trainingOptions = trainOpts;
        } else {
            if (trainOpts == LOOCV) this.trainingOptions = LOOCV0;
            else this.trainingOptions = TimeSeriesClassifier.TrainOpts.LOOCV0LB;
        }
    }

    public abstract double distance(final Sequence first,
                                    final Sequence second) throws Exception;

    public abstract double distance(final Sequence first,
                                    final Sequence second,
                                    final double cutOffValue) throws Exception;

    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);

        switch (this.trainingOptions) {
            case LOOCV0:
                return loocv0(this.trainData);
            case LOOCV:
                return loocv(this.trainData);
            case LOOCV0LB:
                return loocv0LB(this.trainData);
            case LOOCVLB:
                return loocvLB(this.trainData);
            case FastCV:
                return fastParameterSearch(this.trainData);
//            case ApproxCV:
//                return approxParameterSearch(this.trainData, this.nSamples);
        }
        return loocv0(this.trainData);
    }

    public abstract void initNNSTable(final Sequences trainData, final SequenceStatsCache cache) throws Exception;

    @Override
    public int predict(final Sequence query) throws Exception {
        int[] classCounts = new int[this.trainData.getNumClasses()];

        double dist;

        Sequence candidate = trainData.get(0);
        double bsfDistance = distance(query, candidate);
        classCounts[candidate.classificationLabel]++;

        for (int candidateIndex = 1; candidateIndex < trainData.size(); candidateIndex++) {
            candidate = trainData.get(candidateIndex);
            dist = distance(query, candidate);
            if (dist < bsfDistance) {
                bsfDistance = dist;
                classCounts = new int[trainData.getNumClasses()];
                classCounts[candidate.classificationLabel]++;
            } else if (dist == bsfDistance) {
                classCounts[candidate.classificationLabel]++;
            }
        }

        int bsfClass = -1;
        double bsfCount = -1;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > bsfCount) {
                bsfCount = classCounts[i];
                bsfClass = i;
            }
        }
        return bsfClass;
    }

    public int predictWithLb(final Sequence query) throws Exception {
        return predict(query);
    }

    @Override
    public ClassificationResults evaluate(Sequences testData) throws Exception {
        if (useDerivative > 0)
            testData = DerivativeFilter.getFirstDerivative(testData);

        testCache = new SequenceStatsCache(testData, testData.get(0).length());

        final int testSize = testData.size();
        int nCorrect = 0;
        int numClass = trainData.getNumClasses();
        int[][] confMat = new int[numClass][numClass];
        double[] predictions = new double[testSize];

        final long startTime = System.nanoTime();

        for (queryIndex = 0; queryIndex < testData.size(); queryIndex++) {
            Sequence query = testData.get(queryIndex);
            int predictClass;
            if (useLBAtPrediction) predictClass = predictWithLb(query);
            else predictClass = predict(query);
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

    /**
     * Do Leave-One-Out CV with the default parameter ID
     *
     * @param train training dataset
     * @return training results
     */
    public TrainingClassificationResults loocv0(final Sequences train) throws Exception {
        double[] accAndPreds;
        double bsfAcc = -1;
        final double[] predictions = new double[train.size()];

        if (Application.verbose > 1)
            System.out.println("[1-NN] LOOCV0 for " + this.classifierIdentifier);

        final long start = System.nanoTime();
        accAndPreds = loocvAccAndPreds(train, this.bestParamId);
        if (accAndPreds[0] > bsfAcc) {
            bsfAcc = accAndPreds[0];
            System.arraycopy(accAndPreds, 1, predictions, 0, train.size());
        }
        final long end = System.nanoTime();
        trainingTime = 1.0 * (end - start) / 1e9;

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier,
                bsfAcc, start, end, predictions
        );
        results.paramId = bestParamId;

        if (Application.verbose > 0)
            System.out.printf("[" + this.classifierIdentifier + "] LOOCV0 Results: %s, Acc=%.5f, Time=%s%n",
                    getParamInformationString(), bsfAcc, results.doTime());

        return results;
    }

    /**
     * Do Leave-One-Out CV with the default parameter ID and lower bound
     *
     * @param train training dataset
     * @return training results
     */
    public TrainingClassificationResults loocv0LB(final Sequences train) throws Exception {
        double[] accAndPreds;
        double bsfAcc = -1;
        final double[] predictions = new double[train.size()];

        if (Application.verbose > 1)
            System.out.println("[1-NN] LOOCV0 for " + this.classifierIdentifier);

        final long start = System.nanoTime();
        accAndPreds = loocvAccAndPredsLB(train, bestParamId);
        if (accAndPreds[0] > bsfAcc) {
            bsfAcc = accAndPreds[0];
            System.arraycopy(accAndPreds, 1, predictions, 0, train.size());
        }
        final long end = System.nanoTime();
        trainingTime = 1.0 * (end - start) / 1e9;

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier,
                bsfAcc, start, end, predictions);
        results.paramId = bestParamId;

        if (Application.verbose > 0)
            System.out.printf("[" + this.classifierIdentifier + "] LOOCV0 Results: %s, Acc=%.5f, Time=%s%n",
                    getParamInformationString(), bsfAcc, results.doTime());

        return results;
    }

    /**
     * Do Leave-One-Out CV
     *
     * @param train training dataset
     * @return training results
     */
    public TrainingClassificationResults loocv(final Sequences train) throws Exception {
        double[] accAndPreds;
        int[] cvParams = new int[nParams];
        double[] cvAcc = new double[nParams];
        bestParamId = -1;
        double bsfAcc = -1;
        double[] predictions = new double[train.size()];

        if (Application.verbose > 1) {
            System.out.println("[1-NN] LOOCV for " + this.classifierIdentifier + ", training");
            System.out.print("loocv_acc = [");
        }

        final long start = System.nanoTime();

        for (int paramId = 0; paramId < nParams; paramId++) {
            cvParams[paramId] = paramId;
            if (Application.verbose > 1)
                System.out.print(".");
            accAndPreds = loocvAccAndPreds(train, paramId);

            if (Application.verbose > 2)
                System.out.print(accAndPreds[0] + ",");
            cvAcc[paramId] = accAndPreds[0];
            if (accAndPreds[0] > bsfAcc) {
                bestParamId = paramId;
                bsfAcc = accAndPreds[0];
                System.arraycopy(accAndPreds, 1, predictions, 0, train.size());
            }
        }
        final long end = System.nanoTime();
        if (Application.verbose > 2)
            System.out.println("];");
        trainingTime = 1.0 * (end - start) / 1e9;

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier,
                bsfAcc, start, end, predictions);
        results.paramId = bestParamId;
        results.cvAcc = cvAcc;
        results.cvParams = cvParams;

        this.setTrainingData(train);
        this.setParamsFromParamId(bestParamId);
        if (Application.verbose > 0)
            System.out.printf("[" + this.classifierIdentifier + "] LOOCV Results: ParamID:=%d, %s, Acc=%.5f, Time=%s%n",
                    bestParamId, getParamInformationString(), bsfAcc, results.doTime());

        return results;
    }

    /**
     * Do Leave-One-Out CV with lower bound
     *
     * @param train training dataset
     * @return training results
     */
    public TrainingClassificationResults loocvLB(final Sequences train) throws Exception {
        double[] accAndPreds;
        int[] cvParams = new int[nParams];
        double[] cvAcc = new double[nParams];

        bestParamId = -1;
        double bsfAcc = -1;
        double[] predictions = new double[train.size()];

        if (Application.verbose > 1) {
            System.out.println("[1-NN] LOOCV for " + this.classifierIdentifier + ", training");
            System.out.print("loocv_acc = [");
        }

        final long start = System.nanoTime();
        for (int paramId = 0; paramId < nParams; paramId++) {
            cvParams[paramId] = paramId;
            if (Application.verbose > 1)
                System.out.print(".");
            accAndPreds = loocvAccAndPredsLB(train, paramId);

            if (Application.verbose > 2)
                System.out.print(accAndPreds[0] + ",");
            cvAcc[paramId] = accAndPreds[0];
            if (accAndPreds[0] > bsfAcc) {
                bestParamId = paramId;
                bsfAcc = accAndPreds[0];
                System.arraycopy(accAndPreds, 1, predictions, 0, train.size());
            }
        }
        final long end = System.nanoTime();
        if (Application.verbose > 2)
            System.out.println("];");
        trainingTime = 1.0 * (end - start) / 1e9;

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier,
                bsfAcc, start, end, predictions);
        results.paramId = bestParamId;
        results.cvAcc = cvAcc;
        results.cvParams = cvParams;

        this.setTrainingData(train);
        this.setParamsFromParamId(bestParamId);
        if (Application.verbose > 0)
            System.out.printf("[" + this.classifierIdentifier + "] LOOCV Results: ParamID:=%d, %s, Acc=%.5f, Time=%s%n",
                    bestParamId, getParamInformationString(), bsfAcc, results.doTime());

        return results;
    }

    /**
     * Do FastWWSearch
     *
     * @param train training dataset
     * @return training results
     */
    public TrainingClassificationResults fastParameterSearch(final Sequences train) throws Exception {
        bestParamId = -1;
        double bsfAcc = -1;
        double[] accAndPreds;
        int[] cvParams = new int[nParams];
        double[] cvAcc = new double[nParams];
        double[] predictions = new double[train.size()];
        this.maxWindow = train.length();

        if (Application.verbose > 1)
            System.out.println("[1-NN] Fast Parameter Search for " + this.classifierIdentifier + ", training ");

        final long start = System.nanoTime();
        if (Application.verbose > 1)
            System.out.println("[" + this.classifierIdentifier + "] Initialising NNs table for Fast Parameter Search");

        initNNSTable(train, trainCache);

        if (Application.verbose > 1) {
            System.out.print("fastcv_acc = [");
        }
        for (int paramId = 0; paramId < nParams; paramId++) {
            cvParams[paramId] = paramId;
            if (Application.verbose > 1)
                System.out.print(".");
            accAndPreds = fastParameterSearchAccAndPred(train, paramId, train.size());

            if (Application.verbose > 2)
                System.out.print(accAndPreds[0] + ",");
            cvAcc[paramId] = accAndPreds[0];
            if (accAndPreds[0] > bsfAcc) {
                bsfAcc = accAndPreds[0];
                bestParamId = paramId;
                System.arraycopy(accAndPreds, 1, predictions, 0, train.size());
            }
        }
        final long end = System.nanoTime();
        if (Application.verbose > 2)
            System.out.println("];");
        trainingTime = 1.0 * (end - start) / 1e9;

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier, bsfAcc, start, end, predictions);
        results.paramId = bestParamId;
        results.cvAcc = cvAcc;
        results.cvParams = cvParams;

        this.setTrainingData(train);
        this.setParamsFromParamId(bestParamId);
        if (Application.verbose > 0)
            System.out.printf("[" + this.classifierIdentifier + "] Fast Parameter Search Results: ParamID:=%d, %s, Acc=%.5f, Time=%s%n",
                    bestParamId, getParamInformationString(), bsfAcc, results.doTime());

        return results;
    }

    protected double[] loocvAccAndPreds(final Sequences train, final int paramId) throws Exception {
        this.setParamsFromParamId(paramId);

        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[train.size() + 1];
        for (int i = 0; i < train.size(); i++) {
            final Sequence query = train.get(i);
            actual = query.classificationLabel;
            pred = this.classifyLoocv(query, i);
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = (double) correct / train.size();

        return accAndPreds;
    }

    protected double[] loocvAccAndPredsLB(final Sequences train, final int paramId) throws Exception {
        this.setParamsFromParamId(paramId);

        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[train.size() + 1];
        for (int i = 0; i < train.size(); i++) {
            final Sequence query = train.get(i);
            actual = query.classificationLabel;
            pred = this.classifyLoocvLB(query, i);
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = (double) correct / train.size();

        return accAndPreds;
    }

    protected double[] fastParameterSearchAccAndPred(final Sequences train, final int paramId, final int n) {
        this.setParamsFromParamId(paramId);
        int correct = 0;
        double pred, actual;

        final double[] accAndPreds = new double[n + 1];
        for (int i = 0; i < n; i++) {
            actual = train.get(i).classificationLabel;
            pred = -1;
            double bsfCount = -1;
            for (int c = 0; c < classCounts[paramId][i].length; c++) {
                if (classCounts[paramId][i][c] >= bsfCount) {
                    bsfCount = classCounts[paramId][i][c];
                    pred = c;
                }
            }

            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = 1.0 * correct / n;

        return accAndPreds;
    }

    public int classifyLoocv(final Sequence query, final int queryIndex) throws Exception {
        int[] classCounts = new int[this.trainData.getNumClasses()];

        double dist;

        int candidateIndex = (queryIndex > 0) ? 0 : 1;
        int nextIndex = candidateIndex + 1;
        Sequence candidate = trainData.get(candidateIndex);
        double bsfDistance = distance(query, candidate);
        classCounts[candidate.classificationLabel]++;

        for (candidateIndex = nextIndex; candidateIndex < trainData.size(); candidateIndex++) {
            if (queryIndex == candidateIndex)
                continue;
            candidate = trainData.get(candidateIndex);
            dist = distance(query, candidate, bsfDistance);
            if (dist < bsfDistance) {
                bsfDistance = dist;
                classCounts = new int[trainData.getNumClasses()];
                classCounts[candidate.classificationLabel]++;
            } else if (dist == bsfDistance) {
                classCounts[candidate.classificationLabel]++;
            }
        }

        int bsfClass = -1;
        double bsfCount = -1;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > bsfCount) {
                bsfCount = classCounts[i];
                bsfClass = i;
            }
        }
        return bsfClass;
    }

    public int classifyLoocvLB(final Sequence query, final int queryIndex) throws Exception {
        return classifyLoocv(query, queryIndex);
    }
}
