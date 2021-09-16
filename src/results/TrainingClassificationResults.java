package results;

import utils.GenericTools;

public class TrainingClassificationResults {
    public String problem;
    public String classifier;
    public double elapsedTimeSeconds;
    public long elapsedTimeNanoSeconds;
    public double elapsedTimeMilliSeconds;
    public double accuracy;
    public int nbCorrect;
    public int trainSize;
    public double[] predictions;
    public int[] cvParams;
    public double[] cvAcc;
    public double[] cvTime;
    public double[][] cvPreds;

    public int paramId = -1;

    public TrainingClassificationResults(final String classifier,
                                         final long elapsedTimeNanoSeconds,
                                         final int trainSize) {
        this.classifier = classifier;
        this.accuracy = -1;
        this.elapsedTimeNanoSeconds = elapsedTimeNanoSeconds;
        this.elapsedTimeMilliSeconds = 1.0 * this.elapsedTimeNanoSeconds / 1e6;
        this.elapsedTimeSeconds = 1.0 * this.elapsedTimeNanoSeconds / 1e9;
        this.trainSize = trainSize;
        this.nbCorrect = -1;
    }

    public TrainingClassificationResults(final String classifier,
                                         final double accuracy,
                                         final long elapsedTimeNanoSeconds,
                                         final double[] predictions) {
        this.classifier = classifier;
        this.accuracy = accuracy;
        this.elapsedTimeNanoSeconds = elapsedTimeNanoSeconds;
        this.elapsedTimeMilliSeconds = 1.0 * this.elapsedTimeNanoSeconds / 1e6;
        this.elapsedTimeSeconds = 1.0 * this.elapsedTimeNanoSeconds / 1e9;
        this.predictions = predictions.clone();
        this.trainSize = predictions.length;
        this.nbCorrect = (int) (accuracy * trainSize);
    }

    public TrainingClassificationResults(final String classifier,
                                         final double accuracy,
                                         final long startTimeNano,
                                         final long stopTimeNano,
                                         final double[] predictions) {
        this.classifier = classifier;
        this.accuracy = accuracy;
        this.elapsedTimeNanoSeconds = stopTimeNano - startTimeNano;
        this.elapsedTimeMilliSeconds = 1.0 * this.elapsedTimeNanoSeconds / 1e6;
        this.elapsedTimeSeconds = 1.0 * this.elapsedTimeNanoSeconds / 1e9;
        this.predictions = predictions.clone();
        this.trainSize = predictions.length;
        this.nbCorrect = (int) (accuracy * trainSize);
    }

    public String doTime() {
        return GenericTools.doTime(this.elapsedTimeNanoSeconds);
    }

    public String doTimeNs() {
        return GenericTools.doTimeNs(this.elapsedTimeNanoSeconds);
    }

    @Override
    public String toString() {
        if (this.problem != null)
            return "TrainingClassificationResults:" +
                    "\n\tproblem = " + problem +
                    "\n\tclassifier = " + classifier +
                    "\n\tparam_id = " + paramId +
                    "\n\ttraining_time = " + doTimeNs() +
                    "\n\ttraining_time(ns) = " + elapsedTimeNanoSeconds +
                    "\n\ttraining_accuracy = " + accuracy +
                    "\n\tnb_correct = " + nbCorrect + "/" + trainSize;

        return "TrainingClassificationResults:" +
                "\n\tclassifier = " + classifier +
                "\n\tparam_id = " + paramId +
                "\n\ttraining_time = " + doTimeNs() +
                "\n\ttraining_time(ns) = " + elapsedTimeNanoSeconds +
                "\n\ttraining_accuracy = " + accuracy +
                "\n\tnb_correct = " + nbCorrect + "/" + trainSize;
    }
}
