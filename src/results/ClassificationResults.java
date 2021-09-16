package results;

import utils.GenericTools;

public class ClassificationResults {
    public String problem;
    public String classifier;
    public int paramId;
    public double elapsedTimeSeconds;
    public long elapsedTimeNanoSeconds;
    public double elapsedTimeMilliSeconds;
    public double accuracy;
    public int nbCorrect;
    public int testSize;
    public int[][] confMat;
    public double[] predictions;

    public ClassificationResults(final String classifier,
                                 final int paramId,
                                 final int nbCorrect,
                                 final int testSize,
                                 final long startTimeNano,
                                 final long stopTimeNano,
                                 final int[][] confMat,
                                 final double[] predictions) {
        this.classifier = classifier;
        this.paramId = paramId;
        this.nbCorrect = nbCorrect;
        this.testSize = testSize;
        this.accuracy = 1.0 * nbCorrect / testSize;
        this.elapsedTimeNanoSeconds = stopTimeNano - startTimeNano;
        this.elapsedTimeMilliSeconds = 1.0 * this.elapsedTimeNanoSeconds / 1e6;
        this.elapsedTimeSeconds = 1.0 * this.elapsedTimeNanoSeconds / 1e9;
        this.confMat = confMat;
        this.predictions = predictions;
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
            return "ClassificationResults:" +
                    "\n\tproblem = " + problem +
                    "\n\tclassifier = " + classifier +
                    "\n\tparam_id = " + paramId +
                    "\n\tclassification_time = " + doTimeNs() +
                    "\n\tclassification_time(ns) = " + elapsedTimeNanoSeconds +
                    "\n\tclassification_accuracy = " + accuracy +
                    "\n\tnb_correct = " + nbCorrect + "/" + testSize;

        return "ClassificationResults:" +
                "\n\tclassifier = " + classifier +
                "\n\tparam_id = " + paramId +
                "\n\tclassification_time = " + doTimeNs() +
                "\n\tclassification_time(ns) = " + elapsedTimeNanoSeconds +
                "\n\tclassification_accuracy = " + accuracy +
                "\n\tnb_correct = " + nbCorrect + "/" + testSize;
    }
}
