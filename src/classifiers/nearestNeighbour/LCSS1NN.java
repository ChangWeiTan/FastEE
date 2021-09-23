package classifiers.nearestNeighbour;

import classifiers.nearestNeighbour.elasticDistance.LCSS;
import classifiers.nearestNeighbour.elasticDistance.lowerBounds.LbLcss;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.LazyAssessNNLCSS;
import filters.DerivativeFilter;
import utils.GenericTools;

import java.util.ArrayList;
import java.util.Collections;

public class LCSS1NN extends OneNearestNeighbour {
    // parameters
    private int delta;
    private double epsilon;
    private double[] epsilons;
    private int[] deltas;
    private boolean epsilonsAndDeltasRefreshed;

    protected LCSS distComputer = new LCSS();
    protected LbLcss lbComputer = new LbLcss();

    public LCSS1NN(final int paramId, final TrainOpts trainOpts) {
        this.delta = 3;
        this.epsilon = 1;
        this.classifierIdentifier = "LCSS_1NN_R1";
        this.bestParamId = paramId;
//        this.setTrainingData(trainData);
//        this.setParamsFromParamId(paramId);
        init(paramId, trainOpts);
    }

    public LCSS1NN(final int paramId, final TrainOpts trainOpts, final int useDerivative) {
        this(paramId, trainOpts);
        if (useDerivative > 0)
            this.classifierIdentifier = "DLCSS_1NN_R1";;
        this.useDerivative = useDerivative;
        init(paramId, trainOpts);
    }

    public void summary() {
        System.out.println(this);
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions +
                "\n[CLASSIFIER SUMMARY] delta: " + this.delta +
                "\n[CLASSIFIER SUMMARY] epsilon: " + this.epsilon +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final Sequence first, final Sequence second) {
        return distComputer.distance(first.data[0], second.data[0], this.epsilon, this.delta);
    }

    @Override
    public double distance(final Sequence first, final Sequence second, final double cutOffValue) {
        return distComputer.distance(first.data[0], second.data[0], this.epsilon, this.delta);
    }

    public double lowerBound(final Sequence candidate,
                             final int queryIndex,
                             final SequenceStatsCache cache) {
        // U and L are built on query
        return lbComputer.distance(candidate,
                cache.getUE(queryIndex, delta, epsilon),
                cache.getLE(queryIndex, delta, epsilon));
    }

    @Override
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
                if (classCounts[paramId][i][c] > bsfCount) {
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

    @Override
    public void initNNSTable(final Sequences train, final SequenceStatsCache cache) {
        if (train.size() < 2) {
            System.err.println("[INIT-NNS-TABLE] Set is too small: " + train.size() + " sequence. At least 2 sequences needed.");
        }

        candidateNNS = new CandidateNN[nParams][train.size()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.size(); ++len) {
                candidateNNS[paramId][len] = new CandidateNN();
            }
        }
        classCounts = new int[nParams][train.size()][train.getNumClasses()];

        final LazyAssessNNLCSS[] lazyAssessNNS = new LazyAssessNNLCSS[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNNLCSS(cache);
        }
        final ArrayList<LazyAssessNNLCSS> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNNLCSS d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(paramId);

                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNNLCSS challenger = lazyAssessNNS[previous];
                        final LazyAssessNNLCSS.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.delta, this.epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNNLCSS.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(this.delta);
                            prevNN.set(current, r, d, CandidateNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.getNumClasses()];
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            }
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have the NN yet.
                    // Sort the challengers so we have the better chance to organize the good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNNLCSS challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNNLCSS.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.delta, this.epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNNLCSS.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(this.delta);
                            currPNN.set(previous, r, d, CandidateNN.Status.BC);
                            if (d < toBeat) {
                                classCounts[paramId][current] = new int[train.getNumClasses()];
                                classCounts[paramId][current][challenger.getQuery().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][current][challenger.getQuery().classificationLabel]++;
                            }
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeat(toBeat, this.delta, this.epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNNLCSS.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(this.delta);
                            prevNN.set(current, r, d, CandidateNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.getNumClasses()];
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            }
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    final int r = currPNN.r;
                    final double d = currPNN.distance;
                    final int index = currPNN.nnIndex;
                    final double prevEpsilon = epsilon;
                    int tmp = paramId;
                    while (tmp > 0 && paramId % 10 > 0 && prevEpsilon == epsilon && delta >= r) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();
                        tmp--;
                        this.setParamsFromParamId(tmp);
                    }
                }
            }
        }
    }

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

    @Override
    public int predictWithLb(Sequence query) throws Exception {
        int[] classCounts = new int[this.trainData.getNumClasses()];

        double dist;

        Sequence candidate = trainData.get(0);
        double bsfDistance = distance(query, candidate);
        classCounts[candidate.classificationLabel]++;

        for (int candidateIndex = 1; candidateIndex < trainData.size(); candidateIndex++) {
            candidate = trainData.get(candidateIndex);
            dist = lowerBound(candidate, queryIndex, testCache);
            if (dist <= bsfDistance) {
                dist = distance(query, candidate);
                if (dist < bsfDistance) {
                    bsfDistance = dist;
                    classCounts = new int[trainData.getNumClasses()];
                    classCounts[candidate.classificationLabel]++;
                } else if (dist == bsfDistance) {
                    classCounts[candidate.classificationLabel]++;
                }
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

    @Override
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

    @Override
    public int classifyLoocvLB(final Sequence query, final int queryIndex) {
        int[] classCounts = new int[this.trainData.getNumClasses()];

        double dist;

        int candidateIndex = (queryIndex > 0) ? 0 : 1;
        int nextIndex = candidateIndex + 1;
        Sequence candidate = this.trainData.get(candidateIndex);
        double bsfDistance = distance(query, candidate);
        classCounts[candidate.classificationLabel]++;

        for (candidateIndex = nextIndex; candidateIndex < this.trainData.size(); candidateIndex++) {
            if (queryIndex == candidateIndex)
                continue;
            candidate = this.trainData.get(candidateIndex);
            dist = lowerBound(candidate, queryIndex, trainCache);
            if (dist <= bsfDistance) {
                dist = distance(query, candidate);
                if (dist < bsfDistance) {
                    bsfDistance = dist;
                    classCounts = new int[this.trainData.getNumClasses()];
                    classCounts[candidate.classificationLabel]++;
                } else if (dist == bsfDistance) {
                    classCounts[candidate.classificationLabel]++;
                }
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

    @Override
    public void setTrainingData(final Sequences trainData) {
        if (useDerivative > 0) {
            if (!trainDer) {
                this.trainData = DerivativeFilter.getFirstDerivative(trainData);
                this.trainDer = true;
            }
        } else
            this.trainData = trainData;
        this.trainCache = new SequenceStatsCache(this.trainData, this.trainData.get(0).length());
    }

    @Override
    public void setParamsFromParamId(final int paramId) {
        if (paramId < 0) return;

        if (paramId < 100 && this.classifierIdentifier.contains("R1")) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
        }
        if (!epsilonsAndDeltasRefreshed) {
            double stdTrain = GenericTools.stdv_p(trainData);
            double stdFloor = stdTrain * 0.2;
            epsilons = GenericTools.getInclusive10(stdFloor, stdTrain);
            deltas = GenericTools.getInclusive10(0, (trainData.length()) / 4);
            epsilonsAndDeltasRefreshed = true;
        }
        this.delta = deltas[paramId % 10];
        this.epsilon = epsilons[paramId / 10];
    }

    @Override
    public String getParamInformationString() {
        return "delta=" + this.delta + ", epsilon=" + this.epsilon;
    }

    protected int getParamIdFromBandsize(double bandSize, int n) {
        return (int) Math.ceil(bandSize * n);
    }
}
