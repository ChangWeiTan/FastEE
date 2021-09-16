package classifiers.nearestNeighbour;

import classifiers.nearestNeighbour.elasticDistance.MSM;
import classifiers.nearestNeighbour.elasticDistance.lowerBounds.LbMsm;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.LazyAssessNNMSM;
import filters.DerivativeFilter;

import java.util.ArrayList;
import java.util.Collections;

public class MSM1NN extends OneNearestNeighbour {
    // parameters
    private static double[] msmParams = {
            // <editor-fold defaultstate="collapsed" desc="hidden for space">
            0.01,
            0.01375,
            0.0175,
            0.02125,
            0.025,
            0.02875,
            0.0325,
            0.03625,
            0.04,
            0.04375,
            0.0475,
            0.05125,
            0.055,
            0.05875,
            0.0625,
            0.06625,
            0.07,
            0.07375,
            0.0775,
            0.08125,
            0.085,
            0.08875,
            0.0925,
            0.09625,
            0.1,
            0.136,
            0.172,
            0.208,
            0.244,
            0.28,
            0.316,
            0.352,
            0.388,
            0.424,
            0.46,
            0.496,
            0.532,
            0.568,
            0.604,
            0.64,
            0.676,
            0.712,
            0.748,
            0.784,
            0.82,
            0.856,
            0.892,
            0.928,
            0.964,
            1,
            1.36,
            1.72,
            2.08,
            2.44,
            2.8,
            3.16,
            3.52,
            3.88,
            4.24,
            4.6,
            4.96,
            5.32,
            5.68,
            6.04,
            6.4,
            6.76,
            7.12,
            7.48,
            7.84,
            8.2,
            8.56,
            8.92,
            9.28,
            9.64,
            10,
            13.6,
            17.2,
            20.8,
            24.4,
            28,
            31.6,
            35.2,
            38.8,
            42.4,
            46,
            49.6,
            53.2,
            56.8,
            60.4,
            64,
            67.6,
            71.2,
            74.8,
            78.4,
            82,
            85.6,
            89.2,
            92.8,
            96.4,
            100// </editor-fold>
    };  // MSM pre-defined c values
    private double c = 0;                               // c value

    protected MSM distComputer = new MSM();
    protected LbMsm lbComputer = new LbMsm();

    public MSM1NN(final int paramId, final TrainOpts trainOpts) {
        this.classifierIdentifier = "MSM_1NN_R1";
        this.bestParamId = paramId;
//        this.setTrainingData(trainData);
//        this.setParamsFromParamId(paramId);
        init(paramId, trainOpts);
    }

    public MSM1NN(final int paramId, final TrainOpts trainOpts, final int useDerivative) {
        this(paramId, trainOpts);
        if (useDerivative > 0)
            this.classifierIdentifier = "DMSM_1NN_R1";
        this.useDerivative = useDerivative;
    }

    public void summary() {
        System.out.println(this);
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions +
                "\n[CLASSIFIER SUMMARY] c: " + c +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final Sequence first, final Sequence second) {
        return distComputer.distance(first.data[0], second.data[0], this.c);
    }

    @Override
    public double distance(final Sequence first, final Sequence second, final double cutOffValue) {
        return distComputer.distance(first.data[0], second.data[0], this.c, cutOffValue);
    }

    public double lowerBound(Sequence query, Sequence candidate, double queryMax, double queryMin) {
        return lbComputer.distance(query, candidate, this.c, queryMax, queryMin);
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

        final LazyAssessNNMSM[] lazyAssessNNS = new LazyAssessNNMSM[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNNMSM(cache);
        }
        final ArrayList<LazyAssessNNMSM> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNNMSM d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = 0; paramId < nParams; ++paramId) {
                setParamsFromParamId(paramId);

                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNNMSM challenger = lazyAssessNNS[previous];
                        final LazyAssessNNMSM.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.c);

                        // --- Check the result
                        if (rrt == LazyAssessNNMSM.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            prevNN.set(current, d, CandidateNN.Status.NN);
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

                    for (LazyAssessNNMSM challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNNMSM.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.c);

                        // --- Check the result
                        if (rrt == LazyAssessNNMSM.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            currPNN.set(previous, d, CandidateNN.Status.BC);
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
                        rrt = challenger.tryToBeat(toBeat, this.c);

                        // --- Check the result
                        if (rrt == LazyAssessNNMSM.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            prevNN.set(current, d, CandidateNN.Status.NN);
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
                    if (paramId > 0) {
                        double currD = currPNN.distance;
                        double prevD = candidateNNS[paramId - 1][current].distance;
                        int index = currPNN.nnIndex;
                        if (currD == prevD) {
                            for (int j = paramId; j < nParams; j++) {
                                candidateNNS[j][current].set(index, currD, CandidateNN.Status.NN);
                                classCounts[j][current] = classCounts[paramId][current].clone();
                            }
                        }
                    }
                }
            }
        }
    }

    @Override
    public int predictWithLb(Sequence query) throws Exception {
        return predict(query);
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
        this.c = msmParams[paramId];
    }

    @Override
    public String getParamInformationString() {
        return "c=" + this.c;
    }
}
