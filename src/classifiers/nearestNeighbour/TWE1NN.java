package classifiers.nearestNeighbour;

import classifiers.nearestNeighbour.elasticDistance.TWED;
import classifiers.nearestNeighbour.elasticDistance.lowerBounds.LbTwed;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.LazyAssessNNTWE;
import filters.DerivativeFilter;

import java.util.ArrayList;
import java.util.Collections;

public class TWE1NN extends OneNearestNeighbour {
    // parameters
    public static final double[] tweNuParams = {
            // <editor-fold defaultstate="collapsed" desc="hidden for space">
            0.00001,
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1,// </editor-fold>
    };  // set of nu values
    public static final double[] tweLamdaParams = {
            // <editor-fold defaultstate="collapsed" desc="hidden for space">
            0,
            0.011111111,
            0.022222222,
            0.033333333,
            0.044444444,
            0.055555556,
            0.066666667,
            0.077777778,
            0.088888889,
            0.1,// </editor-fold>
    }; // set of lambda values
    private double nu;                                          // nu value (stiffness)
    private double lambda;                                      // lambda value

    protected TWED distComputer = new TWED();
    protected LbTwed lbComputer = new LbTwed();

    public TWE1NN(final int paramId, final TrainOpts trainOpts) {
        this.nu = 0.005;
        this.lambda = 0.5;
        this.classifierIdentifier = "TWE_1NN_R1";
        this.bestParamId = paramId;
//        this.setTrainingData(trainData);
//        this.setParamsFromParamId(paramId);
        init(paramId, trainOpts);
    }

    public TWE1NN(final int paramId, final TrainOpts trainOpts, final int useDerivative) {
        this(paramId, trainOpts);
        if (useDerivative > 0)
            this.classifierIdentifier = "DTWE_1NN_R1";
        this.useDerivative = useDerivative;
        init(paramId, trainOpts);
    }

    public void summary() {
        System.out.println(toString());
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions +
                "\n[CLASSIFIER SUMMARY] nu: " + nu +
                "\n[CLASSIFIER SUMMARY] lambda: " + lambda +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final Sequence first, final Sequence second) {
        return distComputer.distance(first.data[0], second.data[0], this.nu, this.lambda);
    }

    @Override
    public double distance(final Sequence first, final Sequence second, final double cutOffValue) {
        return distComputer.distance(first.data[0], second.data[0], this.nu, this.lambda);
    }

    public double lowerBound(final Sequence query,
                             final Sequence candidate,
                             final double queryMax,
                             final double queryMin) {
        // U and L are built on query
        return lbComputer.distance(query, candidate, queryMax, queryMin, this.nu, this.lambda);
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

        final LazyAssessNNTWE[] lazyAssessNNS = new LazyAssessNNTWE[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNNTWE(cache);
        }
        final ArrayList<LazyAssessNNTWE> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNNTWE d = lazyAssessNNS[previous];
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
                        final LazyAssessNNTWE challenger = lazyAssessNNS[previous];
                        final LazyAssessNNTWE.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.nu, this.lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNNTWE.RefineReturnType.New_best) {
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

                    for (LazyAssessNNTWE challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNNTWE.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.nu, this.lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNNTWE.RefineReturnType.New_best) {
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
                        rrt = challenger.tryToBeat(toBeat, this.nu, this.lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNNTWE.RefineReturnType.New_best) {
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
                    double d = currPNN.distance;
                    int index = currPNN.nnIndex;
                    candidateNNS[paramId][current].set(index, d, CandidateNN.Status.NN);
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
        this.trainCache = new SequenceStatsCache(trainData, trainData.get(0).length());
    }

    @Override
    public void setParamsFromParamId(final int paramId) {
        if (paramId < 0) return;

        if (paramId < 100 && this.classifierIdentifier.contains("R1")) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
        }
        this.nu = tweNuParams[paramId / 10];
        this.lambda = tweLamdaParams[paramId % 10];
    }

    @Override
    public String getParamInformationString() {
        return "nu=" + this.nu + ", lambda=" + this.lambda;
    }
}
