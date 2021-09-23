package classifiers.nearestNeighbour;

import classifiers.nearestNeighbour.elasticDistance.ERP;
import classifiers.nearestNeighbour.elasticDistance.lowerBounds.LbErp;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.LazyAssessNNERP;
import filters.DerivativeFilter;
import utilities.Tools;
import utils.GenericTools;

import java.util.ArrayList;
import java.util.Collections;

public class ERP1NN extends OneNearestNeighbour {
    // parameters
    private double g;                               // g value
    private double bandSize;                        // band size in terms of percentage of sequence's length

    private double[] gValues;                       // set of g values
    private double[] bandSizes;                     // set of band sizes
    private boolean gAndWindowsRefreshed = false;   // indicator if we refresh the params

    protected ERP distComputer = new ERP();
    protected LbErp lbComputer = new LbErp();

    public ERP1NN(final int paramId, final TrainOpts trainOpts) {
        this.g = 0.5;
        this.bandSize = 5;
        this.classifierIdentifier = "ERP_1NN_R1";
        this.bestParamId = paramId;
//        this.setTrainingData(trainData);
//        this.setParamsFromParamId(paramId);
        init(paramId, trainOpts);
    }

    public ERP1NN(final int paramId, final TrainOpts trainOpts, final int useDerivative) {
        this(paramId, trainOpts);
        if (useDerivative > 0)
            this.classifierIdentifier = "DERP_1NN_R1";
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
                "\n[CLASSIFIER SUMMARY] g: " + g +
                "\n[CLASSIFIER SUMMARY] band_size: " + bandSize +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final Sequence first, final Sequence second) {
        return distComputer.distance(first.data[0], second.data[0], this.g, this.bandSize);
    }

    @Override
    public double distance(final Sequence first, final Sequence second, final double cutOffValue) {
        return distComputer.distance(first.data[0], second.data[0], this.g, this.bandSize);
    }


    public double lowerBound(final Sequence candidate,
                             final int queryIndex,
                             final SequenceStatsCache cache,
                             final double cutOffValue) {
        // U and L are built on query
        return lbComputer.distance(candidate,
                cache.getUE(queryIndex, g, bandSize),
                cache.getLE(queryIndex, g, bandSize),
                cutOffValue);
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

        final LazyAssessNNERP[] lazyAssessNNS = new LazyAssessNNERP[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNNERP(cache);
        }
        final ArrayList<LazyAssessNNERP> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNNERP d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(paramId);
                final int band = distComputer.getWindowSize(train.length(), bandSize);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNNERP challenger = lazyAssessNNS[previous];
                        final LazyAssessNNERP.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.g, this.bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNNERP.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(band);
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

                    for (LazyAssessNNERP challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNNERP.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.g, this.bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNNERP.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(band);
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
                        rrt = challenger.tryToBeat(toBeat, this.g, this.bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNNERP.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(band);
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
                    final double prevG = g;
                    int w = ERP.getWindowSize(train.length(), bandSize);
                    int tmp = paramId;
                    while (tmp > 0 && paramId % 10 > 0 && prevG == g && w >= r) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();
                        tmp--;
                        this.setParamsFromParamId(tmp);
                        w = ERP.getWindowSize(train.length(), bandSize);
                    }
                }
            }
        }
    }

    @Override
    public int predictWithLb(Sequence query) {
        int[] classCounts = new int[this.trainData.getNumClasses()];

        double dist;

        Sequence candidate = trainData.get(0);
        double bsfDistance = distance(query, candidate);
        classCounts[candidate.classificationLabel]++;

        for (int candidateIndex = 1; candidateIndex < trainData.size(); candidateIndex++) {
            candidate = trainData.get(candidateIndex);
            dist = lowerBound(candidate, queryIndex, testCache, bsfDistance);
            if (dist < bsfDistance) {
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
            dist = lowerBound(candidate, queryIndex, trainCache, bsfDistance);
            if (dist < bsfDistance) {
                dist = distance(query, candidate, bsfDistance);
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
        if (!this.gAndWindowsRefreshed) {
            double stdv = GenericTools.stdv_p(trainData);
            bandSizes = Tools.getInclusive10(0, 0.25);
            gValues = Tools.getInclusive10(0.2 * stdv, stdv);
            this.gAndWindowsRefreshed = true;
        }
        this.g = gValues[paramId / 10];
        this.bandSize = bandSizes[paramId % 10];
    }

    @Override
    public String getParamInformationString() {
        return "g=" + this.g + ", bandSize=" + this.bandSize;
    }

    protected int getParamIdFromBandsize(double bandSize, int n) {
        return (int) Math.ceil(bandSize * n);
    }
}
