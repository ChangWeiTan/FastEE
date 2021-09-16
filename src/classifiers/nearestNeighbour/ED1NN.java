/* Copyright (C) 2018 Chang Wei Tan, Francois Petitjean, Geoff Webb
 This file is part of FastEE.
 FastEE is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, version 3 of the License.
 LbEnhanced is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with LbEnhanced.  If not, see <http://www.gnu.org/licenses/>. */
package classifiers.nearestNeighbour;

import classifiers.TimeSeriesClassifier;
import classifiers.nearestNeighbour.elasticDistance.Euclidean;
import classifiers.nearestNeighbour.elasticDistance.lowerBounds.LbKim;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.SequenceStatsCache;
import filters.DerivativeFilter;

import static classifiers.TimeSeriesClassifier.TrainOpts.LOOCV;
import static classifiers.TimeSeriesClassifier.TrainOpts.LOOCV0;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 * <p>
 * NN-ED Classifier
 */
public class ED1NN extends OneNearestNeighbour {
    protected Euclidean distComputer = new Euclidean();
    protected LbKim lbComputer = new LbKim();

    public ED1NN(final TrainOpts trainOpts) {
        this.classifierIdentifier = "Euclidean_1NN";
        if (trainOpts == LOOCV) this.trainingOptions = LOOCV0;
        else this.trainingOptions = TimeSeriesClassifier.TrainOpts.LOOCV0LB;
//        this.setTrainingData(trainData);
    }

    public ED1NN(final TrainOpts trainOpts, final int useDerivative) {
        this(trainOpts);
        if (useDerivative > 0)
            this.classifierIdentifier = "DEuclidean_1NN";
        this.useDerivative = useDerivative;
    }

    public void summary() {
        System.out.println(this);
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions;
    }

    @Override
    public double distance(final Sequence first, final Sequence second) {
        return distComputer.distance(first.data[0], second.data[0]);
    }

    @Override
    public double distance(final Sequence first, final Sequence second, final double cutOffValue) {
        return distComputer.distance(first.data[0], second.data[0], cutOffValue);
    }

    public double lowerBound(final Sequence query, final Sequence candidate,
                             final SequenceStatsCache queryCache, final SequenceStatsCache candidateCache,
                             final int queryIndex, final int candidateIndex) {
        return lbComputer.distance(query, candidate, queryCache, candidateCache, queryIndex, candidateIndex);
    }


    @Override
    public void initNNSTable(Sequences trainData, SequenceStatsCache cache) {

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
            dist = lowerBound(query, candidate, testCache, trainCache, queryIndex, candidateIndex);
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
    public int classifyLoocvLB(final Sequence query, final int queryIndex) throws Exception {
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
            dist = lowerBound(query, candidate, trainCache, trainCache, queryIndex, candidateIndex);
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
    public void setParamsFromParamId(int paramId) {
    }

    @Override
    public String getParamInformationString() {
        return "NoParams";
    }

}
