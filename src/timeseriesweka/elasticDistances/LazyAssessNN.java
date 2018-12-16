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
package timeseriesweka.elasticDistances;

import timeseriesweka.fastWWS.SequenceStatsCache;
import weka.core.Instance;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Original code from https://github.com/ChangWeiTan/FastWWSearch
 *
 * Lazy Assess NN
 */
public class LazyAssessNN implements Comparable<LazyAssessNN> {
    public int indexQuery;
    public int indexReference;         // Index for query and reference
    private SequenceStatsCache cache;               // Cache to store the information for the sequences
    private Instance query, reference;              // Query and reference sequences
    private int indexStoppedLB, oldIndexStoppedLB;  // Index where we stop LB
    private int currentW;                           // Current warping window for DTW
    private int minWindowValidityFullDTW;           // Minimum window validity for DTW, ERP, LCSS
    private int nOperationsLBKim;                   // Number of operations for LB Kim
    private double minDist;                         // distance
    private double bestMinDist;                     // best so far distance
    private double EuclideanDist;                   // euclidean distance
    private LBStatus status;                        // Status of Lower Bound
    private double[] currentWeightVector;           // weight vector for WDTW
    private double currentC;                        // parameter for MSM
    private double currentG, currentBandSize;       // parameters for ERP
    private double currentNu, currentLambda;        // parameters for TWED
    private double currentEpsilon;                  // parameter for LCSS
    private int currentDelta;                       // parameter for LCSS

    public LazyAssessNN(Instance query, int index, Instance reference, int indexReference, SequenceStatsCache cache) {
        if (index < indexReference) {
            this.query = query;
            this.indexQuery = index;
            this.reference = reference;
            this.indexReference = indexReference;
        } else {
            this.query = reference;
            this.indexQuery = indexReference;
            this.reference = query;
            this.indexReference = index;
        }
        this.minDist = 0.0;
        this.cache = cache;
        tryLBKim();
        this.bestMinDist = minDist;
        this.status = LBStatus.LB_Kim;
    }

    public LazyAssessNN(SequenceStatsCache cache) {
        this.cache = cache;
    }

    public void set(Instance query, int index, Instance reference, int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        currentW = 0;
        minWindowValidityFullDTW = 0;
        nOperationsLBKim = 0;
        // --- From constructor
        if (index < indexReference) {
            this.query = query;
            this.indexQuery = index;
            this.reference = reference;
            this.indexReference = indexReference;
        } else {
            this.query = reference;
            this.indexQuery = indexReference;
            this.reference = query;
            this.indexReference = index;
        }
        this.minDist = 0.0;
        tryLBKim();
        this.bestMinDist = minDist;
        this.status = LBStatus.LB_Kim;
    }

    public void setAsItIs(Instance query, int index, Instance reference, int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        currentW = 0;
        minWindowValidityFullDTW = 0;
        nOperationsLBKim = 0;
        this.query = query;
        this.indexQuery = index;
        this.reference = reference;
        this.indexReference = indexReference;
        this.minDist = 0.0;
        tryLBKim();
        this.bestMinDist = minDist;
        this.status = LBStatus.LB_Kim;
    }

    public void setWoutKimAsItIs(Instance query, int index, Instance reference, int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        // --- From constructor
        this.query = query;
        this.indexQuery = index;
        this.reference = reference;
        this.indexReference = indexReference;
        this.minDist = 0.0;
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
    }

    public void setWoutKim(Instance query, int index, Instance reference, int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        // --- From constructor
        if (index < indexReference) {
            this.query = query;
            this.indexQuery = index;
            this.reference = reference;
            this.indexReference = indexReference;
        } else {
            this.query = reference;
            this.indexQuery = indexReference;
            this.reference = query;
            this.indexReference = index;
        }
        this.minDist = 0.0;
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
    }

    public void setForTWED(Instance query, int index, Instance reference, int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        // --- From constructor
        if (index < indexReference) {
            this.query = query;
            this.indexQuery = index;
            this.reference = reference;
            this.indexReference = indexReference;
        } else {
            this.query = reference;
            this.indexQuery = indexReference;
            this.reference = query;
            this.indexReference = index;
        }
        this.minDist = 0.0;
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
        this.cache.initLbDistances();
    }

    public void setBestMinDist(double bestMinDist) {
        this.bestMinDist = bestMinDist;
    }

    public double getBestMinDist() {
        return this.bestMinDist;
    }

    public void setCurrentW(final int currentW) {
        if (this.currentW != currentW) {
            this.currentW = currentW;
            if (status == LBStatus.Full_DTW) {
                if (this.currentW >= minWindowValidityFullDTW) {
                    this.status = LBStatus.Full_DTW;
                } else {
                    this.status = LBStatus.Previous_DTW;
                }
            } else {
                this.status = LBStatus.Previous_LB_DTW;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    private void setCurrentWeightVector(final double[] weightVector) {
        this.currentWeightVector = weightVector;
        if (status == LBStatus.Full_WDTW) {
            this.status = LBStatus.Previous_WDTW;
        } else {
            this.status = LBStatus.Previous_LB_WDTW;
            this.oldIndexStoppedLB = indexStoppedLB;
        }
    }

    private void setCurrentC(final double c) {
        if (this.currentC != c) {
            this.currentC = c;
            if (status == LBStatus.Full_MSM) {
                this.status = LBStatus.Previous_MSM;
            } else {
                this.status = LBStatus.Previous_LB_MSM;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    private void setCurrentGandBandSize(final double g, final double bandSize) {
        if (this.currentG != g) {
            this.currentBandSize = bandSize;
            this.currentG = g;
            this.minDist = 0.0;
            this.bestMinDist = minDist;
            indexStoppedLB = oldIndexStoppedLB = 0;
            this.status = LBStatus.Previous_G_LB_ERP;
        } else if (this.currentBandSize != bandSize) {
            this.currentBandSize = bandSize;
            if (status == LBStatus.Full_ERP) {
                if (this.currentBandSize >= minWindowValidityFullDTW) {
                    this.status = LBStatus.Full_ERP;
                } else {
                    this.status = LBStatus.Previous_Band_ERP;
                }
            } else {
                this.status = LBStatus.Previous_Band_LB_ERP;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    private void setCurrentNuAndLambda(final double nu, final double lambda) {
        if (this.currentNu != nu) {
            this.currentLambda = lambda;
            this.currentNu = nu;
            this.minDist = 0.0;
            this.bestMinDist = minDist;
            indexStoppedLB = oldIndexStoppedLB = 0;
            this.status = LBStatus.Previous_LB_TWE;
        } else if (this.currentLambda != lambda) {
            this.currentLambda = lambda;
            if (status == LBStatus.Full_TWE) {
                this.status = LBStatus.Previous_TWE;
            } else {
                this.status = LBStatus.Previous_LB_TWE;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    private void setCurrentDeltaAndEpsilon(final int delta, final double epsilon) {
        if (this.currentEpsilon != epsilon) {
            this.currentEpsilon = epsilon;
            this.currentDelta = delta;
            this.minDist = 0.0;
            this.bestMinDist = minDist;
            indexStoppedLB = oldIndexStoppedLB = 0;
            this.status = LBStatus.Previous_LB_LCSS;
        } else if (this.currentDelta != delta) {
            this.currentDelta = delta;
            if (status == LBStatus.Full_LCSS) {
                if (this.currentDelta >= minWindowValidityFullDTW) {
                    this.status = LBStatus.Full_LCSS;
                } else {
                    this.status = LBStatus.Previous_LCSS;
                }
            } else {
                this.status = LBStatus.Previous_LB_LCSS;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    public RefineReturnType tryEuclidean(final double scoreToBeat) {
        if (bestMinDist >= scoreToBeat) {
            return RefineReturnType.Pruned_with_LB;
        }
        if (EuclideanDist >= scoreToBeat) {
            return RefineReturnType.Pruned_with_Dist;
        }
        EuclideanDist = 0;
        for (int i = query.numAttributes() - 2; i >= 0; i--) {
            final double dist = query.value(i) - reference.value(i);
            EuclideanDist += dist * dist;
        }
        return RefineReturnType.New_best;
    }

    private void tryLBKim() {
        double diffFirsts = query.value(0) - reference.value(0);
        double diffLasts = query.value(query.numAttributes() - 2) - reference.value(reference.numAttributes() - 2);
        minDist = diffFirsts * diffFirsts + diffLasts * diffLasts;
        nOperationsLBKim = 2;
        if (!cache.isMinFirst(indexQuery) && !cache.isMinFirst(indexReference) && !cache.isMinLast(indexQuery) && !cache.isMinLast(indexReference)) {
            final double diffMin = cache.getMin(indexQuery) - cache.getMin(indexReference);
            minDist += diffMin * diffMin;
            nOperationsLBKim++;
        }
        if (!cache.isMaxFirst(indexQuery) && !cache.isMaxFirst(indexReference) && !cache.isMaxLast(indexQuery) && !cache.isMaxLast(indexReference)) {
            final double diffMax = cache.getMax(indexQuery) - cache.getMax(indexReference);
            minDist += diffMax * diffMax;
            nOperationsLBKim++;
        }

        status = LBStatus.LB_Kim;
    }

    /*------------------------------------------------------------------------------------------------------------------
        Lower bounds for DTW
     -----------------------------------------------------------------------------------------------------------------*/
    private void tryContinueLBKeoghQR(final double scoreToBeat) {
        final int length = query.numAttributes() - 1;
        final double[] LEQ = cache.getDTWLE(indexQuery, currentW);
        final double[] UEQ = cache.getDTWUE(indexQuery, currentW);
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < LEQ[index]) {
                final double diff = LEQ[index] - c;
                minDist += diff * diff;
            } else if (UEQ[index] < c) {
                final double diff = UEQ[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    private void tryContinueLBKeoghRQ(final double scoreToBeat) {
        final int length = reference.numAttributes() - 1;
        final double[] LER = cache.getDTWLE(indexReference, currentW);
        final double[] UER = cache.getDTWUE(indexReference, currentW);
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < LER[index]) {
                final double diff = LER[index] - c;
                minDist += diff * diff;
            } else if (UER[index] < c) {
                final double diff = UER[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    private void tryFullLBKeoghQR() {
        final int length = query.numAttributes() - 1;
        final double[] LEQ = cache.getDTWLE(indexQuery, currentW);
        final double[] UEQ = cache.getDTWUE(indexQuery, currentW);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < LEQ[index]) {
                final double diff = LEQ[index] - c;
                minDist += diff * diff;
            } else if (UEQ[index] < c) {
                final double diff = UEQ[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    private void tryFullLBKeoghRQ() {
        final int length = reference.numAttributes() - 1;
        final double[] LER = cache.getDTWLE(indexReference, currentW);
        final double[] UER = cache.getDTWUE(indexReference, currentW);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < LER[index]) {
                final double diff = LER[index] - c;
                minDist += diff * diff;
            } else if (UER[index] < c) {
                final double diff = UER[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    /*------------------------------------------------------------------------------------------------------------------
        Lower bounds for WDTW
     -----------------------------------------------------------------------------------------------------------------*/
    private void tryLBKimWDTW() {
        final double diffFirsts = query.value(0) - reference.value(0);
        final double diffLasts = query.value(query.numAttributes() - 2) - reference.value(reference.numAttributes() - 2);
        minDist = diffFirsts * diffFirsts + diffLasts * diffLasts;
        minDist *= currentWeightVector[0];
        nOperationsLBKim = 2;
        if (!cache.isMinFirst(indexQuery) && !cache.isMinFirst(indexReference) && !cache.isMinLast(indexQuery) && !cache.isMinLast(indexReference)) {
            final double diffMin = cache.getMin(indexQuery) - cache.getMin(indexReference);
            minDist += diffMin * diffMin * currentWeightVector[Math.abs(cache.getIMin(indexQuery) - cache.getIMin(indexReference))];
            nOperationsLBKim++;
        }
        if (!cache.isMaxFirst(indexQuery) && !cache.isMaxFirst(indexReference) && !cache.isMaxLast(indexQuery) && !cache.isMaxLast(indexReference)) {
            final double diffMax = cache.getMax(indexQuery) - cache.getMax(indexReference);
            minDist += diffMax * diffMax * currentWeightVector[Math.abs(cache.getIMin(indexQuery) - cache.getIMin(indexReference))];
            nOperationsLBKim++;
        }
        status = LBStatus.LB_Kim;
    }

    private void tryContinueLBWDTWQR(final double scoreToBeat) {
        final int length = query.numAttributes() - 1;
        final double QMAX = cache.getMax(indexQuery);
        final double QMIN = cache.getMin(indexQuery);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < QMIN) {
                final double diff = QMIN - c;
                minDist += diff * diff * currentWeightVector[0];
            } else if (QMAX < c) {
                final double diff = QMAX - c;
                minDist += diff * diff * currentWeightVector[0];
            }
            indexStoppedLB++;
        }
    }

    private void tryContinueLBWDTWRQ(final double scoreToBeat) {
        final int length = reference.numAttributes() - 1;
        final double QMAX = cache.getMax(indexReference);
        final double QMIN = cache.getMin(indexReference);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < QMIN) {
                final double diff = QMIN - c;
                minDist += diff * diff * currentWeightVector[0];
            } else if (QMAX < c) {
                final double diff = QMAX - c;
                minDist += diff * diff * currentWeightVector[0];
            }
            indexStoppedLB++;
        }
    }

    private void tryFullLBWDTWQR() {
        final int length = query.numAttributes() - 1;
        final double QMAX = cache.getMax(indexQuery);
        final double QMIN = cache.getMin(indexQuery);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < QMIN) {
                final double diff = QMIN - c;
                minDist += diff * diff;
            } else if (QMAX < c) {
                final double diff = QMAX - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
        this.minDist *= currentWeightVector[0];
    }

    private void tryFullLBWDTWRQ() {
        final int length = reference.numAttributes() - 1;
        final double QMAX = cache.getMax(indexReference);
        final double QMIN = cache.getMin(indexReference);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < QMIN) {
                final double diff = QMIN - c;
                minDist += diff * diff;
            } else if (QMAX < c) {
                final double diff = QMAX - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
        this.minDist *= currentWeightVector[0];
    }

    /*------------------------------------------------------------------------------------------------------------------
        Lower bounds for MSM
     -----------------------------------------------------------------------------------------------------------------*/
    private void tryContinueLBMSM(final double scoreToBeat) {
        final int length = query.numAttributes() - 1;
        final double QMAX = cache.getMax(indexQuery);
        final double QMIN = cache.getMin(indexQuery);
        this.minDist = Math.abs(query.value(0) - reference.value(0));
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist < scoreToBeat) {
            int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (index > 0 && index < length - 1) {
                final double curr = reference.value(index);
                final double prev = reference.value(index - 1);
                if (prev <= curr && curr < QMIN) {
                    minDist += Math.min(Math.abs(reference.value(index) - QMIN), this.currentC);
                } else if (prev >= curr && curr >= QMAX) {
                    minDist += Math.min(Math.abs(reference.value(index) - QMAX), this.currentC);
                }
            }
            indexStoppedLB++;
        }
    }

    private void tryFullLBMSM() {
        final int length = query.numAttributes() - 1;
        final double QMAX = cache.getMax(indexQuery);
        final double QMIN = cache.getMin(indexQuery);
        this.minDist = Math.abs(query.value(0) - reference.value(0));
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (index > 0 && index < length - 1) {
                final double curr = reference.value(index);
                final double prev = reference.value(index - 1);
                if (prev <= curr && curr < QMIN) {
                    minDist += Math.min(Math.abs(reference.value(index) - QMIN), this.currentC);
                } else if (prev >= curr && curr >= QMAX) {
                    minDist += Math.min(Math.abs(reference.value(index) - QMAX), this.currentC);
                }
            }
            indexStoppedLB++;
        }
    }

    /*------------------------------------------------------------------------------------------------------------------
        Lower bounds for ERP
     -----------------------------------------------------------------------------------------------------------------*/
    private void tryContinueLBERPQR(final double scoreToBeat) {
        final int length = query.numAttributes() - 1;
        final double[] LEQ = cache.getERPLE(indexQuery, currentG, currentBandSize);
        final double[] UEQ = cache.getERPUE(indexQuery, currentG, currentBandSize);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < LEQ[index]) {
                final double diff = LEQ[index] - c;
                minDist += diff * diff;
            } else if (UEQ[index] < c) {
                final double diff = UEQ[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    private void tryContinueLBERPRQ(final double scoreToBeat) {
        final int length = reference.numAttributes() - 1;
        final double[] LER = cache.getERPLE(indexReference, currentG, currentBandSize);
        final double[] UER = cache.getERPUE(indexReference, currentG, currentBandSize);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < LER[index]) {
                final double diff = LER[index] - c;
                minDist += diff * diff;
            } else if (UER[index] < c) {
                final double diff = UER[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    private void tryFullLBERPQR() {
        final int length = query.numAttributes() - 1;
        final double[] LEQ = cache.getERPLE(indexQuery, currentG, currentBandSize);
        final double[] UEQ = cache.getERPUE(indexQuery, currentG, currentBandSize);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < LEQ[index]) {
                final double diff = LEQ[index] - c;
                minDist += diff * diff;
            } else if (UEQ[index] < c) {
                final double diff = UEQ[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    private void tryFullLBERPRQ() {
        final int length = reference.numAttributes() - 1;
        final double[] LER = cache.getERPLE(indexReference, currentG, currentBandSize);
        final double[] UER = cache.getERPUE(indexReference, currentG, currentBandSize);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < LER[index]) {
                final double diff = LER[index] - c;
                minDist += diff * diff;
            } else if (UER[index] < c) {
                final double diff = UER[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    /*------------------------------------------------------------------------------------------------------------------
        Lower bounds for TWED
     -----------------------------------------------------------------------------------------------------------------*/
    private void tryContinueLBTWED(final double scoreToBeat) {
        final int length = query.numAttributes() - 1;
        final double q0 = query.value(0);
        final double c0 = reference.value(0);
        double diff = q0 - c0;
        this.minDist = Math.min(diff * diff,
                Math.min(q0 * q0 + currentNu + currentLambda,
                        c0 * c0 + currentNu + currentLambda));
        this.indexStoppedLB = 1;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (index > 0) {
                final double curr = reference.value(index);
                final double prev = reference.value(index - 1);
                final double max = Math.max(cache.getMax(indexQuery), prev);
                final double min = Math.min(cache.getMin(indexQuery), prev);
                if (curr < min) {
                    diff = min - curr;
                    this.minDist += Math.min(currentNu, diff * diff);
                } else if (max < curr) {
                    diff = max - curr;
                    this.minDist += Math.min(currentNu, diff * diff);
                }
            }
            indexStoppedLB++;
        }
    }

    private void tryFullLBTWED() {
        final int length = query.numAttributes() - 1;
        final double q0 = query.value(0);
        final double c0 = reference.value(0);
        double diff = q0 - c0;
        this.minDist = Math.min(diff * diff,
                Math.min(q0 * q0 + currentNu + currentLambda,
                        c0 * c0 + currentNu + currentLambda));
        this.indexStoppedLB = 1;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (index > 0) {
                final double curr = reference.value(index);
                final double prev = reference.value(index - 1);
                final double max = Math.max(cache.getMax(indexQuery), prev);
                final double min = Math.min(cache.getMin(indexQuery), prev);
                if (curr < min) {
                    diff = min - curr;
                    this.minDist += Math.min(currentNu, diff * diff);
                } else if (max < curr) {
                    diff = max - curr;
                    this.minDist += Math.min(currentNu, diff * diff);
                }
            }
            indexStoppedLB++;
        }
    }

    /*------------------------------------------------------------------------------------------------------------------
        Lower bounds for LCSS
     -----------------------------------------------------------------------------------------------------------------*/
    private void tryContinueLBLCSS(final double scoreToBeat) {
        final int length = query.numAttributes() - 1;
        final double ub = Math.abs(1.0 - scoreToBeat) * length;
        final double[] LEQ = cache.getLCSSLE(indexQuery, currentDelta, currentEpsilon);
        final double[] UEQ = cache.getLCSSUE(indexQuery, currentDelta, currentEpsilon);
        double lcs = 0;
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && lcs > ub) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (reference.value(index) <= UEQ[index] && reference.value(index) >= LEQ[index]) {
                lcs++;
            }
            indexStoppedLB++;
        }
        this.minDist = 1 - lcs / length;
    }

    private void tryFullLBLCSS() {
        final int length = query.numAttributes() - 1;
        final double[] LEQ = cache.getLCSSLE(indexQuery, currentDelta, currentEpsilon);
        final double[] UEQ = cache.getLCSSUE(indexQuery, currentDelta, currentEpsilon);
        double lcs = 0;
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (reference.value(index) <= UEQ[index] && reference.value(index) >= LEQ[index]) {
                lcs++;
            }
            indexStoppedLB++;
        }
        this.minDist = 1 - lcs / length;
    }

    /*------------------------------------------------------------------------------------------------------------------
        Try to beat DTW
     -----------------------------------------------------------------------------------------------------------------*/
    public RefineReturnType tryToBeatDTW(final double scoreToBeat, final int w) {
        setCurrentW(w);
        switch (status) {
            case Previous_LB_DTW:
            case Previous_DTW:
            case LB_Kim:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_KeoghQR:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBKeoghQR(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_KeoghQR;
                    else status = LBStatus.Full_LB_KeoghQR;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_KeoghQR;
            case Full_LB_KeoghQR:
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_KeoghRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBKeoghRQ(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < reference.numAttributes() - 1) status = LBStatus.Partial_LB_KeoghRQ;
                    else status = LBStatus.Full_LB_KeoghRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_KeoghRQ;
            case Full_LB_KeoghRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final DistanceResults res = DTW.distanceExt(query, reference, currentW);
                minDist = res.distance;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_DTW;
                minWindowValidityFullDTW = res.r;
            case Full_DTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    /*------------------------------------------------------------------------------------------------------------------
        Try to beat WDTW
     -----------------------------------------------------------------------------------------------------------------*/
    public RefineReturnType tryToBeatWDTW(final double scoreToBeat, final double[] weightVector) {
        setCurrentWeightVector(weightVector);
        switch (status) {
            case None:
            case Previous_LB_WDTW:
            case Previous_WDTW:
                if (bestMinDist * weightVector[0] >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_WDTWQR:
                tryContinueLBWDTWQR(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_WDTWQR;
                    else status = LBStatus.Full_LB_WDTWQR;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_WDTWQR;
            case Full_LB_WDTWQR:
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_WDTWRQ:
                tryContinueLBWDTWRQ(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_WDTWRQ;
                    else status = LBStatus.Full_LB_WDTWRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_WDTWRQ;
            case Full_LB_WDTWRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                minDist = WDTW.distance(query, reference, weightVector);
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_WDTW;
            case Full_WDTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    /*------------------------------------------------------------------------------------------------------------------
        Try to beat MSM
     -----------------------------------------------------------------------------------------------------------------*/
    public RefineReturnType tryToBeatMSM(final double scoreToBeat, final double c) {
        setCurrentC(c);
        switch (status) {
            case None:
            case Previous_LB_MSM:
            case Previous_MSM:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_MSM:
                tryContinueLBMSM(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_MSM;
                    else status = LBStatus.Full_LB_MSM;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_MSM;
            case Full_LB_MSM:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                minDist = MSM.distance1(query, reference, currentC);
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_MSM;
            case Full_MSM:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    /*------------------------------------------------------------------------------------------------------------------
        Try to beat ERP
     -----------------------------------------------------------------------------------------------------------------*/
    public RefineReturnType tryToBeatERP(final double scoreToBeat, final double g, final double bandSize) {
        setCurrentGandBandSize(g, bandSize);
        switch (status) {
            case Previous_G_LB_ERP:
            case Previous_Band_LB_ERP:
            case Previous_Band_ERP:
            case LB_Kim:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_ERPQR:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBERPQR(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_ERPQR;
                    else status = LBStatus.Full_LB_ERPQR;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_ERPQR;
            case Full_LB_ERPQR:
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_ERPRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBERPRQ(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < reference.numAttributes() - 1) status = LBStatus.Partial_LB_ERPRQ;
                    else status = LBStatus.Full_LB_ERPRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_ERPRQ;
            case Full_LB_ERPRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final DistanceResults res = ERP.distanceExt(query, reference, currentG, currentBandSize);
                minDist = res.distance;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_ERP;
                minWindowValidityFullDTW = res.r;
            case Full_ERP:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    /*------------------------------------------------------------------------------------------------------------------
        Try to beat TWED
     -----------------------------------------------------------------------------------------------------------------*/
    public RefineReturnType tryToBeatTWED(final double scoreToBeat, final double nu, final double lambda) {
        setCurrentNuAndLambda(nu, lambda);
        switch (status) {
            case None:
            case Previous_LB_TWE:
            case Previous_TWE:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_TWE:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBTWED(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_TWE;
                    else status = LBStatus.Full_LB_TWE;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_TWE;
            case Full_LB_TWE:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                minDist = TWED.distance(query, reference, nu, lambda);
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_TWE;
            case Full_TWE:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    /*------------------------------------------------------------------------------------------------------------------
       Try to beat LCSS
    -----------------------------------------------------------------------------------------------------------------*/
    public RefineReturnType tryToBeatLCSS(final double scoreToBeat, final int delta, final double epsilon) {
        setCurrentDeltaAndEpsilon(delta, epsilon);
        switch (status) {
            case None:
            case Previous_LB_LCSS:
            case Previous_LCSS:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_LCSS:
                tryFullLBLCSS();
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist > scoreToBeat) {
                    if (indexStoppedLB < query.numAttributes() - 1) status = LBStatus.Partial_LB_LCSS;
                    else status = LBStatus.Full_LB_LCSS;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_LCSS;
            case Full_LB_LCSS:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final DistanceResults res = LCSS.distanceExt(query, reference, epsilon, delta);
                minDist = res.distance;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_LCSS;
                minWindowValidityFullDTW = res.r;
            case Full_LCSS:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    @Override
    public String toString() {
        return "" + indexQuery + " - " + indexReference + " - " + bestMinDist;
    }

    public int getOtherIndex(int index) {
        if (index == indexQuery) {
            return indexReference;
        } else {
            return indexQuery;
        }
    }

    public Instance getSequenceForOtherIndex(int index) {
        if (index == indexQuery) {
            return reference;
        } else {
            return query;
        }
    }

    public double getDistance(int window) {
        if ((status == LBStatus.Full_DTW ||
                status == LBStatus.Full_ERP ||
                status == LBStatus.Full_LCSS)
                && minWindowValidityFullDTW <= window) {
            return minDist;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is argMin3 valid already-computed Distance");
    }

    public double getMinDist() {
        return minDist;
    }

    public void setMinDist(double minDist) {
        this.minDist = minDist;
    }

    public double getDistance() {
        if (status == LBStatus.Full_WDTW ||
                status == LBStatus.Full_MSM ||
                status == LBStatus.Previous_MSM ||
                status == LBStatus.Full_TWE) {
            return minDist;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is argMin3 valid already-computed Distance");
    }


    public int getMinWindowValidityForFullDistance() {
        if (status == LBStatus.Full_DTW ||
                status == LBStatus.Full_ERP ||
                status == LBStatus.Full_LCSS) {
            return minWindowValidityFullDTW;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is argMin3 valid already-computed Distance");
    }

    public int getMinwindow() {
        return minWindowValidityFullDTW;
    }

    public void setMinwindow(int w) {
        minWindowValidityFullDTW = w;
    }

    public double getEuclideanDistance() {
        return EuclideanDist;
    }

    @Override
    public int compareTo(LazyAssessNN o) {
        return this.compare(o);

    }

    private int compare(LazyAssessNN o) {
        double num1 = this.getDoubleValueForRanking();
        double num2 = o.getDoubleValueForRanking();
        return Double.compare(num1, num2);
    }

    private double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        switch (status) {
            // DTW
            case Full_DTW:
            case Full_LB_KeoghQR:
            case Full_LB_KeoghRQ:
                return thisD / (query.numAttributes() - 1);
            case LB_Kim:
                return thisD / nOperationsLBKim;
            case Partial_LB_KeoghQR:
            case Partial_LB_KeoghRQ:
                return thisD / indexStoppedLB;
            case Previous_DTW:
                return 0.8 * thisD / (query.numAttributes() - 1);    // DTWDistance(w+1) should be tighter
            case Previous_LB_DTW:
                if (indexStoppedLB == 0) {
                    //lb kim
                    return thisD / nOperationsLBKim;
                } else {
                    //lbkeogh
                    return thisD / oldIndexStoppedLB;
                }

            // WDTW
            case Full_WDTW:
            case Full_LB_WDTWQR:
            case Full_LB_WDTWRQ:
                return thisD / (query.numAttributes() - 1);
            case Partial_LB_WDTWQR:
            case Partial_LB_WDTWRQ:
                return thisD / indexStoppedLB;
            case Previous_WDTW:
                return 0.8 * thisD / (query.numAttributes() - 1);
            case Previous_LB_WDTW:
                return thisD / oldIndexStoppedLB;

            // MSM
            case Full_MSM:
            case Full_LB_MSM:
                return thisD / (query.numAttributes() - 1);
            case Partial_LB_MSM:
                return thisD / indexStoppedLB;
            case Previous_MSM:
                return 0.8 * thisD / (query.numAttributes() - 1);
            case Previous_LB_MSM:
                return thisD / oldIndexStoppedLB;

            // ERP
            case Full_ERP:
            case Full_LB_ERPQR:
            case Full_LB_ERPRQ:
                return thisD / (query.numAttributes() - 1);
            case Partial_LB_ERPQR:
            case Partial_LB_ERPRQ:
                return thisD / indexStoppedLB;
            case Previous_Band_ERP:
                return 0.8 * thisD / (query.numAttributes() - 1);
            case Previous_G_LB_ERP:
            case Previous_Band_LB_ERP:
                if (indexStoppedLB == 0) {
                    //lb kim
                    return thisD / nOperationsLBKim;
                } else {
                    //lbkeogh
                    return thisD / oldIndexStoppedLB;
                }

            // TWE
            case Full_TWE:
            case Full_LB_TWE:
                return thisD / (query.numAttributes() - 1);
            case Partial_LB_TWE:
                return thisD / indexStoppedLB;
            case Previous_TWE:
                return 0.8 * thisD / (query.numAttributes() - 1);
            case Previous_LB_TWE:
                return thisD / oldIndexStoppedLB;

            // LCSS
            case Full_LCSS:
            case Full_LB_LCSS:
                return thisD / (query.numAttributes() - 1);
            case Partial_LB_LCSS:
                return thisD / indexStoppedLB;
            case Previous_LCSS:
                return 0.8 * thisD / (query.numAttributes() - 1);
            case Previous_LB_LCSS:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }

    @Override
    public boolean equals(Object o) {
        LazyAssessNN d = (LazyAssessNN) o;
        return (this.indexQuery == d.indexQuery && this.indexReference == d.indexReference);
    }

    public LBStatus getStatus() {
        return status;
    }

    public void setFullDistStatus() {
        this.status = LBStatus.Full_DTW;
    }

    public void setFullWDTWStatus() {
        this.status = LBStatus.Full_WDTW;
    }

    public void setFullERPStatus() {
        this.status = LBStatus.Full_ERP;
    }

    public void setFullMSMStatus() {
        this.status = LBStatus.Full_MSM;
    }

    public void setFullLCSSStatus() {
        this.status = LBStatus.Full_LCSS;
    }

    public double getBestLB() {
        return bestMinDist;
    }

    public Instance getQuery() {
        return query;
    }

    public Instance getReference() {
        return reference;
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Internal types
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    public enum RefineReturnType {
        Pruned_with_LB, Pruned_with_Dist, New_best
    }

    public enum LBStatus {
        None, LB_Kim,
        Partial_LB_KeoghQR, Full_LB_KeoghQR, Partial_LB_KeoghRQ, Full_LB_KeoghRQ,   // DTW
        Previous_LB_DTW, Previous_DTW, Full_DTW, Partial_DTW,                       // DTW
        Partial_LB_WDTWQR, Partial_LB_WDTWRQ, Full_LB_WDTWQR, Full_LB_WDTWRQ,       // WDTW
        Previous_LB_WDTW, Previous_WDTW, Full_WDTW,                                 // WDTW
        Partial_LB_MSM, Full_LB_MSM, Previous_LB_MSM, Previous_MSM, Full_MSM,       // MSM
        Partial_LB_ERPQR, Partial_LB_ERPRQ, Full_LB_ERPQR, Full_LB_ERPRQ,           // ERP
        Previous_G_LB_ERP, Previous_Band_LB_ERP, Previous_Band_ERP, Full_ERP,       // ERP
        Partial_LB_TWE, Full_LB_TWE, Previous_LB_TWE, Previous_TWE, Full_TWE,       // TWE
        Partial_LB_LCSS, Full_LB_LCSS, Previous_LB_LCSS, Previous_LCSS, Full_LCSS   // LCSS
    }
}
