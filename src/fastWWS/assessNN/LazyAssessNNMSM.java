package fastWWS.assessNN;

import classifiers.nearestNeighbour.elasticDistance.MSM;
import datasets.Sequence;
import fastWWS.SequenceStatsCache;

public class LazyAssessNNMSM extends LazyAssessNN {
    private final MSM distComputer = new MSM();

    public LazyAssessNNMSM(final SequenceStatsCache cache) {
        super(cache);
    }

    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        currentC = 0;
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

    private void tryContinueLBMSM(final double scoreToBeat) {
        final int length = query.length();
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

    public RefineReturnType tryToBeat(final double scoreToBeat, final double c) {
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
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_MSM;
                    else status = LBStatus.Full_LB_MSM;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_MSM;
            case Full_LB_MSM:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                minDist = distComputer.distance(query.data[0], reference.data[0], currentC);
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_MSM;
            case Full_MSM:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        // ERP
        switch (status) {
            // MSM
            case Full_MSM:
            case Full_LB_MSM:
                return thisD / query.length();
            case Partial_LB_MSM:
                return thisD / indexStoppedLB;
            case Previous_MSM:
                return 0.8 * thisD / query.length();
            case Previous_LB_MSM:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
