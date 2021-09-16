package fastWWS.assessNN;

import classifiers.nearestNeighbour.elasticDistance.WDTW;
import datasets.Sequence;
import fastWWS.SequenceStatsCache;

public class LazyAssessNNWDTW extends LazyAssessNN {
    private final WDTW distComputer = new WDTW();

    public LazyAssessNNWDTW(final SequenceStatsCache cache) {
        super(cache);
    }

    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference) {
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

    private void setCurrentWeightVector(final double[] weightVector) {
        this.currentWeightVector = weightVector;
        if (status == LBStatus.Full_WDTW) {
            this.status = LBStatus.Previous_WDTW;
        } else {
            this.status = LBStatus.Previous_LB_WDTW;
            this.oldIndexStoppedLB = indexStoppedLB;
        }
    }

    private void tryContinueLBWDTWQR(final double scoreToBeat) {
        final int length = query.length();
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
        final int length = reference.length();
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


    public RefineReturnType tryToBeat(final double scoreToBeat, final double[] weightVector) {
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
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_WDTWQR;
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
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_WDTWRQ;
                    else status = LBStatus.Full_LB_WDTWRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_WDTWRQ;
            case Full_LB_WDTWRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                minDist = distComputer.distance(query.data[0], reference.data[0], weightVector);
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_WDTW;
            case Full_WDTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        // WDTW
        switch (status) {
            case Full_WDTW:
            case Full_LB_WDTWQR:
            case Full_LB_WDTWRQ:
                return thisD / query.length();
            case Partial_LB_WDTWQR:
            case Partial_LB_WDTWRQ:
                return thisD / indexStoppedLB;
            case Previous_WDTW:
                return 0.8 * thisD / query.length();
            case Previous_LB_WDTW:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
