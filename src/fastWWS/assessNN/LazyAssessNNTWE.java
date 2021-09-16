package fastWWS.assessNN;

import classifiers.nearestNeighbour.elasticDistance.TWED;
import datasets.Sequence;
import fastWWS.SequenceStatsCache;

public class LazyAssessNNTWE extends LazyAssessNN {
    private final TWED distComputer = new TWED();

    public LazyAssessNNTWE(final SequenceStatsCache cache) {
        super(cache);
    }

    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        this.currentNu = 0;
        this.currentLambda = 0;
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

    private void tryContinueLBTWED(final double scoreToBeat) {
        final int length = query.length();
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

    public RefineReturnType tryToBeat(final double scoreToBeat, final double nu, final double lambda) {
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
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_TWE;
                    else status = LBStatus.Full_LB_TWE;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_TWE;
            case Full_LB_TWE:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                minDist = distComputer.distance(query.data[0], reference.data[0], nu, lambda);
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_TWE;
            case Full_TWE:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        // TWE
        switch (status) {
            case Full_TWE:
            case Full_LB_TWE:
                return thisD / query.length();
            case Partial_LB_TWE:
                return thisD / indexStoppedLB;
            case Previous_TWE:
                return 0.8 * thisD / query.length();
            case Previous_LB_TWE:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }

    public double getDistance() {
        return minDist;
    }
}
