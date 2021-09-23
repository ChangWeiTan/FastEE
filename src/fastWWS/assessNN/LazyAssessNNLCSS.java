package fastWWS.assessNN;

import classifiers.nearestNeighbour.elasticDistance.LCSS;
import datasets.Sequence;
import fastWWS.SequenceStatsCache;
import results.WarpingPathResults;

public class LazyAssessNNLCSS extends LazyAssessNN {
    private final LCSS distComputer = new LCSS();

    public LazyAssessNNLCSS(final SequenceStatsCache cache) {
        super(cache);
    }

    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        minWindowValidity = 0;
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
                if (this.currentDelta < minWindowValidity) {
                    this.status = LBStatus.Previous_LCSS;
                }
            } else {
                this.status = LBStatus.Previous_LB_LCSS;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    private void tryFullLBLCSS() {
        final int length = query.length();
        final double[] LEQ = cache.getLE(indexQuery, currentDelta, currentEpsilon);
        final double[] UEQ = cache.getUE(indexQuery, currentDelta, currentEpsilon);
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

    public RefineReturnType tryToBeat(final double scoreToBeat, final int delta, final double epsilon) {
        setCurrentDeltaAndEpsilon(delta, epsilon);
        switch (status) {
            case None:
            case Previous_LB_LCSS:
            case Previous_LCSS:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_LCSS:
//                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryFullLBLCSS();
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist > scoreToBeat) {
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_LCSS;
                    else status = LBStatus.Full_LB_LCSS;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_LCSS;
            case Full_LB_LCSS:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = distComputer.distanceExt(query.data[0], reference.data[0], currentEpsilon, currentDelta);
                minDist = res.distance;
                minWindowValidity = res.distanceFromDiagonal;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_LCSS;
            case Full_LCSS:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_Dist;
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
            case Full_LCSS:
            case Full_LB_LCSS:
                return thisD / query.length();
            case Partial_LB_LCSS:
                return thisD / indexStoppedLB;
            case Previous_LCSS:
                return 0.8 * thisD / query.length();
            case Previous_LB_LCSS:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }

    public double getDistance(final int window) {
        if ((status == LBStatus.Full_LCSS) && minWindowValidity <= window) {
            return minDist;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is argMin3 valid already-computed Distance");
    }

    public int getMinWindowValidityForFullDistance() {
        if (status == LBStatus.Full_LCSS) {
            return minWindowValidity;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is argMin3 valid already-computed Distance");
    }
}
