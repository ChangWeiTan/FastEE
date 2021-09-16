package fastWWS.assessNN;

import classifiers.nearestNeighbour.elasticDistance.DTW;
import fastWWS.SequenceStatsCache;
import results.WarpingPathResults;

public class LazyAssessNNDTW extends LazyAssessNN {
    private final DTW distComputer = new DTW();

    public LazyAssessNNDTW(final SequenceStatsCache cache) {
        super(cache);
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final int w) {
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
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_KeoghQR;
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
                    if (indexStoppedLB < reference.length()) status = LBStatus.Partial_LB_KeoghRQ;
                    else status = LBStatus.Full_LB_KeoghRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_KeoghRQ;
            case Full_LB_KeoghRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = distComputer.distanceExt(query.data[0], reference.data[0], currentW);
                minDist = res.distance;
                minWindowValidity = res.distanceFromDiagonal;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_DTW;
            case Full_DTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }
}
