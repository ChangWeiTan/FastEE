package classifiers;

import classifiers.nearestNeighbour.*;
import datasets.Sequences;

import static classifiers.TimeSeriesClassifier.TrainOpts.*;

public class FastElasticEnsemble extends ElasticEnsemble {
    public FastElasticEnsemble() {
        super();
        this.classifierIdentifier = "FastElasticEnsemble";
        this.shortName = "FastEE";
        this.trainingOptions = FastCV;
    }

    @Override
    OneNearestNeighbour getClassifier(ConstituentClassifiers classifier) throws Exception {
        OneNearestNeighbour baseClf;
        switch (classifier) {
            case Euclidean_1NN:
                baseClf = new ED1NN(LOOCV0LB);
                baseClf.useLBAtPrediction = false;
                break;
            case DTW_R1_1NN:
                baseClf = new DTW1NN(100, LOOCV0LB);
                baseClf.useLBAtPrediction = false;
                break;
            case DDTW_R1_1NN:
                baseClf = new DTW1NN(100, LOOCV0LB);
                baseClf.classifierIdentifier = baseClf.classifierIdentifier.replace("DTW", "DDTW");
                baseClf.useLBAtPrediction = false;
                break;
            case DTW_Rn_1NN:
                baseClf = new DTW1NN(-1, this.trainingOptions);
//                baseClf.useLBAtPrediction = false;
                break;
            case DDTW_Rn_1NN:
                baseClf = new DTW1NN(-1, this.trainingOptions);
                baseClf.classifierIdentifier = baseClf.classifierIdentifier.replace("DTW", "DDTW");
                baseClf.useLBAtPrediction = false;
                break;
            case WDTW_1NN:
                baseClf = new WDTW1NN(-1, this.trainingOptions);
                baseClf.useLBAtPrediction = false;
                break;
            case WDDTW_1NN:
                baseClf = new WDTW1NN(-1, this.trainingOptions);
                baseClf.classifierIdentifier = baseClf.classifierIdentifier.replace("DTW", "DDTW");
                baseClf.useLBAtPrediction = false;
                break;
            case LCSS_1NN:
                baseClf = new LCSS1NN(-1, this.trainingOptions);
                baseClf.useLBAtPrediction = false;
                break;
            case ERP_1NN:
                baseClf = new ERP1NN(-1, this.trainingOptions);
                baseClf.useLBAtPrediction = false;
                break;
            case MSM_1NN:
                baseClf = new MSM1NN(-1, this.trainingOptions);
                baseClf.useLBAtPrediction = false;
                break;
            case TWE_1NN:
                baseClf = new TWE1NN(-1, this.trainingOptions);
                baseClf.useLBAtPrediction = false;
                break;
            default:
                throw new Exception("Unsupported classifier type");
        }
        baseClf.useLBAtPrediction = true;
        return baseClf;
    }
}
