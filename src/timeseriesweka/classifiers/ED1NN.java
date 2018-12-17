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
package timeseriesweka.classifiers;

import timeseriesweka.fastWWS.SequenceStatsCache;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * NN-ED Classifier
 */
public class ED1NN extends OneNearestNeighbour {
    public ED1NN() {
        this.classifierIdentifier = "Euclidean_1NN";
        this.allowLoocv = false;
        this.singleParamCv = true;
    }

    @Override
    public void initFastWWS(Instances train, SequenceStatsCache cache) {

    }

    @Override
    public void initFastWWS(Instances train, SequenceStatsCache cache, int n) {

    }

    @Override
    public int initFastWWSEstimate(Instances train, SequenceStatsCache cache, long start, double timeLimit, int instanceLimit) {
        return 0;
    }

    @Override
    public void initFastWWSApproximate(Instances train, SequenceStatsCache cache, int nSamples) {

    }

    @Override
    public double[] fastWWSApproximate(Instances train, int nSamples) throws Exception {
        return super.loocv(train);
    }

    /*------------------------------------------------------------------------------------------------------------------
        Distances
     -----------------------------------------------------------------------------------------------------------------*/
    public final double distance(Instance first, Instance second) {
        double sum = 0;
        for (int a = 0; a < first.numAttributes() - 1; a++) {
            sum += (first.value(a) - second.value(a)) * (first.value(a) - second.value(a));
        }

        return sum;
    }

    @Override
    public void setParamsFromParamId(Instances train, int paramId) {
    }

    @Override
    public double classifyWithLowerBound(Instance instance) {
        return classifyInstance(instance);
    }

    @Override
    public String getParamInformationString() {
        return "NoParams";
    }
}
