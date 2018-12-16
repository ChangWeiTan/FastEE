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

import weka.core.EuclideanDistance;
import weka.core.Instances;

import java.text.DecimalFormat;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Superclass for all the elastic distances used
 */
public abstract class ElasticDistances extends EuclideanDistance {
    final static int MAX_SEQ_LENGTH = 4000;         // maximum sequence length possible
    final static int DIAGONAL = 0;                  // value for diagonal
    final static int LEFT = 1;                      // value for left
    final static int UP = 2;                        // value for up

    static boolean paramsRefreshed = false;

    protected DecimalFormat df = new DecimalFormat("#0.####");

    public abstract void setParamsFromParamID(Instances train, int paramId);
}
