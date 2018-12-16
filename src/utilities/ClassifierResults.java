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
package utilities;

import java.util.ArrayList;

/**
 * Original code from http://www.timeseriesclassification.com/code.php
 * https://bitbucket.org/TonyBagnall/time-series-classification/src/f47c300b937af1d06aac86bdbe228cb9a9d5e00d?at=default
 */
public class ClassifierResults {
    public double buildTime;
    private int numClasses;
    private int numInstances;

    public double acc;
    public double f1;

    private ArrayList<Double> actualClassValues;
    private ArrayList<Double> predictedClassValues;

    public ClassifierResults() {
        actualClassValues = new ArrayList<>();
        predictedClassValues = new ArrayList<>();
    }

    public int getNumClasses() {
        return numClasses;
    }

    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    public int getNumInstances() {
        return numInstances;
    }

    public void setNumInstances(int numInstances) {
        this.numInstances = numInstances;
    }

    private double[][] buildConfusionMatrix() {
        double[][] confusionMatrix = new double[numClasses][numClasses];
        for (int i = 0; i < predictedClassValues.size(); ++i) {
            double actual = actualClassValues.get(i);
            double predicted = predictedClassValues.get(i);
            ++confusionMatrix[(int) actual][(int) predicted];
        }
        return confusionMatrix;
    }

    public int numInstances() {
        return predictedClassValues.size();
    }
}
