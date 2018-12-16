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

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * Original code from http://www.timeseriesclassification.com/code.php
 * https://bitbucket.org/TonyBagnall/time-series-classification/src/f47c300b937af1d06aac86bdbe228cb9a9d5e00d?at=default
 */
public class ClassifierTools {
    public static Instances loadTrain(String datapath, String problem) {
        return loadData(datapath + problem + "/" + problem + "_TRAIN");
    }

    public static Instances loadTest(String datapath, String problem) {
        return loadData(datapath + problem + "/" + problem + "_TEST");
    }

    public static Instances loadData(String fullPath) {
        if (fullPath.substring(fullPath.length() - 5, fullPath.length()).equalsIgnoreCase(".ARFF")) {
            fullPath = fullPath.substring(0, fullPath.length() - 5);
        }

        Instances d = null;
        FileReader r;
        try {
            r = new FileReader(fullPath + ".arff");
            d = new Instances(r);
            d.setClassIndex(d.numAttributes() - 1);
        } catch (IOException e) {
            System.out.println("Unable to load data on path " + fullPath + " Exception thrown =" + e);
            System.exit(0);
        }
        return d;
    }

    public static Instances loadData(File file) throws IOException {
        Instances inst = new Instances(new FileReader(file));
        inst.setClassIndex(inst.numAttributes() - 1);
        return inst;
    }

    public static double accuracy(Instances test, Classifier c) {
        double a = 0;
        int size = test.numInstances();
        Instance d;
        double predictedClass, trueClass;
        for (int i = 0; i < size; i++) {
            d = test.instance(i);
            try {
                predictedClass = c.classifyInstance(d);
                trueClass = d.classValue();
                if (trueClass == predictedClass) a++;
            } catch (Exception e) {
                System.out.println(" Error with instance " + i + " with Classifier " + c.getClass().getName() + " Exception =" + e);
                e.printStackTrace();
                System.exit(0);
            }
        }
        return a / size;
    }
}
