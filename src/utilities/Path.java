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

import java.io.File;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Path files
 */
public class Path {
    final static String osName = System.getProperty("os.name");
    final static String userName = System.getProperty("user.name");

    public static String outputPath = setOutputPath();
    public static String datasetPath = setDatasetPath();

    private static String setOutputPath() {
        outputPath = System.getProperty("user.dir") + "/output/";

        File dir = new File(outputPath);
        if (!dir.exists()) dir.mkdirs();
        return outputPath;
    }

    private static String setDatasetPath() {
        if (osName.contains("Window")) {
            datasetPath = "C:/Users/" + userName + "/workspace/Dataset/TSC_Problems/";
        } else {
            datasetPath = "/home/" + userName + "/workspace/Dataset/TSC_Problems/";
        }

        return datasetPath;
    }

    private static String setDatasetPath(String dpath) {
        datasetPath = dpath;
        return datasetPath;
    }
}
