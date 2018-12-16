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
package development;

/**
 * Original code from http://www.timeseriesclassification.com/code.php
 * https://bitbucket.org/TonyBagnall/time-series-classification/src/f47c300b937af1d06aac86bdbe228cb9a9d5e00d?at=default
 */
public class DataSets {
    private final static String osName = System.getProperty("os.name");
    private final static String userName = System.getProperty("user.name");

    public static String problemPath = setProblemPath();

    private static String setProblemPath() {
        if (osName.contains("Window")) {
            problemPath = "C:/Users/" + userName + "/workspace/Dataset/TSC_Problems/";
        } else {
            problemPath = "/home/" + userName + "/workspace/Dataset/TSC_Problems/";
        }

        return problemPath;
    }

    //<editor-fold defaultstate="collapsed" desc="fileNames: All datasets">
    public static String[] allFileNames = {
            "AALTDChallenge",
            "Acsf1",
            "Adiac",        // 390,391,176,37
            "ArrowHead",    // 36,175,251,3
            "Beef",         // 30,30,470,5
            "BeetleFly",    // 20,20,512,2
            "BirdChicken",  // 20,20,512,2
            "Car",          // 60,60,577,4
            "CBF",                      // 30,900,128,3
            "ChlorineConcentration",    // 467,3840,166,3
            "CinCECGtorso", // 40,1380,1639,4
            "Coffee", // 28,28,286,2
            "Computers", // 250,250,720,2
            "CricketX", // 390,390,300,12
            "CricketY", // 390,390,300,12
            "CricketZ", // 390,390,300,12
            "DiatomSizeReduction", // 16,306,345,4
            "DistalPhalanxOutlineCorrect", // 600,276,80,2
            "DistalPhalanxOutlineAgeGroup", // 400,139,80,3
            "DistalPhalanxTW", // 400,139,80,6
            "Earthquakes", // 322,139,512,2
            "ECG200",   //100, 100, 96
            "ECG5000",  //4500, 500,140
            "ECGFiveDays", // 23,861,136,2
            "ElectricDevices", // 8926,7711,96,7
            "FaceAll", // 560,1690,131,14
            "FaceFour", // 24,88,350,4
            "FacesUCR", // 200,2050,131,14
            "FiftyWords", // 450,455,270,50
            "Fish", // 175,175,463,7
            "FordA", // 3601,1320,500,2
            "FordB", // 3636,810,500,2
            "GunPoint", // 50,150,150,2
            "Ham",      //105,109,431
            "HandOutlines", // 1000,370,2709,2
            "Haptics", // 155,308,1092,5
            "Herring", // 64,64,512,2
            "InlineSkate", // 100,550,1882,7
            "InsectWingbeatSound",//1980,220,256
            "ItalyPowerDemand", // 67,1029,24,2
            "LargeKitchenAppliances", // 375,375,720,3
            "Lightning2", // 60,61,637,2
            "Lightning7", // 70,73,319,7
            "Mallat", // 55,2345,1024,8
            "Meat",//60,60,448
            "MedicalImages", // 381,760,99,10
            "MiddlePhalanxOutlineCorrect", // 600,291,80,2
            "MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
            "MiddlePhalanxTW", // 399,154,80,6
            "MNIST",
            "MoteStrain", // 20,1252,84,2
            "NonInvasiveFetalECGThorax1", // 1800,1965,750,42
            "NonInvasiveFetalECGThorax2", // 1800,1965,750,42
            "OliveOil", // 30,30,570,4
            "OSULeaf", // 200,242,427,6
            "PhalangesOutlinesCorrect", // 1800,858,80,2
            "Phoneme",//1896,214, 1024
            "Plane", // 105,105,144,7
            "Plaid",
            "ProximalPhalanxOutlineCorrect", // 600,291,80,2
            "ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
            "ProximalPhalanxTW", // 400,205,80,6
            "RefrigerationDevices", // 375,375,720,3
            "ScreenType", // 375,375,720,3
            "ShapeletSim", // 20,180,500,2
            "ShapesAll", // 600,600,512,60
            "SmallKitchenAppliances", // 375,375,720,3
            "SonyAIBORobotSurface1", // 20,601,70,2
            "SonyAIBORobotSurface2", // 27,953,65,2
            "StarlightCurves", // 1000,8236,1024,3
            "Strawberry",//370,613,235
            "SwedishLeaf", // 500,625,128,15
            "Symbols", // 25,995,398,6
            "SyntheticControl", // 300,300,60,6
            "ToeSegmentation1", // 40,228,277,2
            "ToeSegmentation2", // 36,130,343,2
            "Trace", // 100,100,275,4
            "TwoLeadECG", // 23,1139,82,2
            "TwoPatterns", // 1000,4000,128,4
            "UWaveGestureLibraryX", // 896,3582,315,8
            "UWaveGestureLibraryY", // 896,3582,315,8
            "UWaveGestureLibraryZ", // 896,3582,315,8
            "UWaveGestureLibraryAll", // 896,3582,945,8
            "Wafer", // 1000,6164,152,2
            "Wine",//54	57	234
            "WordSynonyms", // 267,638,270,25
            "Worms", //77, 181,900,5
            "WormsTwoClass",//77, 181,900,5
            "Yoga" // 300,3000,426,2
    };
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="fileNames: The new 85 UCR datasets">    
    public static String[] fileNames = {
            "Adiac",        // 390,391,176,37
            "ArrowHead",    // 36,175,251,3
            "Beef",         // 30,30,470,5
            "BeetleFly",    // 20,20,512,2
            "BirdChicken",  // 20,20,512,2
            "Car",          // 60,60,577,4
            "CBF",                      // 30,900,128,3
            "ChlorineConcentration",    // 467,3840,166,3
            "CinCECGtorso", // 40,1380,1639,4
            "Coffee", // 28,28,286,2
            "Computers", // 250,250,720,2
            "CricketX", // 390,390,300,12
            "CricketY", // 390,390,300,12
            "CricketZ", // 390,390,300,12
            "DiatomSizeReduction", // 16,306,345,4
            "DistalPhalanxOutlineCorrect", // 600,276,80,2
            "DistalPhalanxOutlineAgeGroup", // 400,139,80,3
            "DistalPhalanxTW", // 400,139,80,6
            "Earthquakes", // 322,139,512,2
            "ECG200",   //100, 100, 96
            "ECG5000",  //4500, 500,140
            "ECGFiveDays", // 23,861,136,2
            "ElectricDevices", // 8926,7711,96,7
            "FaceAll", // 560,1690,131,14
            "FaceFour", // 24,88,350,4
            "FacesUCR", // 200,2050,131,14
            "FiftyWords", // 450,455,270,50
            "Fish", // 175,175,463,7
            "FordA", // 3601,1320,500,2
            "FordB", // 3636,810,500,2
            "GunPoint", // 50,150,150,2
            "Ham",      //105,109,431
            "HandOutlines", // 1000,370,2709,2
            "Haptics", // 155,308,1092,5
            "Herring", // 64,64,512,2
            "InlineSkate", // 100,550,1882,7
            "InsectWingbeatSound",//1980,220,256
            "ItalyPowerDemand", // 67,1029,24,2
            "LargeKitchenAppliances", // 375,375,720,3
            "Lightning2", // 60,61,637,2
            "Lightning7", // 70,73,319,7
            "Mallat", // 55,2345,1024,8
            "Meat",//60,60,448
            "MedicalImages", // 381,760,99,10
            "MiddlePhalanxOutlineCorrect", // 600,291,80,2
            "MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
            "MiddlePhalanxTW", // 399,154,80,6
            "MoteStrain", // 20,1252,84,2
            "NonInvasiveFetalECGThorax1", // 1800,1965,750,42
            "NonInvasiveFetalECGThorax2", // 1800,1965,750,42
            "OliveOil", // 30,30,570,4
            "OSULeaf", // 200,242,427,6
            "PhalangesOutlinesCorrect", // 1800,858,80,2
            "Phoneme",//1896,214, 1024
            "Plane", // 105,105,144,7
            "ProximalPhalanxOutlineCorrect", // 600,291,80,2
            "ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
            "ProximalPhalanxTW", // 400,205,80,6
            "RefrigerationDevices", // 375,375,720,3
            "ScreenType", // 375,375,720,3
            "ShapeletSim", // 20,180,500,2
            "ShapesAll", // 600,600,512,60
            "SmallKitchenAppliances", // 375,375,720,3
            "SonyAIBORobotSurface1", // 20,601,70,2
            "SonyAIBORobotSurface2", // 27,953,65,2
            "StarlightCurves", // 1000,8236,1024,3
            "Strawberry",//370,613,235
            "SwedishLeaf", // 500,625,128,15
            "Symbols", // 25,995,398,6
            "SyntheticControl", // 300,300,60,6
            "ToeSegmentation1", // 40,228,277,2
            "ToeSegmentation2", // 36,130,343,2
            "Trace", // 100,100,275,4
            "TwoLeadECG", // 23,1139,82,2
            "TwoPatterns", // 1000,4000,128,4
            "UWaveGestureLibraryX", // 896,3582,315,8
            "UWaveGestureLibraryY", // 896,3582,315,8
            "UWaveGestureLibraryZ", // 896,3582,315,8
            "UWaveGestureLibraryAll", // 896,3582,945,8
            "Wafer", // 1000,6164,152,2
            "Wine",//54	57	234
            "WordSynonyms", // 267,638,270,25
            "Worms", //77, 181,900,5
            "WormsTwoClass",//77, 181,900,5
            "Yoga" // 300,3000,426,2
    };
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Five splits of the new 85 UCR datasets">
    public static String[][] fiveSplits = {
            {"Adiac",        // 390,391,176,37
                    "ArrowHead",    // 36,175,251,3
                    "Beef",         // 30,30,470,5
                    "BeetleFly",    // 20,20,512,2
                    "BirdChicken",  // 20,20,512,2
                    "Car",          // 60,60,577,4
                    "CBF",                      // 30,900,128,3
                    "ChlorineConcentration",    // 467,3840,166,3
                    "CinCECGtorso", // 40,1380,1639,4
                    "Coffee", // 28,28,286,2
                    "Computers", // 250,250,720,2
                    "CricketX", // 390,390,300,12
                    "CricketY", // 390,390,300,12
                    "CricketZ", // 390,390,300,12
                    "DiatomSizeReduction", // 16,306,345,4
                    "DistalPhalanxOutlineCorrect", // 600,276,80,2
                    "DistalPhalanxOutlineAgeGroup", // 400,139,80,3
                    "DistalPhalanxTW", // 400,139,80,6
                    "Earthquakes" // 322,139,512,2
            },
            {
                    "ECG200",   //100, 100, 96
                    "ECG5000",  //4500, 500,140
                    "ECGFiveDays", // 23,861,136,2
                    "FaceFour", // 24,88,350,4
                    "FacesUCR", // 200,2050,131,14
                    "FiftyWords", // 450,455,270,50
                    "Fish", // 175,175,463,7
                    "GunPoint", // 50,150,150,2
                    "Ham",      //105,109,431
                    "Haptics", // 155,308,1092,5
                    "Herring", // 64,64,512,2
                    "ItalyPowerDemand", // 67,1029,24,2
                    "LargeKitchenAppliances", // 375,375,720,3
                    "Lightning2", // 60,61,637,2
                    "Lightning7", // 70,73,319,7
                    "Mallat", // 55,2345,1024,8
                    "Meat",//60,60,448
                    "MedicalImages", // 381,760,99,10
            },
            {
                    "MiddlePhalanxOutlineCorrect", // 600,291,80,2
                    "MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
                    "MiddlePhalanxTW", // 399,154,80,6
                    "MoteStrain", // 20,1252,84,2
                    "OliveOil", // 30,30,570,4
                    "OSULeaf", // 200,242,427,6
                    "Plane", // 105,105,144,7
                    "ProximalPhalanxOutlineCorrect", // 600,291,80,2
                    "ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
                    "ProximalPhalanxTW", // 400,205,80,6
                    "RefrigerationDevices", // 375,375,720,3
                    "ScreenType", // 375,375,720,3
                    "ShapeletSim", // 20,180,500,2
                    "SmallKitchenAppliances", // 375,375,720,3
                    "SonyAIBORobotSurface1", // 20,601,70,2
                    "SonyAIBORobotSurface2", // 27,953,65,2
                    "Strawberry",//370,613,235
                    "SwedishLeaf", // 500,625,128,15
                    "Symbols", // 25,995,398,6
                    "SyntheticControl" // 300,300,60,6
            },
            {
                    "ToeSegmentation1", // 40,228,277,2
                    "ToeSegmentation2", // 36,130,343,2
                    "Trace", // 100,100,275,4
                    "TwoLeadECG", // 23,1139,82,2
                    "TwoPatterns", // 1000,4000,128,4
                    "UWaveGestureLibraryX", // 896,3582,315,8
                    "UWaveGestureLibraryY", // 896,3582,315,8
                    "UWaveGestureLibraryZ", // 896,3582,315,8
                    "Wafer", // 1000,6164,152,2
                    "Wine",//54	57	234
                    "WordSynonyms", // 267,638,270,25
                    "Worms", //77, 181,900,5
                    "WormsTwoClass",//77, 181,900,5
                    "Yoga", // 300,3000,426,2
                    "InlineSkate", // 100,550,1882,7
                    "InsectWingbeatSound",//1980,220,256
                    "FaceAll", // 560,1690,131,14
                    "PhalangesOutlinesCorrect", // 1800,858,80,2
                    "Phoneme", //1896,214, 1024
                    "ShapesAll", // 600,600,512,60
            },
            {
                    "ElectricDevices", // 8926,7711,96,7
                    "FordA", // 3601,1320,500,2
                    "FordB", // 3636,810,500,2
                    "HandOutlines", // 1000,370,2709,2
                    "NonInvasiveFetalECGThorax1", // 1800,1965,750,42
                    "NonInvasiveFetalECGThorax2", // 1800,1965,750,42
                    "StarlightCurves", // 1000,8236,1024,3
                    "UWaveGestureLibraryAll", // 896,3582,945,8
            }
    };
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="ucrNames: 46 UCR Data sets">
    public static String[] ucrNames = {
            "Adiac", // 390,391,176,37
            "Beef", // 30,30,470,5
            "Car", // 60,60,577,4
            "CBF", // 30,900,128,3
            "ChlorineConcentration", // 467,3840,166,3
            "CinCECGtorso", // 40,1380,1639,4
            "Coffee", // 28,28,286,2
            "CricketX", // 390,390,300,12
            "CricketY", // 390,390,300,12
            "CricketZ", // 390,390,300,12
            "DiatomSizeReduction", // 16,306,345,4
            "ECGFiveDays", // 23,861,136,2
            "FaceAll", // 560,1690,131,14
            "FaceFour", // 24,88,350,4
            "FacesUCR", // 200,2050,131,14
            "FiftyWords", // 450,455,270,50
            "Fish", // 175,175,463,7
            "GunPoint", // 50,150,150,2
            "Haptics", // 155,308,1092,5
            "InlineSkate", // 100,550,1882,7
            "ItalyPowerDemand", // 67,1029,24,2
            "Lightning2", // 60,61,637,2
            "Lightning7", // 70,73,319,7
            "Mallat", // 55,2345,1024,8
            "MedicalImages", // 381,760,99,10
            "MoteStrain", // 20,1252,84,2
            "NonInvasiveFetalECGThorax1", // 1800,1965,750,42
            "NonInvasiveFetalECGThorax2", // 1800,1965,750,42
            "OliveOil", // 30,30,570,4
            "OSULeaf", // 200,242,427,6
            "Plane", // 105,105,144,7
            "SonyAIBORobotSurface1", // 20,601,70,2
            "SonyAIBORobotSurface2", // 27,953,65,2
            "StarLightCurves", // 1000,8236,1024,3
            "SwedishLeaf", // 500,625,128,15
            "Symbols", // 25,995,398,6
            "SyntheticControl", // 300,300,60,6
            "Trace", // 100,100,275,4
            "TwoLeadECG", // 23,1139,82,2
            "TwoPatterns", // 1000,4000,128,4
            "UWaveGestureLibraryX", // 896,3582,315,8
            "UWaveGestureLibraryY", // 896,3582,315,8
            "UWaveGestureLibraryZ", // 896,3582,315,8
            "Wafer", // 1000,6164,152,2
            "WordSynonyms", // 267,638,270,25
            "Yoga" // 300,3000,426,2
    };
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="ucrSmall: Small UCR Data sets">
    public static String[] ucrSmall = {
            "Beef", // 30,30,470,5
            "Car", // 60,60,577,4
            "Coffee", // 28,28,286,2
            "Cricket_X", // 390,390,300,12
            "Cricket_Y", // 390,390,300,12
            "Cricket_Z", // 390,390,300,12
            "DiatomSizeReduction", // 16,306,345,4
            "fish", // 175,175,463,7
            "GunPoint", // 50,150,150,2
            "ItalyPowerDemand", // 67,1029,24,2
            "MoteStrain", // 20,1252,84,2
            "OliveOil", // 30,30,570,4
            "Plane", // 105,105,144,7
            "SonyAIBORobotSurface", // 20,601,70,2
            "SonyAIBORobotSurfaceII", // 27,953,65,2
            "SyntheticControl", // 300,300,60,6
            "Trace", // 100,100,275,4
            "TwoLeadECG", // 23,1139,82,2
    };
    //</editor-fold>

    static int[] ucrTestSizes = {391, 175, 30, 20, 20, 60, 900, 3840, 1380, 28, 250, 390, 390, 390, 306, 276, 139, 139,
            139, 100, 4500, 861, 7711, 1690, 88, 2050, 455, 175, 1320, 810, 150, 105, 370, 308, 64, 550, 1980, 1029,
            375, 61, 73, 2345, 60, 760, 291, 154, 154, 1252, 1965, 1965, 30, 242, 858, 1896, 105, 291, 205, 205, 375,
            375, 180, 600, 375, 601, 953, 8236, 370, 625, 995, 300, 228, 130, 100, 1139, 4000, 3582, 3582, 3582, 3582,
            6164, 54, 638, 77, 77, 3000};
}

