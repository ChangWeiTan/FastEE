package datasets;

import application.Application;
import utils.GenericTools;

import java.io.*;

import static utils.GenericTools.*;

public class DatasetLoader {
    private String fileDelimiter;

    public DatasetLoader() {
        fileDelimiter = ",";
    }

    public DatasetLoader(String fileDelimiter) {
        this.fileDelimiter = fileDelimiter;
    }

    private int[] getFileInformation(final String fileName, boolean hasHeader) throws IOException {
        final FileReader input = new FileReader(fileName);
        final LineNumberReader lineNumberReader = new LineNumberReader(input);
        String line;
        String[] lineArray = null;
        final int[] fileInfo = new int[2];

        try {
            boolean lengthCheck = true;

            while ((line = lineNumberReader.readLine()) != null) {
                if (lengthCheck) {
                    lengthCheck = false;
                    lineArray = line.split(fileDelimiter);
                }
            }
        } finally {
            input.close();
        }

        //this output array contains file information
        if (hasHeader) {
            //number of rows;
            fileInfo[0] = lineNumberReader.getLineNumber() == 0 ? lineNumberReader.getLineNumber() : lineNumberReader.getLineNumber() - 1;
        } else {
            //number of rows;
            fileInfo[0] = lineNumberReader.getLineNumber();
        }

        assert lineArray != null;
        fileInfo[1] = lineArray.length;  //number of columns;

        return fileInfo;
    }

    public Sequences readUCRTrain(final String datasetName, final String datasetPath, final boolean norm) {
        final String path = datasetPath + datasetName + "/" + datasetName + "_TRAIN.tsv";
        return readTSVFileToSequences(path, true, norm);
    }

    public Sequences readUCRTest(final String datasetName, final String datasetPath, final boolean norm) {
        final String path = datasetPath + datasetName + "/" + datasetName + "_TEST.tsv";
        return readTSVFileToSequences(path, true, norm);
    }

    public Sequences readTSVFileToSequences(final String fileName, boolean targetColumnIsFirst, boolean norm) {
        this.fileDelimiter = "\t";
        return readCSVFileToSequences(fileName, targetColumnIsFirst, norm);
    }

    public Sequences readCSVFileToSequences(final String fileName, boolean targetColumnIsFirst, boolean norm) {
        BufferedReader br = null;
        Sequences dataset = null;
        String line;
        String[] lineArray;
        String[] arrStr = fileName.split("/");
        arrStr = arrStr[arrStr.length - 1].split("_");

        String problem = arrStr[0];
        Sequences.DatasetType datasetType;
        if (arrStr[arrStr.length - 1].toLowerCase().contains("train"))
            datasetType = Sequences.DatasetType.Train;
        else if (arrStr[arrStr.length - 1].toLowerCase().contains("test"))
            datasetType = Sequences.DatasetType.Test;
        else
            datasetType = Sequences.DatasetType.Val;

        long usedMem;
        int label;
        int[] fileInfo;
        final File f = new File(fileName);
        boolean hasMissing = false;

        try {
            if (Application.verbose > 1)
                System.out.print("[DATASET-LOADER] reading [" + f.getName() + "]: ");
            final long startTime = System.nanoTime();

            // useful for reading large files;
            fileInfo = getFileInformation(fileName, false); // 0=> no. of rows 1=> no. columns
            final int expectedSize = fileInfo[0];
            final int seriesLength = fileInfo[1] - 1;  //-1 to exclude target the column

            // initialise
            dataset = new Sequences(expectedSize);
            dataset.problem = problem;
            dataset.datasetType = datasetType;
            br = new BufferedReader(new FileReader(fileName));

            while ((line = br.readLine()) != null) {
                lineArray = line.split(fileDelimiter);

                double[] tmp = new double[seriesLength];

                // read the data
                if (targetColumnIsFirst) {
                    for (int j = 1; j <= seriesLength; j++) {
                        tmp[j - 1] = Double.parseDouble(lineArray[j]);
                        if (isMissing(tmp[j - 1]))
                            hasMissing = true;
                    }
                    label = Integer.parseInt(lineArray[0]);
                } else {
                    int j;
                    for (j = 0; j < seriesLength; j++) {
                        tmp[j] = Double.parseDouble(lineArray[j]);
                        if (isMissing(tmp[j]))
                            hasMissing = true;
                    }
                    label = Integer.parseInt(lineArray[j]);
                }

                if (hasMissing) {
                    tmp = fillWithNoise(tmp, seriesLength);
                }
                if (norm) {
                    tmp = znormalise(tmp);
                }

                dataset.add(new Sequence(tmp, label));
            }
            final long endTime = System.nanoTime();
            final long elapsed = endTime - startTime;
            final String timeDuration = GenericTools.doTime(1.0 * elapsed / 1e6);
            if (Application.verbose > 1)
                System.out.println(" finished in " + timeDuration);
        } catch (IOException e) {
            System.err.println(e.getMessage());
            e.printStackTrace();
            System.exit(-1);
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
//        dataset.reorderClass();
        if (Application.iteration > 0)
            dataset.shuffle(Application.iteration);
        return dataset;
    }

    public static void main(String[] args) {
        String datasetName = "BeetleFly";
        // Get project and dataset path
        String osName = System.getProperty("os.name");
        String username = System.getProperty("user.name");

        String datasetPath;
        if (osName.contains("Window")) {
            datasetPath = "C:/Users/" + username + "/workspace/Dataset/UCRArchive_2018/";
        } else {
            datasetPath = "/home/" + username + "/workspace/Dataset/UCRArchive_2018/";
        }

        DatasetLoader loader = new DatasetLoader();
        Sequences train = loader.readUCRTrain(datasetName, datasetPath, true);
        Sequences test = loader.readUCRTest(datasetName, datasetPath, true);
    }
}

