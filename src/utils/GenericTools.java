package utils;

import datasets.Sequences;
import weka.core.Instances;

import java.util.Random;

public class GenericTools {
    static Random random = new Random(100);

    public static String doTime(double elapsedTimeNanoSeconds) {
        final double duration = elapsedTimeNanoSeconds / 1e6;
        return String.format("%d s %.3f ms", (int) (duration / 1000), (duration % 1000));
    }

    public static String doTimeNs(double elapsedTimeNanoSeconds) {
        int hour = (int) (elapsedTimeNanoSeconds / 3.6e+12);
        int min = (int) (elapsedTimeNanoSeconds / 6e+10);
        int s = (int) (elapsedTimeNanoSeconds / 1e9);
        int ms = (int) (elapsedTimeNanoSeconds / 1e6);
        int us = (int) (elapsedTimeNanoSeconds / 1e3);
        StringBuilder str = new StringBuilder();
        if (hour > 0)
            str.append((hour % 60)).append(" H ");
        if (min > 0)
            str.append((min % 60)).append(" M ");
        if (s > 0)
            str.append((s % 60)).append(" s ");
        if (ms > 0)
            str.append((ms % 1000)).append(" ms ");
        if (us > 0)
            str.append((us % 1000)).append(" us ");

        str.append(((int) (elapsedTimeNanoSeconds % 1000))).append(" ns");

        return str.toString();
    }

    public static boolean isMissing(double a) {
        return Double.isNaN(a);
    }

    public static double[] fillWithNoise(final double[] data, final int maxLen) {
        final int seqLen = data.length;
        final double[] arr = new double[maxLen];

        System.arraycopy(data, 0, arr, 0, seqLen);

        for (int i = 0; i < maxLen; i++) {
            if (isMissing(arr[i]))
                arr[i] = random.nextDouble() / 1000;
        }
        return arr;
    }

    public static double[] znormalise(final double[] values) {
        double sum = 0.0;
        double standardDeviation = 0.0;
        int length = values.length;

        for (double num : values) {
            sum += num;
        }
        double m = sum / length;

        for (double num : values) {
            standardDeviation += Math.pow(num - m, 2);
        }
        double sd = Math.sqrt(standardDeviation / (length));

        final double[] normalizedValues = new double[length];
        if (sd <= 0)
            sd = 1;
        for (int i = 0; i < length; i++) {
            if (Double.isNaN(values[i])) {
                normalizedValues[i] = values[i];
            } else {
                normalizedValues[i] = (values[i] - m) / sd;
            }
        }
        return normalizedValues;
    }

    public static double min3(final double a, final double b, final double c) {
        return (a <= b) ? (Math.min(a, c)) : Math.min(b, c);
    }

    public static int argMin3(final double a, final double b, final double c) {
        return (a <= b) ? ((a <= c) ? 0 : 2) : (b <= c) ? 1 : 2;
    }

    public static void println(Object str) {
        System.out.println(str);
    }

    public static double stdv_p(Sequences input) {
        double sumx = 0;
        double sumx2 = 0;
        double[] ins2array;
        for (int i = 0; i < input.size(); i++) {
            ins2array = input.get(i).data[0];
            for (int j = 0; j < ins2array.length; j++) {//-1 to avoid classVal
                sumx += ins2array[j];
                sumx2 += ins2array[j] * ins2array[j];
            }
        }
        int n = input.size() * (input.length());
        double mean = sumx / n;
        return Math.sqrt(sumx2 / (n) - mean * mean);
    }

    public static int[] getInclusive10(final int min, final int max) {
        int[] output = new int[10];

        double diff = 1.0 * (max - min) / 9;
        double[] doubleOut = new double[10];
        doubleOut[0] = min;
        output[0] = min;
        for (int i = 1; i < 9; i++) {
            doubleOut[i] = doubleOut[i - 1] + diff;
            output[i] = (int) Math.round(doubleOut[i]);
        }
        output[9] = max; // to make sure max isn't omitted due to double imprecision
        return output;
    }

    public static double[] getInclusive10(final double min, final double max) {
        double[] output = new double[10];
        double diff = 1.0 * (max - min) / 9;
        output[0] = min;
        for (int i = 1; i < 9; i++) {
            output[i] = output[i - 1] + diff;
        }
        output[9] = max;

        return output;
    }
}
