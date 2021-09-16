package datasets;

public class Sequence {
    // properties of the time series
    public double[][] data; // data in multivariate format, dim x length
    public int classificationLabel;

    public Sequence(final double[] series, int label) {
        this.data = new double[1][series.length];
        this.data[0] = series.clone();
        this.classificationLabel = label;
    }

    public Sequence(final double[][] series, int label) {
        this.data = series.clone();
        this.classificationLabel = label;
    }

    public Sequence(final double[][] series) {
        this.data = series.clone();
    }

    public void setValue(int index, double val) {
        setValue(0, index, val);
    }

    public void setValue(int dim, int index, double val) {
        this.data[dim][index] = val;
    }

    public double value(final int i) {
        return value(0, i);
    }

    public double value(final int d, final int i) {
        return this.data[d][i];
    }

    public void chopSeries(double ratio) {
        int subsetSize = (int) (ratio * length());
        double[][] choppedData = new double[data.length][subsetSize];
        for (int i = 0; i < data.length; i++){
            System.arraycopy(data[i], 0, choppedData[i], 0, subsetSize);
        }
        this.data = choppedData.clone();
    }

    public int dim() {
        if (this.data != null)
            return this.data.length;
        else
            return 0;
    }

    public int length() {
        if (this.data != null)
            return this.data[0].length;
        else
            return 0;
    }
}
