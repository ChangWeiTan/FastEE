package datasets;

import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.*;

public class Sequences {
    public enum DatasetType {
        Train,
        Test,
        Val,
    }

    public String problem;      // problem
    public DatasetType datasetType;
    private ArrayList<Sequence> timeseriesData; // arraylist of sequences (time series)
    private TIntIntMap classMap;                // key = class label, value = class size
    private Map<Integer, Integer> initialClassLabels;   // class labels
    private boolean isReordered;
    private int dimension;                      // dimension of the series
    private int largestClass;                   // largest class in the series
    private int length;                         // length of time series
    private int maxLength;
    private int minLength;

    public Sequences() {
        this.timeseriesData = new ArrayList<>();
        this.classMap = new TIntIntHashMap();
        this.dimension = 1; // at least 1 dimension
    }

    public Sequences(final int size) {
        this.timeseriesData = new ArrayList<>(size);
        this.classMap = new TIntIntHashMap();
        this.dimension = 1; // at least 1 dimension
    }

    public Sequences(final int size, final int length) {
        this.timeseriesData = new ArrayList<>(size);
        this.classMap = new TIntIntHashMap();
        this.length = length;
        this.dimension = 1; // at least 1 dimension
    }

    public Sequences(final int size, final int length, final int dim) {
        this.timeseriesData = new ArrayList<>(size);
        this.classMap = new TIntIntHashMap();
        this.length = length;
        this.dimension = dim;
    }


    /**
     * Initialise sequences from other sequences with fixed size
     *
     * @param sequences other sequences
     * @param size      fixed size
     */
    public Sequences(final Sequences sequences, final int size) {
        // initialise sequences from other sequences
        this.timeseriesData = new ArrayList<>(size);
        this.classMap = new TIntIntHashMap(sequences.classMap.size());
        this.problem = sequences.problem;
        this.datasetType = sequences.datasetType;
        this.dimension = sequences.dimension;
        this.largestClass = sequences.largestClass;
        this.isReordered = sequences.isReordered;
        this.initialClassLabels = sequences.initialClassLabels;
        this.length = sequences.length;
        this.maxLength = sequences.maxLength;
        this.minLength = sequences.minLength;
    }

    /**
     * Initialise sequences from other sequences with the same size
     *
     * @param sequences other sequences
     */
    public Sequences(final Sequences sequences) {
        // initialise sequences from other sequences
        this.timeseriesData = new ArrayList<>(sequences.timeseriesData);
        this.classMap = new TIntIntHashMap(sequences.classMap);
        this.problem = sequences.problem;
        this.datasetType = sequences.datasetType;
        this.dimension = sequences.dimension;
        this.largestClass = sequences.largestClass;
        this.isReordered = sequences.isReordered;
        this.initialClassLabels = sequences.initialClassLabels;
        this.length = sequences.length;
        this.maxLength = sequences.maxLength;
        this.minLength = sequences.minLength;
    }

    public final void summary() {
        System.out.println("[DATA SUMMARY] Problem: " + problem + " (" + datasetType + ")" +
                "\n[DATA SUMMARY] Size: " + size() +
                "\n[DATA SUMMARY] Dim: " + dim() +
                "\n[DATA SUMMARY] Length: " + length());
        if (this.maxLength > Integer.MIN_VALUE)
            System.out.println("[DATA SUMMARY] Max Length: " + this.maxLength);
        if (this.minLength < Integer.MAX_VALUE)
            System.out.println("[DATA SUMMARY] Min Length: " + this.minLength);
        System.out.println("[DATA SUMMARY] Num Classes: " + getNumClasses() +
                "\n[DATA SUMMARY] Largest class: " + this.largestClass +
                "\n[DATA SUMMARY] Class Distribution: " + getClassDistribution());
    }

    public void add(final Sequence instance) {
        this.timeseriesData.add(instance);
        this.dimension = instance.dim();
        this.minLength = Math.min(instance.length(), this.minLength);
        this.maxLength = Math.max(instance.length(), this.maxLength);
        if (this.maxLength == this.minLength)
            this.length = instance.length();

        int label = instance.classificationLabel;
        if (this.classMap.containsKey(label)) {
            this.classMap.put(label, this.classMap.get(label) + 1);
        } else {
            this.classMap.put(label, 1);
        }
    }

    public Sequence get(final int i) {
        return this.timeseriesData.get(i);
    }

    public void setLargestClass(int largestClass) {
        this.largestClass = largestClass;
    }

    public void setInitialClassOrder(final Map<Integer, Integer> initial_order) {
        this.initialClassLabels = initial_order;
    }

    public void setReordered(final boolean status) {
        this.isReordered = status;
    }

    public Map<Integer, Integer> getInitialClassLabels() {
        return this.initialClassLabels;
    }

    public Sequences reorderClassLabels(Map<Integer, Integer> newOrder) {
        final Sequences newDataset = new Sequences(this.size(), this.length(), this.dim());
        //key = old label, value = new label, easier to build this way, later we swap to new=>old
        if (newOrder == null) {
            newOrder = new HashMap<>();
        }

        final int size = this.size();

        int newLabel = 0;
        int tempLabel;

        for (int i = 0; i < size; i++) {
            final Integer oldLabel = this.timeseriesData.get(i).classificationLabel;

            if (newOrder.containsKey(oldLabel)) {
                tempLabel = newOrder.get(oldLabel);
            } else {
                newOrder.put(oldLabel, newLabel);
                tempLabel = newLabel;
                newLabel++;
            }
            this.timeseriesData.get(i).classificationLabel = tempLabel;

            newDataset.add(this.timeseriesData.get(i));
        }

        newDataset.setInitialClassOrder(newOrder);
        newDataset.setReordered(true);
        newDataset.problem = this.problem;
        newDataset.datasetType = this.datasetType;
        newDataset.setLargestClass(newLabel);
        return newDataset;
    }

    public void reorderClass() {
        int count = 0;
        boolean saveInitClassLabel = false;
        if (this.initialClassLabels == null) {
            this.initialClassLabels = new HashMap<>();
            saveInitClassLabel = true;
        }
        TIntIntMap newClassMap = new TIntIntHashMap(this.classMap.size());
        for (int label : this.classMap.keys()) {
            newClassMap.put(count, this.classMap.get(label));
            if (saveInitClassLabel) this.initialClassLabels.put(label, count);
            count++;
        }
        this.classMap = newClassMap;
        for (Sequence timeseriesDatum : timeseriesData) {
            timeseriesDatum.classificationLabel = this.initialClassLabels.get(timeseriesDatum.classificationLabel);
        }
    }

    public void shuffle(int seed) {
        Collections.shuffle(this.timeseriesData, new Random(seed));
    }

    public Sequences stratifySubset(double ratio) {
        int subsetSize = (int) (ratio * timeseriesData.size());
        Sequences subsetData = new Sequences(this);
        subsetData.timeseriesData = new ArrayList<>(subsetSize);
        subsetData.classMap = new TIntIntHashMap();
        for (Integer c : this.classMap.keys()) {
            subsetData.classMap.put(c, 0);
        }
        // reorder and group the ones with the same class
        for (int index = 1; index < this.timeseriesData.size(); ++index) {
            Sequence instance1 = this.timeseriesData.get(index - 1);
            for (int j = index; j < this.timeseriesData.size(); ++j) {
                Sequence instance2 = this.timeseriesData.get(j);
                if (instance1.classificationLabel == instance2.classificationLabel) {

                    final Sequence tmp_series = this.timeseriesData.get(j);
                    this.timeseriesData.set(j, this.timeseriesData.get(index));
                    this.timeseriesData.set(index, tmp_series);

                    ++index;
                }
            }
        }

        int step = timeseriesData.size() / subsetSize;
        for (int j = 0; j < timeseriesData.size(); j += step) {
            subsetData.add(timeseriesData.get(j));
        }
        return subsetData;
    }

    public void chopSeries(double ratio) {
        for (Sequence s : timeseriesData) {
            s.chopSeries(ratio);
        }
        this.maxLength = timeseriesData.get(0).length();
    }

    public int size() {
        return this.timeseriesData.size();
    }

    public int length() {
        return this.maxLength;
    }

    public int dim() {
        return this.dimension;
    }

    public int getNumClasses() {
        return this.classMap.size();
    }

    public String getClassDistribution() {
        StringBuilder str = new StringBuilder();
        int i = 0;
        for (int key : classMap.keys()) {
            str.append(key).append(": ").append(classMap.get(key));
            if (i < classMap.size() - 1)
                str.append(", ");
            i++;
        }

        return str.toString();
    }
}
