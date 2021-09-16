package utils;

/**
 * Stores a tuple
 */
public class IndexedDouble implements Comparable<IndexedDouble>{
    public double value;
    public int index;

    public IndexedDouble(int index, double value) {
        this.value = value;
        this.index = index;
    }

    @Override
    public int compareTo(IndexedDouble other) {
        return Double.compare(value, other.value);
    }

    @Override
    public String toString() {
        return "" + this.index + "(" + this.value + ")";
    }

}
