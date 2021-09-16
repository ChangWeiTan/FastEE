package utils;

/**
 * Stores a tuple
 */
public class StrDouble implements Comparable<StrDouble>{
    public double value;
    public String str;

    public StrDouble(String str, double value) {
        this.value = value;
        this.str = str;
    }

    @Override
    public int compareTo(StrDouble other) {
        return Double.compare(other.value, value);
    }
}
