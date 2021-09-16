package utils;

/**
 * Stores a tuple
 */
public class StrInt implements Comparable<StrInt>{
    public int value;
    public String str;

    public StrInt(String str, int value) {
        this.value = value;
        this.str = str;
    }

    @Override
    public int compareTo(StrInt other) {
        return Integer.compare(other.value, value);
    }
}
