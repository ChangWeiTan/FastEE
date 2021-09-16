package utils;

/**
 * Stores a tuple
 */
public class StrLong implements Comparable<StrLong>{
    public long value;
    public String str;

    public StrLong(String str, long value) {
        this.value = value;
        this.str = str;
    }

    @Override
    public int compareTo(StrLong other) {
        return Long.compare(other.value, value);
    }
}
