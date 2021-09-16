package results;

public class WarpingPathResults {
    public double distance;
    public int distanceFromDiagonal; //The smallest window that would give the same distance
    public boolean earlyAbandon;

    public WarpingPathResults() {
        this.earlyAbandon = true;
    }

    public WarpingPathResults(double d, int distanceFromDiagonal) {
        this.distance = d;
        this.distanceFromDiagonal = distanceFromDiagonal;
        this.earlyAbandon = false;
    }

    public WarpingPathResults(double d, int distanceFromDiagonal, boolean earlyAbandon) {
        this.distance = d;
        this.distanceFromDiagonal = distanceFromDiagonal;
        this.earlyAbandon = earlyAbandon;
    }
}
