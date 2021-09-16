package fastWWS;

public class CandidateNN {
    public enum Status {
        NN,                         // This is the Nearest Neighbour
        BC,                         // Best Candidate so far
    }

    public int nnIndex;               // Index of the sequence in train
    public int r;                   // Window validity
    public int paramValidity;
    public double distance;         // Computed lower bound
    public Status status;

    public CandidateNN() {
        this.nnIndex = Integer.MIN_VALUE;                 // Will be an invalid, negative, index.
        this.r = Integer.MAX_VALUE;                     // Max: stands for "haven't found yet"
        this.distance = Double.POSITIVE_INFINITY;       // Infinity: stands for "not computed yet".
        this.status = Status.BC;                        // By default, we don't have any found NN.
    }


    public void set(final int index, final int r, final int p, final double distance, final Status status) {
        this.nnIndex = index;
        this.r = r;
        this.paramValidity = p;
        this.distance = distance;
        this.status = status;
    }

    public void set(final int index, final int r, final double distance, final Status status) {
        this.nnIndex = index;
        this.r = r;
        this.distance = distance;
        this.status = status;
    }

    public void set(final int index, final double distance, final Status status) {
        this.nnIndex = index;
        this.r = -1;
        this.distance = distance;
        this.status = status;
    }

    public boolean isNN() {
        return this.status == Status.NN;
    }

    @Override
    public String toString() {
        return String.format("%d (%.4f)", this.nnIndex, this.distance);
    }

    @Override
    public boolean equals(final Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        CandidateNN that = (CandidateNN) o;

        return nnIndex == that.nnIndex;
    }

    public int compareTo(CandidateNN potentialNN) {
        return Double.compare(this.distance, potentialNN.distance);
    }

}
