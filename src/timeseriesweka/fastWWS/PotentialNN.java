/* Copyright (C) 2018 Chang Wei Tan, Francois Petitjean, Geoff Webb
 This file is part of FastEE.
 FastEE is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, version 3 of the License.
 LbEnhanced is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with LbEnhanced.  If not, see <http://www.gnu.org/licenses/>. */
package timeseriesweka.fastWWS;

/**
 * Code for the paper "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification"
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 *
 * Original code from https://github.com/ChangWeiTan/FastWWSearch
 *
 * Potential NN
 */
public class PotentialNN {
    public enum Status {
        NN,                         // This is the Nearest Neighbour
        BC,                         // Best Candidate so far
    }

    public int index;               // Index of the sequence in train[]
    public int r;                   // Window validity
    public double distance;         // Computed lower bound

    private Status status;

    public PotentialNN() {
        this.index = Integer.MIN_VALUE;                 // Will be an invalid, negative, index.
        this.r = Integer.MAX_VALUE;						// Max: stands for "haven't found yet"
        this.distance = Double.POSITIVE_INFINITY;       // Infinity: stands for "not computed yet".
        this.status = Status.BC;                        // By default, we don't have any found NN.
    }

    public void set(int index, int r, double distance, Status status) {
        this.index = index;
        this.r = r;
        this.distance = distance;
        this.status = status;
    }

    public void set(int index, double distance, Status status) {
        this.index = index;
        this.r = -1;
        this.distance = distance;
        this.status = status;
    }

    public boolean isNN() {
        return this.status == Status.NN;
    }

    @Override
    public String toString() {
        return "" + this.index;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        PotentialNN that = (PotentialNN) o;

        return index == that.index;
    }

    public int compareTo(PotentialNN potentialNN) {
        return Double.compare(this.distance, potentialNN.distance);
    }
}
