package classifiers.nearestNeighbour.elasticDistance;

import java.text.DecimalFormat;

/**
 * Super class for Elastic Distances
 */
public class ElasticDistances {
    final static int MAX_SEQ_LENGTH = 4000;         // maximum sequence length possible
    final static int DIAGONAL = 0;                  // value for diagonal
    final static int LEFT = 1;                      // value for left
    final static int UP = 2;                        // value for up

    static boolean paramsRefreshed = false;

    protected DecimalFormat df = new DecimalFormat("#0.####");

    static double dist(double a, double b) {
        double d = a - b;
        return d * d;
    }
}
