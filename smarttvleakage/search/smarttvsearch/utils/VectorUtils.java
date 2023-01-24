package smarttvsearch.utils;

import java.util.ArrayList;
import java.util.List;


public class VectorUtils {

    public static List<Integer> getDiffs(int[] values) {
        if (values.length <= 1) {
            return null;
        }

        List<Integer> diffs = new ArrayList<Integer>();

        for (int idx = 1; idx < values.length; idx++) {
            diffs.add(Math.abs(values[idx] - values[idx - 1]));
        }

        return diffs;
    }


    public static double average(List<Integer> values) {
        if (values.isEmpty()) {
            return 0.0;
        }

        int sum = 0;
        for (int idx = 0; idx < values.size(); idx++) {
            sum += values.get(idx);
        }

        return ((double) sum) / ((double) values.size());
    }

    public static double stdDev(List<Integer> values) {
        if (values.isEmpty()) {
            return 0.0;
        }

        double avg = VectorUtils.average(values);
        double diffSum = 0.0;

        for (int idx = 0; idx < values.size(); idx++) {
            double diff = ((double) values.get(idx)) - avg;
            diffSum += (diff * diff);
        }

        return Math.sqrt(diffSum / ((double) values.size()));
    }
}
