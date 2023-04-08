package smarttvsearch.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.Arrays;


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

    public static int max(List<Integer> values) {
        int maxVal = Integer.MIN_VALUE;

        if (values == null) {
            return maxVal;
        }

        for (int value : values) {
            if (value > maxVal) {
                maxVal = value;
            }
        }

        return maxVal;
    }

    public static int[] argsortDescending(int[] values) {
        int[] result = new int[values.length];

        for (int idx0 = 0; idx0 < values.length; idx0++) {
            result[idx0] = idx0;

            for (int idx1 = idx0 - 1; idx1 >= 0; idx1--) {
                int val0 = values[result[idx1 + 1]];
                int val1 = values[result[idx1]];

                if (val0 > val1) {
                    int temp = result[idx1 + 1];
                    result[idx1 + 1] = result[idx1];
                    result[idx1] = temp;
                }
            }
        }

        return result;
    }

    public static String[] sortStringSet(Set<String> strings) {
        String[] result = new String[strings.size()];

        int idx = 0;
        for (String str : strings) {
            result[idx] = str;
            idx += 1;
        }

        Arrays.sort(result);
        return result;
    }
}
