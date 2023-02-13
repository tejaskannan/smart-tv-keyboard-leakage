package smarttvsearch.tests.utils;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import java.util.List;
import java.util.Arrays;
import smarttvsearch.utils.VectorUtils;


public class VectorUtilsTests {

    @Test
    public void testDiff() {
        int[] vector = new int[] { 1, 6, 16, 24, 28, 30, 34, 41, 44 };

        List<Integer> diffs = VectorUtils.getDiffs(vector);
        int[] expected = new int[] { 5, 10, 8, 4, 2, 4, 7, 3 };

        assertEquals(expected.length, diffs.size());

        for (int idx = 0; idx < diffs.size(); idx++) {
            assertEquals((int) expected[idx], (int) diffs.get(idx));
        }
    }

    @Test
    public void testAverage() {
        List<Integer> vector = Arrays.asList(5, 10, 8, 4, 2, 4, 7, 3);
        assertEquals(5.375, VectorUtils.average(vector), 1e-4);
    }

    @Test
    public void testStdDev() {
        List<Integer> vector = Arrays.asList(5, 10, 8, 4, 2, 4, 7, 3);
        assertEquals(2.5464435, VectorUtils.stdDev(vector), 1e-4);
    }

    @Test
    public void testArgSort0() {
        double[] values = new double[] { 0.75, -1.0, 1.25, 1.6, -1.7 };
        int[] expected = new int[] { 3, 2, 0, 1, 4 };

        int[] result = VectorUtils.argsortDescending(values);

        assertEquals(expected.length, result.length);
        for (int idx = 0; idx < expected.length; idx++) {
            assertEquals(expected[idx], result[idx]);
        }
    }

    @Test
    public void testArgSort1() {
        double[] values = new double[] { 10.0, 2.0, -100.0, -6.0 };
        int[] expected = new int[] { 0, 1, 3, 2 };

        int[] result = VectorUtils.argsortDescending(values);

        assertEquals(expected.length, result.length);
        for (int idx = 0; idx < expected.length; idx++) {
            assertEquals(expected[idx], result[idx]);
        }
    }

    @Test
    public void testArgSort2() {
        double[] values = new double[0];
        int[] result = VectorUtils.argsortDescending(values);

        assertEquals(0, result.length);
    }
}
