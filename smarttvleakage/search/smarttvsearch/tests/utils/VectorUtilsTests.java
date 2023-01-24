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
            assertEquals(expected[idx], diffs.get(idx));
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

}
