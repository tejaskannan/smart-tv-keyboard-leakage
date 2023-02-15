package smarttvsearch.tests.utils;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import java.util.List;
import java.util.Arrays;
import smarttvsearch.utils.SearchUtils;
import smarttvsearch.utils.sounds.SmartTVSound;
import smarttvsearch.utils.sounds.SamsungSound;


public class SearchUtilsTests {

    @Test
    public void testNoDeletes() {
        SmartTVSound[] moveSounds = new SmartTVSound[] { new SamsungSound("key_select"), new SamsungSound("key_select"), new SamsungSound("key_select") };

        boolean[] expected = new boolean[] { false, false, false };
        boolean[] observed = SearchUtils.markDeletedMovesBySound(moveSounds);

        arrayEquals(expected, observed);
    }

    @Test
    public void testSingleDelete() {
        SmartTVSound[] moveSounds = new SmartTVSound[] { new SamsungSound("key_select"), new SamsungSound("key_select"), new SamsungSound("delete"), new SamsungSound("key_select") };

        boolean[] expected = new boolean[] { false, true, false, false };
        boolean[] observed = SearchUtils.markDeletedMovesBySound(moveSounds);

        arrayEquals(expected, observed);
    }

    @Test
    public void testIndependentDeletes() {
        SmartTVSound[] moveSounds = new SmartTVSound[] { new SamsungSound("key_select"), new SamsungSound("key_select"), new SamsungSound("delete"), new SamsungSound("key_select"), new SamsungSound("key_select"), new SamsungSound("delete") };

        boolean[] expected = new boolean[] { false, true, false, false, true, false };
        boolean[] observed = SearchUtils.markDeletedMovesBySound(moveSounds);

        arrayEquals(expected, observed);
    }

    @Test
    public void testConsecutiveDeletes() {
        SmartTVSound[] moveSounds = new SmartTVSound[] { new SamsungSound("key_select"), new SamsungSound("key_select"), new SamsungSound("delete"), new SamsungSound("delete"), new SamsungSound("key_select"), new SamsungSound("delete") };

        boolean[] expected = new boolean[] { true, true, false, false, true, false };
        boolean[] observed = SearchUtils.markDeletedMovesBySound(moveSounds);

        arrayEquals(expected, observed);
    }

    @Test
    public void testStartDeletes() {
        SmartTVSound[] moveSounds = new SmartTVSound[] { new SamsungSound("delete"), new SamsungSound("delete"), new SamsungSound("key_select"), new SamsungSound("delete"), new SamsungSound("key_select") };

        boolean[] expected = new boolean[] { false, false, true, false, false };
        boolean[] observed = SearchUtils.markDeletedMovesBySound(moveSounds);

        arrayEquals(expected, observed);
    }

    public static void arrayEquals(boolean[] expected, boolean[] observed) {
        assertEquals(expected.length, observed.length);

        for (int idx = 0; idx < expected.length; idx++) {
            assertEquals(expected[idx], observed[idx]);
        }
    }
}
