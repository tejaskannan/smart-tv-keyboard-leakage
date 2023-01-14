package smarttvsearch.tests.utils;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import java.util.List;
import java.util.Arrays;
import smarttvsearch.utils.KeyboardUtils;


public class KeyboardUtilsTests {

    @Test
    public void test49ersSimple() {
        List<String> keys = Arrays.asList("4", "9", "e", "r", "s");
        String result = KeyboardUtils.keysToString(keys);
        assertEquals(result, "49ers");
    }

    @Test
    public void test49ersCapsSingle() {
        List<String> keys = Arrays.asList("4", "9", "e", "<CAPS>", "r", "s");
        String result = KeyboardUtils.keysToString(keys);
        assertEquals(result, "49eRs");
    }

    @Test
    public void test49ersCapsLock() {
        List<String> keys = Arrays.asList("4", "<CAPS>", "<CAPS>", "9", "e", "r", "<CAPS>", "s");
        String result = KeyboardUtils.keysToString(keys);
        assertEquals(result, "49ERs");
    }

    @Test
    public void test49ersDelete() {
        List<String> keys = Arrays.asList("4", "5", "<BACK>", "9", "e", "r", "s");
        String result = KeyboardUtils.keysToString(keys);
        assertEquals(result, "49ers");
    }

    @Test
    public void test49ersDeleteCaps() {
        List<String> keys = Arrays.asList("4", "5", "<BACK>", "<CAPS>", "<CAPS>", "9", "e", "r", "<CAPS>", "s");
        String result = KeyboardUtils.keysToString(keys);
        assertEquals(result, "49ERs");
    }

    @Test
    public void testDeleteAll() {
        List<String> keys = Arrays.asList("4", "9", "<BACK>", "9", "e", "r", "s", "<DELETEALL>", "b", "r", "o", "c", "k");
        String result = KeyboardUtils.keysToString(keys);
        assertEquals(result, "brock");
    }

    @Test
    public void testCharacterTranslationSimple() {
        List<String> keys = Arrays.asList("b", "r", "o", "c", "k", "<SPACE>", "p", "u", "r", "d", "y");
        String result = KeyboardUtils.keysToString(keys);
        assertEquals(result, "brock purdy");
    }

    @Test
    public void testCharacterTranslationCaps() {
        List<String> keys = Arrays.asList("<CAPS>", "b", "r", "o", "c", "<CAPS>", "<CAPS>", "k", "<SPACE>", "p", "u", "r", "<CAPS>", "d", "y");
        String result = KeyboardUtils.keysToString(keys);
        assertEquals(result, "BrocK PURdy");
    }
}

