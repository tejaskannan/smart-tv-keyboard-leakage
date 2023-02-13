package smarttvsearch.tests.keyboard;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.Set;
import smarttvsearch.keyboard.Keyboard;
import smarttvsearch.keyboard.MultiKeyboard;
import smarttvsearch.utils.Direction;


public class KeyboardTests {

    private Keyboard samsungKeyboardStd;

    public KeyboardTests() {
        this.samsungKeyboardStd = new Keyboard("../../keyboard/samsung/samsung_keyboard.json");
    }

    @Test
    public void testSamsungStdNeighborsBasic() {
        String[] expected = new String[] { "<CHANGE>", "w", "1", "a" };
        Set<String> observed = this.samsungKeyboardStd.getNeighbors("q", false, -1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "u", "k", "m", "h" };
        observed = this.samsungKeyboardStd.getNeighbors("j", false, -1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "!", "<UP>", "<RIGHT>" };
        observed = this.samsungKeyboardStd.getNeighbors("-", false, -1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "f", "c", "b" };
        observed = this.samsungKeyboardStd.getNeighbors("v", false, -1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "<RETURN>", "<CANCEL>", "!" };
        observed = this.samsungKeyboardStd.getNeighbors("<DONE>", false, -1, Direction.ANY);
        this.validateKeys(observed, expected);
    }

    @Test
    public void testSamsungStdNeighborsShortcuts() {
        String[] expected = new String[] { "<CHANGE>", "w", "1", "a" };
        Set<String> observed = this.samsungKeyboardStd.getNeighbors("q", false, 1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "u", "k", "m", "h" };
        observed = this.samsungKeyboardStd.getNeighbors("j", false, 1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "!", "<UP>", "<RIGHT>", "<DONE>" };
        observed = this.samsungKeyboardStd.getNeighbors("-", false, 1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "f", "c", "b", "<SPACE>" };
        observed = this.samsungKeyboardStd.getNeighbors("v", false, 1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "<RETURN>", "<CANCEL>", "!" };
        observed = this.samsungKeyboardStd.getNeighbors("<DONE>", false, 1, Direction.ANY);
        this.validateKeys(observed, expected);
    }

    @Test
    public void testSamsungStdNeighborsShortcutsAndWraparound() {
        String[] expected = new String[] { "<CHANGE>", "w", "1", "a" };
        Set<String> observed = this.samsungKeyboardStd.getNeighbors("q", true, 1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "u", "k", "m", "h" };
        observed = this.samsungKeyboardStd.getNeighbors("j", true, 1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "!", "<UP>", "<RIGHT>", "<DONE>" };
        observed = this.samsungKeyboardStd.getNeighbors("-", true, 1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "f", "c", "b", "<SPACE>" };
        observed = this.samsungKeyboardStd.getNeighbors("v", true, 1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "<RETURN>", "<CANCEL>", "!", "<LANGUAGE>" };
        observed = this.samsungKeyboardStd.getNeighbors("<DONE>", true, 1, Direction.ANY);
        this.validateKeys(observed, expected);

        expected = new String[] { "<CAPS>", "q", "<LANGUAGE>", "<RETURN>" };
        observed = this.samsungKeyboardStd.getNeighbors("<CHANGE>", true, 1, Direction.ANY);
        this.validateKeys(observed, expected);
    }

    @Test
    public void testSamsungStdDistBasic() {
        String[] expected = new String[] { "a" };
        Direction[] directions = null;
        Set<String> observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("a", 0, false, false, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "3", "g", "r", "v", "<SPACE>" };
        directions = new Direction[] { Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("a", 4, false, false, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "6", "8", "h", "k", "m", "o", "t" };
        directions = new Direction[] { Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("u", 2, false, false, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "z", "s", "e", "f", "b", "<SETTINGS>" };
        directions = new Direction[] { Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("c", 2, false, false, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "8", "o", "k", "m" };
        directions = new Direction[] { Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("q", 8, false, false, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "?", "@", "*", "<DOWN>", "<DONE>", "<CANCEL>" };
        directions = new Direction[] { Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("-", 2, false, false, directions);
        this.validateKeys(observed, expected);
    }

    @Test
    public void testSamsungStdDistWraparound() {
        String[] expected = new String[] { "3", "<DELETEALL>", "*", "-", "<RIGHT>", "@", "g", "r", "v", "<SPACE>" };
        Direction[] directions = new Direction[] { Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY };
        Set<String> observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("a", 4, true, false, directions);
        this.validateKeys(observed, expected);
    }

    @Test
    public void testSamsungStdDistShortcuts() {
        String[] expected = new String[] { "<SETTINGS>", "<WWW>", "c" };
        Direction[] directions = new Direction[] { Direction.ANY };
        Set<String> observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("<SPACE>", 1, false, true, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "z", "s", "e", "f", "b", "<SETTINGS>", "<WWW>" };
        directions = new Direction[] { Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("c", 2, false, true, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "8", "o", "k", "m", ".", "<LEFT>" };
        directions = new Direction[] { Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("q", 8, false, true, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "q", "s", "c", "<CHANGE>", "<SPACE>" };
        directions = new Direction[] { Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("z", 2, false, true, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "?", "@", "*", "<DOWN>", "<DONE>", "<CANCEL>", "<RETURN>" };
        directions = new Direction[] { Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("-", 2, false, true, directions);
        this.validateKeys(observed, expected);
    }

    @Test
    public void testSamsungStdDistShortcutsWraparound() {
        String[] expected = new String[] { "q", "s", "c", "<CHANGE>", "<SPACE>", "<DONE>" };
        Direction[] directions = new Direction[] { Direction.ANY, Direction.ANY };
        Set<String> observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("z", 2, true, true, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "9", "o", "k", ".", "/", "2", "e", "f", "c", "<SPACE>" };
        directions = new Direction[] { Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("<DONE>", 5, true, true, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "<RETURN>", "*", "@", "?", "/", "<CHANGE>", "a", "x", "<SPACE>", "<DONE>", "<BACK>", "<DELETEALL>" };
        directions = new Direction[] { Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("6", 7, true, true, directions);
        this.validateKeys(observed, expected);

        expected = new String[] { "<DONE>", "!", "<UP>", "<LEFT>", "<RIGHT>", "<DOWN>", "<DELETEALL>", "<RETURN>" };
        directions = new Direction[] { Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY };
        observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("4", 10, true, true, directions);
        this.validateKeys(observed, expected);
    }

    @Test
    public void testSamsungStdDistShortcutsWraparound4() {
        String[] expected = new String[] { "<DONE>", "!", "<UP>", "<LEFT>", "<COM>", "<DOWN>", "z", "h", "m", "s", "<DELETEALL>", "<RETURN>" };
        Direction[] directions = new Direction[] { Direction.ANY, Direction.HORIZONTAL, Direction.HORIZONTAL, Direction.HORIZONTAL, Direction.HORIZONTAL, Direction.HORIZONTAL, Direction.ANY, Direction.ANY, Direction.ANY, Direction.ANY };
        Set<String> observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("4", 10, true, true, directions);
        this.validateKeys(observed, expected);
    }

    @Test
    public void testSamsungStdDistShortcutsWraparound2() {
        String[] expected = new String[] { "<DONE>", "-", "<DOWN>", "<CANCEL>" };

        int numMoves = 13;
        Direction[] directions = new Direction[numMoves];
        for (int idx = 0; idx < numMoves; idx++) {
            directions[idx] = Direction.ANY;
        }

        Set<String> observed = this.samsungKeyboardStd.getKeysForDistanceCumulative("2", numMoves, true, true, directions);
        this.validateKeys(observed, expected);
    }

    private void validateKeys(Set<String> observed, String[] expected) {
        assertEquals(expected.length, observed.size());

        for (String elem : expected) {
            assertTrue(observed.contains(elem));
        }
    }
}

