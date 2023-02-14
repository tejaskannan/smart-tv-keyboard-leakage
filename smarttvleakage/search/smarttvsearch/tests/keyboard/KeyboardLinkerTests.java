package smarttvsearch.tests.keyboard;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.List;
import java.util.Arrays;
import smarttvsearch.keyboard.KeyboardLinker;
import smarttvsearch.keyboard.KeyboardPosition;


public class KeyboardLinkerTests {

    private KeyboardLinker samsungLinker;

    public KeyboardLinkerTests() {
        this.samsungLinker = new KeyboardLinker("../../keyboard/samsung/link.json");
    }

    @Test
    public void testSamsungLinkerLowercase() {
        List<KeyboardPosition> expected = Arrays.asList(new KeyboardPosition("S", "samsung_standard_caps"));
        validate(this.samsungLinker.getLinkedKeys("s", "samsung_standard"), expected);

        expected = Arrays.asList(new KeyboardPosition("1", "samsung_standard_caps"));
        validate(this.samsungLinker.getLinkedKeys("1", "samsung_standard"), expected);
    }

    private void validate(List<KeyboardPosition> observed, List<KeyboardPosition> expected) {
        assertEquals(expected.size(), observed.size());

        for (int idx = 0; idx < observed.size(); idx++) {
            assertEquals(expected.get(idx), observed.get(idx));
        }
    }
}

