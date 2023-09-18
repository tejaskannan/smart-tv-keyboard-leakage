package smarttvsearch.keyboard;

import java.io.File;
import java.util.HashMap;
import java.util.Set;
import java.util.List;
import smarttvsearch.utils.KeyboardType;
import smarttvsearch.utils.SmartTVType;
import smarttvsearch.utils.SpecialKeys;
import smarttvsearch.utils.Direction;
import smarttvsearch.utils.FileUtils;


public class MultiKeyboard {

    public static final String SAMSUNG_STD = "samsung_keyboard";
    public static final String SAMSUNG_CAPS = "samsung_keyboard_caps";
    public static final String SAMSUNG_SPECIAL_0 = "samsung_keyboard_special_0";
    public static final String SAMSUNG_SPECIAL_1 = "samsung_keyboard_special_1";

    public static final String APPLETV_STD = "appletv_keyboard_standard";
    public static final String APPLETV_CAPS = "appletv_keyboard_caps";
    public static final String APPLETV_SPECIAL = "appletv_keyboard_special";

    public static final String ABC_STD = "abc_keyboard";
    public static final String ABC_CAPS = "abc_keyboard_caps";
    public static final String ABC_SPECIAL = "abc_keyboard_special";

    private HashMap<String, Keyboard> keyboards;
    private String startKeyboard;
    private String startKey;
    private KeyboardType keyboardType;
    private SmartTVType tvType;
    private KeyboardLinker linker;

    public MultiKeyboard(KeyboardType keyboardType, String graphFolder) {
        keyboards = new HashMap<String, Keyboard>();
        this.keyboardType = keyboardType;

        String path;
        if (keyboardType == KeyboardType.SAMSUNG) {
            path = FileUtils.joinPath(graphFolder, String.format("%s.json", MultiKeyboard.SAMSUNG_STD));
            keyboards.put(MultiKeyboard.SAMSUNG_STD, new Keyboard(path));

            path = FileUtils.joinPath(graphFolder, String.format("%s.json", MultiKeyboard.SAMSUNG_CAPS));
            keyboards.put(MultiKeyboard.SAMSUNG_CAPS, new Keyboard(path));

            path = FileUtils.joinPath(graphFolder, String.format("%s.json", MultiKeyboard.SAMSUNG_SPECIAL_0));
            keyboards.put(MultiKeyboard.SAMSUNG_SPECIAL_0, new Keyboard(path));

            path = FileUtils.joinPath(graphFolder, String.format("%s.json", MultiKeyboard.SAMSUNG_SPECIAL_1));
            keyboards.put(MultiKeyboard.SAMSUNG_SPECIAL_1, new Keyboard(path));

            this.startKeyboard = MultiKeyboard.SAMSUNG_STD;
            this.startKey = "q";
        } else if (keyboardType == KeyboardType.APPLE_TV) {
            path = FileUtils.joinPath(graphFolder, String.format("%s.json", MultiKeyboard.APPLETV_STD));
            keyboards.put(MultiKeyboard.APPLETV_STD, new Keyboard(path));

            path = FileUtils.joinPath(graphFolder, String.format("%s.json", MultiKeyboard.APPLETV_CAPS));
            keyboards.put(MultiKeyboard.APPLETV_CAPS, new Keyboard(path));

            path = FileUtils.joinPath(graphFolder, String.format("%s.json", MultiKeyboard.APPLETV_SPECIAL));
            keyboards.put(MultiKeyboard.APPLETV_SPECIAL, new Keyboard(path));

            this.startKeyboard = MultiKeyboard.APPLETV_STD;
            this.startKey = "a";
        } else if (keyboardType == KeyboardType.ABC) {
            path = FileUtils.joinPath(graphFolder, String.format("%s.json", MultiKeyboard.ABC_STD));
            keyboards.put(MultiKeyboard.ABC_STD, new Keyboard(path));

            path = FileUtils.joinPath(graphFolder, String.format("%s.json", MultiKeyboard.ABC_CAPS));
            keyboards.put(MultiKeyboard.ABC_CAPS, new Keyboard(path));

            path = FileUtils.joinPath(graphFolder, String.format("%s.json", MultiKeyboard.ABC_SPECIAL));
            keyboards.put(MultiKeyboard.ABC_SPECIAL, new Keyboard(path));

            this.startKeyboard = MultiKeyboard.ABC_STD;
            this.startKey = "a";
        } else {
            throw new IllegalArgumentException("Unknown tv type: " + tvType.name());
        }

        // Create the implicit keyboard linker
        path = FileUtils.joinPath(graphFolder, "link.json");
        linker = new KeyboardLinker(path);
    }

    public KeyboardType getKeyboardType() {
        return this.keyboardType;
    }

    public String getStartKeyboard() {
        return this.startKeyboard;
    }

    public String getStartKey() {
        return this.startKey;
    }

    public String getNextKeyboard(String pressedKey, String currentKeyboard) {
        if (this.getKeyboardType() == KeyboardType.SAMSUNG) {
            if (pressedKey.equals(SpecialKeys.CHANGE)) {
                if (currentKeyboard.equals(MultiKeyboard.SAMSUNG_STD) || currentKeyboard.equals(MultiKeyboard.SAMSUNG_CAPS)) {
                    return MultiKeyboard.SAMSUNG_SPECIAL_0;
                } else {
                    return MultiKeyboard.SAMSUNG_STD;
                }
            } else if (pressedKey.equals(SpecialKeys.NEXT)) {
                if (currentKeyboard.equals(MultiKeyboard.SAMSUNG_SPECIAL_0)) {
                    return MultiKeyboard.SAMSUNG_SPECIAL_1;
                } else if (currentKeyboard.equals(MultiKeyboard.SAMSUNG_SPECIAL_1)) {
                    return MultiKeyboard.SAMSUNG_SPECIAL_0;
                }
            }
        } else if (this.getKeyboardType() == KeyboardType.ABC) {
            if (pressedKey.equals(SpecialKeys.CHANGE)) {
                if (currentKeyboard.equals(MultiKeyboard.ABC_SPECIAL)) {
                    return MultiKeyboard.ABC_STD;
                } else {
                    return MultiKeyboard.ABC_SPECIAL;
                }
            }
        }

        return currentKeyboard;
    }

    public List<KeyboardPosition> getLinkedKeys(String key, String currentKeyboard) {
        return this.linker.getLinkedKeys(key, currentKeyboard);
    }

    public Set<String> getNeighbors(String key, boolean useWraparound, int shortcutIdx, Direction direction, String keyboardName) {
        Keyboard keyboard = this.keyboards.get(keyboardName);
        return keyboard.getNeighbors(key, useWraparound, shortcutIdx, direction);
    }

    public Set<String> getKeysForDistance(String key, int distance, boolean useWraparound, int shortcutIdx, Direction[] directions, String keyboardName) {
        Keyboard keyboard = this.keyboards.get(keyboardName);
        return keyboard.getKeysForDistance(key, distance, useWraparound, shortcutIdx, directions);
    }

    public Set<String> getKeysForDistanceCumulative(String key, int distance, boolean useWraparound, boolean useShortcuts, Direction[] directions, String keyboardName, boolean shouldEnforceMinDistance) {
        Keyboard keyboard = this.keyboards.get(keyboardName);

        if (keyboard == null) {
            System.out.println(keyboardName);
        }

        Set<String> withDirections = keyboard.getKeysForDistanceCumulative(key, distance, useWraparound, useShortcuts, directions);

        if (shouldEnforceMinDistance) {
            Direction[] anyDirections = new Direction[directions.length];
            for (int idx = 0; idx < directions.length; idx++) {
                anyDirections[idx] = Direction.ANY;
            }

            Set<String> withoutDirections = keyboard.getKeysForDistanceCumulative(key, distance, useWraparound, useShortcuts, anyDirections);
            withDirections.retainAll(withoutDirections);
        }

        return withDirections;
    }

    public boolean isClickable(String key, String keyboardName) {
        Keyboard keyboard = this.keyboards.get(keyboardName);
        return keyboard.isClickable(key);
    }
}
