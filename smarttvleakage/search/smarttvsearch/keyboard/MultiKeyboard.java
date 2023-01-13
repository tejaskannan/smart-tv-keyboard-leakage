package smarttvsearch.keyboard;

import java.io.File;
import java.util.HashMap;
import java.util.Set;
import smarttvsearch.utils.SmartTVType;
import smarttvsearch.utils.SpecialKeys;
import smarttvsearch.utils.Direction;


public class MultiKeyboard {

    public static final String SAMSUNG_STD = "samsung_keyboard";
    public static final String SAMSUNG_CAPS = "samsung_keyboard_caps";
    public static final String SAMSUNG_SPECIAL_0 = "samsung_keyboard_special_0";
    public static final String SAMSUNG_SPECIAL_1 = "samsung_keyboard_special_1";

    public static final String APPLETV_STD = "appletv_keyboard";
    public static final String APPLETV_CAPS = "appletv_keyboard_caps";
    public static final String APPLETV_SPECIAL = "appletv_keyboard_special";

    private HashMap<String, Keyboard> keyboards;
    private String startKeyboard;
    private String startKey;
    private SmartTVType tvType;

    public MultiKeyboard(SmartTVType tvType, String graphFolder) {
        keyboards = new HashMap<String, Keyboard>();
        this.tvType = tvType;

        String path;
        if (tvType == SmartTVType.SAMSUNG) {
            path = String.format("%s%s%s.json", graphFolder, File.separator, MultiKeyboard.SAMSUNG_STD);
            keyboards.put(MultiKeyboard.SAMSUNG_STD, new Keyboard(path));

            path = String.format("%s%s%s.json", graphFolder, File.separator, MultiKeyboard.SAMSUNG_CAPS);
            keyboards.put(MultiKeyboard.SAMSUNG_CAPS, new Keyboard(path));

            path = String.format("%s%s%s.json", graphFolder, File.separator, MultiKeyboard.SAMSUNG_SPECIAL_0);
            keyboards.put(MultiKeyboard.SAMSUNG_SPECIAL_0, new Keyboard(path));

            path = String.format("%s%s%s.json", graphFolder, File.separator, MultiKeyboard.SAMSUNG_SPECIAL_1);
            keyboards.put(MultiKeyboard.SAMSUNG_SPECIAL_1, new Keyboard(path));

            this.startKeyboard = MultiKeyboard.SAMSUNG_STD;
            this.startKey = "q";
        } else if (tvType == SmartTVType.APPLETV) {
            this.startKeyboard = MultiKeyboard.APPLETV_STD;
            this.startKey = "a";
        } else {
            throw new IllegalArgumentException("Unknown tv type: " + tvType.name());
        }
    }

    public SmartTVType getTVType() {
        return this.tvType;
    }

    public String getStartKeyboard() {
        return this.startKeyboard;
    }

    public String getStartKey() {
        return this.startKey;
    }

    public String getNextKeyboard(String pressedKey, String currentKeyboard) {
        if (this.getTVType() == SmartTVType.SAMSUNG) {
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
        }

        return currentKeyboard;
    }

    public Set<String> getNeighbors(String key, boolean useWraparound, boolean useShortcuts, Direction direction, String keyboardName) {
        Keyboard keyboard = this.keyboards.get(keyboardName);
        return keyboard.getNeighbors(key, useWraparound, useShortcuts, direction);
    }

    public Set<String> getKeysForDistance(String key, int distance, boolean useWraparound, boolean useShortcuts, Direction[] directions, String keyboardName) {
        Keyboard keyboard = this.keyboards.get(keyboardName);
        return keyboard.getKeysForDistance(key, distance, useWraparound, useShortcuts, directions);
    }

    public Set<String> getKeysForDistanceCumulative(String key, int distance, boolean useWraparound, boolean useShortcuts, Direction[] directions, String keyboardName) {
        Keyboard keyboard = this.keyboards.get(keyboardName);
        return keyboard.getKeysForDistanceCumulative(key, distance, useWraparound, useShortcuts, directions);
    }
}
