package smarttvsearch.utils;

import java.util.Stack;
import java.util.List;
import java.util.HashMap;

public class KeyboardUtils {

    private static HashMap<String, String> characterTranslation = createTranslationMap();

    private static HashMap<String, String> createTranslationMap() {
        HashMap<String, String> translation = new HashMap<String, String>();
        translation.put(SpecialKeys.WWW, "www");
        translation.put(SpecialKeys.COM, "com");
        translation.put(SpecialKeys.SPACE, " ");
        translation.put(SpecialKeys.LEFT, "");
        translation.put(SpecialKeys.RIGHT, "");
        translation.put(SpecialKeys.UP, "");
        translation.put(SpecialKeys.DOWN, "");
        translation.put(SpecialKeys.LANGUAGE, "");
        translation.put(SpecialKeys.DONE, "");
        translation.put(SpecialKeys.SETTINGS, "");
        translation.put(SpecialKeys.CHANGE, "");
        translation.put(SpecialKeys.NEXT, "");
        return translation;
    }

    public static boolean isSamsungSelectKey(String key) {
        return key.equals(SpecialKeys.LEFT) || key.equals(SpecialKeys.RIGHT) || key.equals(SpecialKeys.UP) || key.equals(SpecialKeys.DOWN) || key.equals(SpecialKeys.SPACE) || key.equals(SpecialKeys.DONE) || key.equals(SpecialKeys.CHANGE) || key.equals(SpecialKeys.NEXT) || key.equals(SpecialKeys.CAPS) || key.equals(SpecialKeys.LANGUAGE) || key.equals(SpecialKeys.SETTINGS) || key.equals(SpecialKeys.RETURN) || key.equals(SpecialKeys.CANCEL);
    }

    public static boolean isSamsungDeleteKey(String key) {
        return key.equals(SpecialKeys.DELETE) || key.equals(SpecialKeys.DELETE_ALL);
    }

    public static boolean isAppleTVDeleteKey(String key) {
        return key.equals(SpecialKeys.DELETE);
    }

    public static String keysToString(List<String> keys) {
        Stack<String> characters = new Stack<String>();

        boolean isCapsLock = false;
        boolean didPrevTurnOffCaps = false;
        String key;
        String character;

        for (int idx = 0; idx < keys.size(); idx++) {
            key = keys.get(idx);

            if (key.equals(SpecialKeys.CAPS)) {
                if (isCapsLock) {
                    isCapsLock = false;  // Turns off caps lock
                    didPrevTurnOffCaps = true;
                } else if ((idx > 0) && (keys.get(idx - 1).equals(SpecialKeys.CAPS))) {
                    isCapsLock = true;
                    didPrevTurnOffCaps = false;
                }
            } else if (key.equals(SpecialKeys.DELETE)) {
                if (!characters.empty()) {
                    characters.pop();  // Delete the last character
                }
            } else if (key.equals(SpecialKeys.DELETE_ALL)) {
                characters.clear();
            } else {
                character = KeyboardUtils.characterTranslation.getOrDefault(key, key);

                if (isCapsLock || ((idx > 0) && (keys.get(idx - 1).equals(SpecialKeys.CAPS)) && (!didPrevTurnOffCaps))) {
                    character = character.toUpperCase();
                }

                characters.push(character);
                didPrevTurnOffCaps = false;
            }
        }

        // Build the result string. The iteration over a stack occurs in reverse order by default.
        StringBuilder resultBuilder = new StringBuilder();
        for (String c : characters) {
            resultBuilder.append(c);
        }

        return resultBuilder.toString();
    }

}
