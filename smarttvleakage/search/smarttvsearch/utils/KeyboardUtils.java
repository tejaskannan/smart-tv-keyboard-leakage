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
        return translation;
    }

    public static String keysToString(List<String> keys) {
        Stack<String> characters = new Stack<String>();

        boolean isCapsLock = false;
        boolean didPrevTurnOffCaps = false;
        String key;
        String character;

        for (int idx = 0; idx < keys.size(); idx++) {
            key = keys.get(idx);

            if (key == SpecialKeys.CAPS) {
                if (isCapsLock) {
                    isCapsLock = false;  // Turns off caps lock
                    didPrevTurnOffCaps = true;
                } else if ((idx > 0) && (keys.get(idx - 1).equals(SpecialKeys.CAPS))) {
                    isCapsLock = true;
                    didPrevTurnOffCaps = false;
                }
            } else if (key == SpecialKeys.DELETE) {
                if (!characters.empty()) {
                    characters.pop();  // Delete the last character
                }
            } else if (key == SpecialKeys.DELETE_ALL) {
                characters.clear();
            } else {
                character = KeyboardUtils.characterTranslation.getOrDefault(key, key);

                if (isCapsLock || ((idx > 0) && (keys.get(idx - 1) == SpecialKeys.CAPS) && (!didPrevTurnOffCaps))) {
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
