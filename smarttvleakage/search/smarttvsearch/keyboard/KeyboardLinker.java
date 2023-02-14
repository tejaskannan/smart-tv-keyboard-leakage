package smarttvsearch.keyboard;

import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;
import org.json.JSONObject;

import smarttvsearch.utils.FileUtils;


public class KeyboardLinker {

    private JSONObject linkerMap;  // Keyboard Name -> { key -> { linked_keyboard_name -> linked_key }}

    public KeyboardLinker(String path) {
        this.linkerMap = FileUtils.readJsonObject(path);
    }

    public List<KeyboardPosition> getLinkedKeys(String key, String currentKeyboard) {
        List<KeyboardPosition> result = new ArrayList<KeyboardPosition>();

        if (!this.linkerMap.has(currentKeyboard)) {
            return result;
        }

        JSONObject keyboardLinks = this.linkerMap.getJSONObject(currentKeyboard);

        if (!keyboardLinks.has(key)) {
            return result;
        }

        JSONObject keyLinks = keyboardLinks.getJSONObject(key);
        Iterator<String> keyboardNameIter = keyLinks.keys();

        while (keyboardNameIter.hasNext()) {
            String keyboardName = keyboardNameIter.next();
            KeyboardPosition position = new KeyboardPosition(keyLinks.getString(keyboardName), keyboardName);
            result.add(position);
        }

        return result;
    }
}
