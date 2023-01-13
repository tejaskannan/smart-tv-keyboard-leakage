package smarttvsearch.utils;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import org.json.JSONObject;
import org.json.JSONArray;


public class FileUtils {

    public static String readTxtFile(String path) {
        BufferedReader reader;
        StringBuilder stringBuilder = new StringBuilder();

        try {
            reader = new BufferedReader(new FileReader(path));
            
            String line = reader.readLine();
            while (line != null) {
                stringBuilder.append(line);
                line = reader.readLine();
            }

            return stringBuilder.toString();
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        return null;
    }

    public static JSONObject readJsonObject(String path) {
        String jsonString = FileUtils.readTxtFile(path);

        if (jsonString == null) {
            return null;
        }

        return new JSONObject(jsonString);
    }

    public static JSONArray readJsonArray(String path) {
        String jsonString = FileUtils.readTxtFile(path);

        if (jsonString == null) {
            return null;
        }

        return new JSONArray(jsonString);
    }
}
