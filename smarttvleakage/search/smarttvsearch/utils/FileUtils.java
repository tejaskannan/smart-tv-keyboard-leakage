package smarttvsearch.utils;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.File;
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

    public static void writeJsonObject(JSONObject jsonObject, String path) {
        try {
            FileWriter writer = new FileWriter(path);
            writer.write(jsonObject.toString());
            writer.close();
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
    }

    public static void writeJsonArray(JSONArray jsonArray, String path) {
        try {
            FileWriter writer = new FileWriter(path);
            writer.write(jsonArray.toString());
            writer.close();
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
    }

    public static String joinPath(String part0, String part1) {
        return String.format("%s%s%s", part0, File.separator, part1);
    }
}
