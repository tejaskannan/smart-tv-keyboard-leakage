package smarttvsearch.prior;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.ResultSet;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.util.Map;
import java.util.HashMap;
import org.json.JSONObject;
import org.json.JSONArray;

import smarttvsearch.utils.SpecialKeys;
import smarttvsearch.utils.FileUtils;


public class EnglishPrior extends LanguagePrior {

    private String dbUrl;
    private Connection dbConn;
    private Map<String, Map<String, Integer>> cache;

    public static final String START_CHAR = "<S>";
    public static final String END_CHAR = "<E>";
    private static final String PREFIX_SELECT_QUERY = "SELECT prefix, next, count FROM prefixes WHERE prefix = ?;";
    private static final String WORD_SELECT_QUERY = "SELECT COUNT(*) AS count FROM words WHERE word = ?;";

    public EnglishPrior(String path) {
        super(path);

        this.dbUrl = String.format("jdbc:sqlite:%s", path);
        this.dbConn = null;
        this.cache = new HashMap<String, Map<String, Integer>>();

        // Read the metadata
        String metadataPath = String.format("%s_metadata.json", path.substring(0, path.length() - 3));
        JSONObject metadata = FileUtils.readJsonObject(metadataPath);
        this.totalCount = metadata.getInt("total_count");
    }

    @Override
    public void build(boolean hasCounts) {
        try {
            this.dbConn = DriverManager.getConnection(this.dbUrl);
        } catch (SQLException ex) {
            System.out.println(ex.getMessage());
        }
    }

    @Override
    public boolean isValidKey(String key) {
        return !key.equals(SpecialKeys.COM) && !key.equals(SpecialKeys.WWW);
    }

    @Override
    public String[] processWord(String word) {
        return new String[] { word };
    }

    @Override
    public boolean isValid(String word) {
        if ((word == null) || (word.length() == 0)) {
            return false;
        }

        // Get the results by looking in the SQL database
        try (PreparedStatement pstmt = this.dbConn.prepareStatement(WORD_SELECT_QUERY)) {
            // Issue the query
            pstmt.setString(1, word);
            ResultSet queryResults = pstmt.executeQuery();

            queryResults.next();
            int count = queryResults.getInt("count");
            queryResults.close();

            return (count > 0);
        } catch (SQLException ex) {
             System.out.println(ex.getMessage());
        }

        return false;
    }

    @Override
    public int find(String word) {
        if ((word == null) || (word.length() == 0)) {
            return this.getTotalCount();
        }

        // Construct the prefix for this word
        int wordCount = 0;
        String prefix;
        String lastChar;

        if (word.endsWith(END_CHAR)) {
            prefix = word.substring(0, word.length() - END_CHAR.length());
            lastChar = END_CHAR;
        } else {
            if (word.length() == 1) {
                prefix = START_CHAR;
            } else {
                prefix = word.substring(0, word.length() - 1);
            }

            lastChar = String.valueOf(word.charAt(word.length() - 1));
        }

        Map<String, Integer> prefixMap;
        
        if (this.cache.containsKey(prefix)) {  // Used the cached copy to avoid going to the database
            prefixMap = this.cache.get(prefix);
        } else {  // Look up the word in the database
            prefixMap = new HashMap<String, Integer>();

            // Get the results by looking in the SQL database
            try (PreparedStatement pstmt = this.dbConn.prepareStatement(PREFIX_SELECT_QUERY)) {
                // Issue the query
                pstmt.setString(1, prefix);
                ResultSet queryResults = pstmt.executeQuery();

                while (queryResults.next()) {
                    String next = queryResults.getString("next");
                    int count = queryResults.getInt("count");
                    prefixMap.put(next, count);
                }

                this.cache.put(prefix, prefixMap);
            } catch (SQLException ex) {
                 System.out.println(ex.getMessage());
            }
        }

        return prefixMap.getOrDefault(lastChar, 0);
    }

    public String[] nextMostCommon(String prefix, int topk) {
        // Use the start character instead of an empty string
        if ((prefix == null) || (prefix.length() == 0)) {
            prefix = START_CHAR;
        }

        Map<String, Integer> characterCounts;

        if (this.cache.containsKey(prefix)) {
            characterCounts = this.cache.get(prefix);
        } else {
            characterCounts = new HashMap<String, Integer>();

            // Get the results by looking in the SQL database
            try (PreparedStatement pstmt = this.dbConn.prepareStatement(PREFIX_SELECT_QUERY)) {
                // Issue the query
                pstmt.setString(1, prefix);
                ResultSet queryResults = pstmt.executeQuery();

                while (queryResults.next()) {
                    String next = queryResults.getString("next");
                    int count = queryResults.getInt("count");
                    characterCounts.put(next, count);
                }

                this.cache.put(prefix, characterCounts);
            } catch (SQLException ex) {
                 System.out.println(ex.getMessage());
            }
        }

        // Get the topk most common next characters
        int resultLength = Math.min(characterCounts.size(), topk);
        int[] topCounts = new int[resultLength];
        String[] topResults = new String[resultLength];
        int resultIdx;

        for (String nextChar : characterCounts.keySet()) {
            int count = characterCounts.get(nextChar);

            if (count <= topCounts[resultLength - 1]) {
                continue;  // Skip entries which are already below the current threshold
            }

            topCounts[resultLength - 1] = count;
            topResults[resultLength - 1] = nextChar;
            resultIdx = resultLength - 1;

            while ((resultIdx > 0) && ((topCounts[resultIdx - 1] < topCounts[resultIdx]))) {
                int tempCount = topCounts[resultIdx - 1];
                topCounts[resultIdx - 1] = topCounts[resultIdx];
                topCounts[resultIdx] = tempCount;

                String tempChar = topResults[resultIdx - 1];
                topResults[resultIdx - 1] = topResults[resultIdx];
                topResults[resultIdx] = tempChar;

                resultIdx -= 1;
            }
        }

        return topResults;
    }
}
