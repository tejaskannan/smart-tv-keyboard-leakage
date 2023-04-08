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


public class NGramPrior extends LanguagePrior {

    private String dbUrl;
    private Connection dbConn;
    private Map<String, Map<Character, Integer>> cache;
    private int[] ngramSizes;

    private static final String START_CHAR = "<S>";
    private static final String END_CHAR = "<E>";
    private static final String SELECT_QUERY = "SELECT prefix, next, count FROM ngrams WHERE prefix = ?;";

    public NGramPrior(String path) {
        super(path);

        this.dbUrl = String.format("jdbc:sqlite:%s", path);
        this.dbConn = null;
        this.cache = new HashMap<String, Map<Character, Integer>>();

        // Read the metadata
        String metadataPath = String.format("%s_metadata.json", path.substring(0, path.length() - 3));
        JSONObject metadata = FileUtils.readJsonObject(metadataPath);

        JSONArray ngramSizesArray = metadata.getJSONArray("ngram_sizes");
        this.ngramSizes = new int[ngramSizesArray.length()];

        for (int idx = 0; idx < ngramSizesArray.length(); idx++) {
            this.ngramSizes[idx] = ngramSizesArray.getInt(idx);
        }

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
    public double normalizeCount(int count, String word) {
        String ngram = toNGram(word, this.ngramSizes[this.ngramSizes.length - 1] - 1);
        Map<Character, Integer> nextCounts = this.cache.get(ngram);

        if (nextCounts == null) {
            return ((double) count) / ((double) this.getTotalCount());
        } else {
            int totalCount = 0;
            for (Integer nextCount : nextCounts.values()) {
                totalCount += nextCount;
            }

            return ((double) count) / ((double) totalCount);
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
    public int find(String word) {
        if ((word == null) || (word.length() == 0)) {
            return this.getTotalCount();
        }

        // Construct the ngram to represent this word
        int wordCount = 0;

        for (int ngramIdx = this.ngramSizes.length - 1; ngramIdx >= 0; ngramIdx--) {
            String ngram = toNGram(word, this.ngramSizes[ngramIdx] - 1);
            char lastChar = word.charAt(word.length() - 1);

            if (this.cache.containsKey(ngram)) {
                Map<Character, Integer> ngramMap = this.cache.get(ngram);
                wordCount = ngramMap.getOrDefault(lastChar, 0);
            } else {
                Map<Character, Integer> ngramMap = new HashMap<Character, Integer>();

                // Get the results by looking in the SQL database
                try (PreparedStatement pstmt = this.dbConn.prepareStatement(SELECT_QUERY)) {
                    // Issue the query
                    pstmt.setString(1, ngram);
                    ResultSet queryResults = pstmt.executeQuery();

                    while (queryResults.next()) {
                        String next = queryResults.getString("next");

                        if (!next.equals(END_CHAR)) {
                            char nextChar = next.charAt(0);
                            int count = queryResults.getInt("count");
                            ngramMap.put(nextChar, count);
                        }
                    }

                    this.cache.put(ngram, ngramMap);

                    wordCount = ngramMap.getOrDefault(lastChar, 0);
                } catch (SQLException ex) {
                    System.out.println(ex.getMessage());
                }
            }

            if (wordCount > 0) {
                return wordCount;
            }
        }

        return 0;
    }

    public static String toNGram(String word, int ngramSize) {
        int adjustedLength = Math.max(word.length() - 1, 0);  // Clip off the last character, as we will search for the prefix

        if (word.length() > ngramSize) {
            return word.substring(adjustedLength - ngramSize, adjustedLength);
        } else {
            StringBuilder resultBuilder = new StringBuilder();

            // Prepend start characters. We logically skip the final character
            // as this we will fetch all suffixes in one query for efficiency.
            for (int idx = 0; idx < ngramSize - adjustedLength; idx++) {
                resultBuilder.append(START_CHAR);
            }

            resultBuilder.append(word.substring(0, adjustedLength));
            return resultBuilder.toString();
        }
    }
}
