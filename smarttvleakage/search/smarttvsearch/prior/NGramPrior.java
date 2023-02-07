package smarttvsearch.prior;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.ResultSet;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.util.Map;
import java.util.HashMap;


public class NGramPrior extends LanguagePrior {

    private String dbUrl;
    private Connection dbConn;
    private Map<String, Map<Character, Integer>> cache;
    private static final String START_CHAR = "<S>";
    private static final String END_CHAR = "<E>";
    private static final String SELECT_QUERY = "SELECT prefix, next, count FROM ngrams WHERE prefix = ?;";
    private static final int NGRAM_SIZE = 5;

    public NGramPrior(String path) {
        super(path);

        this.dbUrl = String.format("jdbc:sqlite:%s", path);
        this.dbConn = null;
        this.cache = new HashMap<String, Map<Character, Integer>>();
    }

    @Override
    public void build(boolean hasCounts) {
        try {
            this.dbConn = DriverManager.getConnection(this.dbUrl);

            // Get the total count
            Statement stmt = this.dbConn.createStatement();
            ResultSet queryResults = stmt.executeQuery("SELECT SUM(count) FROM ngrams;");

            if (queryResults.next()) {
                this.totalCount = queryResults.getInt(1);
            }
        } catch (SQLException ex) {
            System.out.println(ex.getMessage());
        }
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
        String ngram = toNGram(word);
        char lastChar = word.charAt(word.length() - 1);

        if (this.cache.containsKey(ngram)) {
            Map<Character, Integer> ngramMap = this.cache.get(ngram);
            return ngramMap.getOrDefault(lastChar, 1);  // Apply Laplace Smoothing
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
                return ngramMap.getOrDefault(lastChar, 1);  // Apply Laplace Smoothing
            } catch (SQLException ex) {
                System.out.println(ex.getMessage());
            }
        }

        return 1;
    }

    public static String toNGram(String word) {
        int adjustedLength = Math.max(word.length() - 1, 0);

        if (word.length() > NGRAM_SIZE) {
            return word.substring(adjustedLength - NGRAM_SIZE, adjustedLength);
        } else {
            StringBuilder resultBuilder = new StringBuilder();

            // Prepend start characters. We logically skip the final character
            // as this we will fetch all suffixes in one query for efficiency.
            for (int idx = 0; idx < NGRAM_SIZE - adjustedLength; idx++) {
                resultBuilder.append(START_CHAR);
            }

            resultBuilder.append(word.substring(0, adjustedLength));
            return resultBuilder.toString();
        }
    }
}
