package smarttvsearch.prior;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;


public abstract class LanguagePrior {

    private String path;
    private int totalCount;
    protected HashMap<String, Integer> dictionary;

    public LanguagePrior(String path) {
        this.path = path;
        this.dictionary = new HashMap<String, Integer>();
        this.totalCount = 0;
    }

    public String getPath() {
        return this.path;
    }

    public int getTotalCount() {
        return this.totalCount;
    }

    public abstract int find(String word);
    public abstract String[] processWord(String word);

    public int[] findForPrefix(String prefix, String[] nextKeys) {
        int[] result = new int[nextKeys.length];

        for (int idx = 0; idx < nextKeys.length; idx++) {
            result[idx] = find(prefix + nextKeys[idx]);
        }

        return result;
    }

    public double normalizeCount(int count) {
        return ((double) count) / ((double) this.getTotalCount());
    }

    public boolean isValid(String word) {
        return true;
    }

    public boolean isValidKey(String key) {
        return true;
    }

    public void build(boolean hasCounts) {
        BufferedReader reader;
        String word;
        Integer count, existingCount;
        int idx;

        try {
            reader = new BufferedReader(new FileReader(this.getPath()));
            String line = reader.readLine();

            while (line != null) {
                if (hasCounts) {
                    String[] tokens = line.split(" ", 0);

                    if (tokens.length == 2) {
                        word = tokens[0];
                    } else {
                        StringBuilder builder = new StringBuilder();
                        for (idx = 0; idx < tokens.length - 1; idx++) {
                            builder.append(tokens[idx]);
                        }

                        word = builder.toString();
                    }

                    count = Integer.parseInt(tokens[tokens.length - 1]);
                } else {
                    word = line;
                    count = 1;
                }

                String[] processed = this.processWord(word);

                for (idx = 0; idx < processed.length; idx++) {
                    existingCount = this.dictionary.getOrDefault(processed[idx], 0);
                    this.dictionary.put(processed[idx], existingCount + count);
                    this.totalCount += count;
                }

                line = reader.readLine(); // Read the next line
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
