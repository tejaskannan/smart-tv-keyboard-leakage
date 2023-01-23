package smarttvsearch.prior;


public class PrefixPrior extends LanguagePrior {

    public PrefixPrior(String path) {
        super(path);
    }

    @Override
    public int find(String word) {
        return this.dictionary.getOrDefault(word, 0);
    }

    @Override
    public String[] processWord(String word) {
        String[] prefixes = new String[word.length()];

        for (int endIdx = 1; endIdx <= word.length(); endIdx++) {
            prefixes[endIdx - 1] = word.substring(0, endIdx);
        }

        return prefixes;
    }

    @Override
    public boolean isValid(String word) {
        return this.dictionary.containsKey(word);
    }

    @Override
    public boolean isValidKey(String key) {
        // Validate that the key is numeric
        if (key.length() != 1) {
            return false;
        }

        char first = key.charAt(0);
        return (first >= '0') && (first <= '9');
    }
}
