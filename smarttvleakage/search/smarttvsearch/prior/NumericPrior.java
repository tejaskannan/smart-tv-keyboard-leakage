package smarttvsearch.prior;


public class NumericPrior extends LanguagePrior {

    public NumericPrior(String path) {
        super(path);
        this.totalCount = 10;
    }

    @Override
    public String[] processWord(String word) {
        return new String[] { word };
    }

    @Override
    public void build(boolean hasCounts) {
        return;
    }

    @Override
    public int find(String word) {
        return this.isValid(word) ? 1 : 0;
    }

    @Override
    public boolean isValidKey(String key) {
        if (key.length() != 1) {
            return false;
        }

        char c = key.charAt(0);
        return (c >= '0') && (c <= '9');
    }

    @Override
    public boolean isValid(String word) {
        for (int idx = 0; idx < word.length(); idx++) {
            char c = word.charAt(idx);
            if ((c < '0') || (c > '9')) {
                return false;
            }
        }

        return true;
    }
}
