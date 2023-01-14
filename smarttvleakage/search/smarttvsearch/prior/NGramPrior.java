package smarttvsearch.prior;


public class NGramPrior extends LanguagePrior {

    public NGramPrior(String path) {
        super(path);
    }

    public String[] processWord(String word) {
        return new String[] { word };
    }

    public int find(String word) {
        return this.dictionary.getOrDefault(word, 0);
    }

}
