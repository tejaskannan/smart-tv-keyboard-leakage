package smarttvsearch.utils;

import java.util.List;


public class SearchState implements Comparable<SearchState> {

    private List<String> keys;
    private double score;
    private String keyboardName;

    public SearchState(List<String> keys, double score, String keyboardName) {
        this.keys = keys;
        this.score = score;
        this.keyboardName = keyboardName;
    }

    public double getScore() {
        return this.score;
    }

    public String getKeyboardName() {
        return this.keyboardName;
    }

    public String getLastKey() {
        if (this.keys.isEmpty()) {
            return null;
        } else {
            int length = this.keys.size();
            return this.keys.get(length - 1);
        }
    }

    public String getString() {
        return KeyboardUtils.keysToString(this.keys);
    }

    public int compareTo(SearchState other) {
        if (this.getScore() > other.getScore()) {
            return 1;
        } else if (this.getScore() < other.getScore()) {
            return -1;
        } else {
            return 0;
        }
    }

}
