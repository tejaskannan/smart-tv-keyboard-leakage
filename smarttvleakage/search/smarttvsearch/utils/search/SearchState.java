package smarttvsearch.utils.search;

import java.util.List;
import smarttvsearch.utils.KeyboardUtils;


public class SearchState implements Comparable<SearchState> {

    private String currentKey;
    private List<String> keys;
    private double score;
    private String keyboardName;
    private String keysAsString;
    private int moveIdx;

    public SearchState(String currentKey, List<String> keys, double score, String keyboardName, int moveIdx) {
        this.currentKey = currentKey;
        this.keys = keys;
        this.score = score;
        this.keyboardName = keyboardName;
        this.keysAsString = KeyboardUtils.keysToString(keys);
        this.moveIdx = moveIdx;
    }

    public String getCurrentKey() {
        return this.currentKey;
    }

    public double getScore() {
        return this.score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    public int getMoveIdx() {
        return this.moveIdx;
    }

    public String getKeyboardName() {
        return this.keyboardName;
    }

    public List<String> getKeys() {
        return this.keys;
    }

    @Override
    public String toString() {
        return this.keysAsString;
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
