package smarttvsearch.utils.search;

public class BFSState {

    private String key;
    private int distance;

    public BFSState(String key, int distance) {
        this.key = key;
        this.distance = distance;
    }

    public String getKey() {
        return this.key;
    }

    public int getDistance() {
        return this.distance;
    }
}
