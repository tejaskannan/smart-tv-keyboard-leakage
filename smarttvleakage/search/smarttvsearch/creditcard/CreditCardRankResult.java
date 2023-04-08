package smarttvsearch.creditcard;


public class CreditCardRankResult {

    private int rank;
    private boolean didFind;

    public CreditCardRankResult(int rank, boolean didFind) {
        this.rank = rank;
        this.didFind = didFind;
    }

    public int getRank() {
        return this.rank;
    }

    public boolean didFind() {
        return this.didFind;
    }
}
