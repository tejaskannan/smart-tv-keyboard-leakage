package smarttvsearch.search;

import java.util.ArrayList;
import java.util.List;


public class SearchResult {

    private String guess;
    private int rank;
    private int numCandidates;
    private int numPossibleGuesses;
    private double score;

    public SearchResult(String guess, int rank, int numCandidates, int numPossibleGuesses, double score) {
        this.guess = guess;
        this.rank = rank;
        this.numCandidates = numCandidates;
        this.numPossibleGuesses = numPossibleGuesses;
        this.score = score;
    }

    public String getGuess() {
        return this.guess;
    }

    public int getRank() {
        return this.rank;
    }

    public int getNumCandidates() {
        return this.numCandidates;
    }

    public int getNumPossibleGuesses() {
        return this.numPossibleGuesses;
    }

    public double getScore() {
        return this.score;
    }

    public void addNumCandidates(int toAdd) {
        this.numCandidates += toAdd;
    }

    public static List<String> toGuessList(List<SearchResult> results) {
        List<String> guesses = new ArrayList<String>();

        for (SearchResult result : results) {
            guesses.add(result.getGuess());
        }

        return guesses;
    }

    public static List<Double> toScoreList(List<SearchResult> results) {
        List<Double> scores = new ArrayList<Double>();

        for (SearchResult result : results) {
            scores.add(result.getScore());
        }

        return scores;
    }
}
