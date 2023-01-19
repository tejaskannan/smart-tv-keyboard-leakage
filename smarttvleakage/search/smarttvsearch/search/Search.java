package smarttvsearch.search;

import java.util.PriorityQueue;
import java.util.List;
import java.util.HashSet;
import java.util.Set;
import java.util.ArrayList;

import smarttvsearch.keyboard.MultiKeyboard;
import smarttvsearch.prior.LanguagePrior;
import smarttvsearch.utils.search.SearchState;
import smarttvsearch.utils.search.VisitedState;
import smarttvsearch.utils.Move;


public class Search {

    private Move[] moveSeq;
    private MultiKeyboard keyboard;
    private LanguagePrior languagePrior;
    private String startKey;

    private PriorityQueue<SearchState> frontier;
    private HashSet<VisitedState> visited;
    private HashSet<String> guessed;

    public Search(Move[] moveSeq, MultiKeyboard keyboard, LanguagePrior languagePrior, String startKey) {
        this.moveSeq = moveSeq;
        this.keyboard = keyboard;
        this.languagePrior = languagePrior;
        this.startKey = startKey;

        this.frontier = new PriorityQueue<SearchState>();
        this.visited = new HashSet<VisitedState>();
        this.guessed = new HashSet<String>();

        // Generate the initial state and place this on the queue
        SearchState initState = new SearchState(startKey, new ArrayList<String>(), 0.0, keyboard.getStartKeyboard(), 0);
        this.frontier.add(initState);
    }

    public String next() {
        while (!this.frontier.isEmpty()) {
            // Get the top-ranked search state
            SearchState currentState = this.frontier.poll();
            int moveIdx = currentState.getMoveIdx();

            // Check if we are out of moves. If so, return the string if not guessed already.
            if (moveIdx >= this.moveSeq.length) {
                String guess = currentState.toString();

                if ((guess.length() > 0) && (!this.guessed.contains(guess)) && (this.languagePrior.isValid(guess))) {
                    return guess;
                }
            } else {
                // Get the move at this step
                Move move = this.moveSeq[moveIdx];
                String currentKey = currentState.getCurrentKey();
        
                // Get all neighboring keys for this move. TODO: handle possible suboptimal moves
                Set<String> neighbors = this.keyboard.getKeysForDistanceCumulative(currentKey, move.getNumMoves(), true, true, move.getDirections(), currentState.getKeyboardName());

                for (String neighborKey : neighbors) {
                    if (!this.languagePrior.isValidKey(neighborKey)) {
                        continue;  // Do not add invalid keys to the queue
                    }

                    // Make the candidate search state
                    List<String> nextKeys = new ArrayList<String>(currentState.getKeys());
                    nextKeys.add(neighborKey);

                    String nextKeyboard = this.keyboard.getNextKeyboard(neighborKey, currentState.getKeyboardName());
                    VisitedState visitedState = new VisitedState(nextKeys, nextKeyboard);

                    if (!this.visited.contains(visitedState)) {
                        // Make the candidate search state
                        SearchState candidateState = new SearchState(neighborKey, nextKeys, 0.0, nextKeyboard, moveIdx + 1);

                        int incrementalCount = this.languagePrior.find(candidateState.toString());

                        if (incrementalCount > 0) {
                            double incrementalScore = this.languagePrior.normalizeCount(incrementalCount);
                            double score = currentState.getScore() - Math.log(incrementalScore);

                            candidateState.setScore(score);
                            this.frontier.add(candidateState);
                            this.visited.add(visitedState);
                        }
                    }
                }
            }
        }

        return null;
    }


}


