package smarttvsearch.search;

import java.util.PriorityQueue;
import java.util.List;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.ArrayList;

import smarttvsearch.keyboard.MultiKeyboard;
import smarttvsearch.prior.LanguagePrior;
import smarttvsearch.suboptimal.SuboptimalMoveModel;
import smarttvsearch.utils.sounds.SmartTVSound;
import smarttvsearch.utils.sounds.SamsungSound;
import smarttvsearch.utils.search.SearchState;
import smarttvsearch.utils.search.VisitedState;
import smarttvsearch.utils.KeyboardUtils;
import smarttvsearch.utils.Move;
import smarttvsearch.utils.Direction;
import smarttvsearch.utils.SpecialKeys;
import smarttvsearch.utils.SmartTVType;
import smarttvsearch.utils.sounds.SamsungSound;


public class Search {

    private Move[] moveSeq;
    private MultiKeyboard keyboard;
    private LanguagePrior languagePrior;
    private String startKey;
    private SuboptimalMoveModel suboptimalModel;
    private boolean doesEndWithDone;
    private boolean useDirections;
    private boolean shouldAccumulateScore;
    private SmartTVType tvType;

    private PriorityQueue<SearchState> frontier;
    private HashSet<VisitedState> visited;
    private HashSet<String> guessed;

    public Search(Move[] moveSeq, MultiKeyboard keyboard, LanguagePrior languagePrior, String startKey, SuboptimalMoveModel suboptimalModel, SmartTVType tvType, boolean useDirections, boolean shouldAccumulateScore) {
        this.moveSeq = moveSeq;
        this.keyboard = keyboard;
        this.languagePrior = languagePrior;
        this.startKey = startKey;
        this.suboptimalModel = suboptimalModel;
        this.useDirections = useDirections;
        this.shouldAccumulateScore = shouldAccumulateScore;
        this.tvType = tvType;

        this.frontier = new PriorityQueue<SearchState>();
        this.visited = new HashSet<VisitedState>();
        this.guessed = new HashSet<String>();

        // Generate the initial state and place this on the queue
        SearchState initState = new SearchState(startKey, new ArrayList<String>(), 0.0, keyboard.getStartKeyboard(), 0);
        this.frontier.add(initState);

        // Check if the last move has a select sound. If so, then we end with a move to `done`
        this.doesEndWithDone = false;

        if (tvType == SmartTVType.SAMSUNG) {
            Move lastMove = moveSeq[moveSeq.length - 1];
            SamsungSound lastSound = (SamsungSound) lastMove.getEndSound();
            this.doesEndWithDone = lastSound.isSelect();
        }
    }

    public String next() {
        while (!this.frontier.isEmpty()) {
            // Get the top-ranked search state
            SearchState currentState = this.frontier.poll();
            int moveIdx = currentState.getMoveIdx();

            // Check if we are out of moves. If so, return the string if not guessed already.
            if (this.isFinished(moveIdx))  {
                String guess = currentState.toString();
                String lastKey = currentState.getKeys().get(moveIdx - 1);

                if ((guess.length() > 0) && (!this.guessed.contains(guess)) && (this.languagePrior.isValid(guess))) {
                    this.guessed.add(guess);
                    return guess;
                }
            } else {
                // Get the move at this step
                Move move = this.moveSeq[moveIdx];
                String currentKey = currentState.getCurrentKey();

                int numMoves = move.getNumMoves();
                int numSuboptimal = this.suboptimalModel.getLimit(moveIdx);
                HashMap<String, Double> neighborKeys = new HashMap<String, Double>();  // Neighbor -> Score factor

                for (int offset = -1 * numSuboptimal; offset <= numSuboptimal; offset++) {
                    int adjustedNumMoves = numMoves + offset;
                    if (adjustedNumMoves < 0) {
                        continue;
                    }

                    Direction[] directions;

                    if ((offset == 0) && this.useDirections) {
                        directions = move.getDirections();
                    } else {
                        directions = new Direction[adjustedNumMoves];
                        for (int dirIdx = 0; dirIdx < adjustedNumMoves; dirIdx++) {
                            directions[dirIdx] = Direction.ANY;
                        }
                    }

                    Set<String> neighbors = this.keyboard.getKeysForDistanceCumulative(currentKey, adjustedNumMoves, true, true, directions, currentState.getKeyboardName());
                    double scoreFactor = this.suboptimalModel.getScoreFactor(offset);
   
                    for (String neighbor : neighbors) {
                        Double prevFactor = neighborKeys.get(neighbor);

                        if ((prevFactor == null) || (scoreFactor > prevFactor)) {
                            neighborKeys.put(neighbor, scoreFactor);
                        }
                    }
                }

                int nextMoveIdx = moveIdx + 1;

                if (currentState.toString().equals("2599248044395")) {
                    System.out.println(move);
                    System.out.println(neighborKeys);
                }

                for (String neighborKey : neighborKeys.keySet()) {
                    if (!this.isValidKey(neighborKey, nextMoveIdx, move.getEndSound())) {
                        continue;  // Do not add invalid keys to the queue
                    }

                    // Make the candidate search state
                    List<String> nextKeys = new ArrayList<String>(currentState.getKeys());
                    nextKeys.add(neighborKey);

                    String nextKeyboard = this.keyboard.getNextKeyboard(neighborKey, currentState.getKeyboardName());
                    VisitedState visitedState = new VisitedState(nextKeys, nextKeyboard);

                    if (!this.visited.contains(visitedState)) {
                        // Make the candidate search state
                        SearchState candidateState = new SearchState(neighborKey, nextKeys, 0.0, nextKeyboard, nextMoveIdx);

                        int incrementalCount = this.languagePrior.find(candidateState.toString());

                        if (incrementalCount > 0) {
                            double score = this.languagePrior.normalizeCount(incrementalCount);
                            score *= neighborKeys.get(neighborKey);
                            score = -1 * Math.log(score);

                            if (this.shouldAccumulateScore) {
                                score = currentState.getScore() + score;
                            }

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

    private boolean isValidKey(String key, int nextMoveIdx, SmartTVSound endSound) {
        if (this.isFinished(nextMoveIdx) && this.doesEndWithDone) {
            return key.equals(SpecialKeys.DONE);
        } else if (this.tvType == SmartTVType.SAMSUNG) {
            SamsungSound endSamsungSound = (SamsungSound) endSound;

            if (endSamsungSound.isSelect()) {
                return KeyboardUtils.isSamsungSelectKey(key);
            } else if (endSamsungSound.isDelete()) {
                return KeyboardUtils.isSamsungDeleteKey(key);
            } else {
                return this.languagePrior.isValidKey(key) && !KeyboardUtils.isSamsungSelectKey(key) && !KeyboardUtils.isSamsungDeleteKey(key);
            }
        } else {
            return this.languagePrior.isValidKey(key);
        }
    }

    private boolean isFinished(int moveIdx) {
        if (this.doesEndWithDone) {
            return moveIdx >= this.moveSeq.length;
        } else {
            return moveIdx > this.moveSeq.length;
        }
    }

}
