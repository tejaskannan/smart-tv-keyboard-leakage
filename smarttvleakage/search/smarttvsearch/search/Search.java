package smarttvsearch.search;

import java.util.PriorityQueue;
import java.util.List;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.ArrayList;

import smarttvsearch.keyboard.MultiKeyboard;
import smarttvsearch.keyboard.KeyboardExtender;
import smarttvsearch.keyboard.KeyboardPosition;
import smarttvsearch.prior.LanguagePrior;
import smarttvsearch.suboptimal.SuboptimalMoveModel;
import smarttvsearch.utils.sounds.SmartTVSound;
import smarttvsearch.utils.sounds.SamsungSound;
import smarttvsearch.utils.search.SearchState;
import smarttvsearch.utils.search.VisitedState;
import smarttvsearch.utils.KeyboardUtils;
import smarttvsearch.utils.SpecialKeys;
import smarttvsearch.utils.Move;
import smarttvsearch.utils.Direction;
import smarttvsearch.utils.SpecialKeys;
import smarttvsearch.utils.SmartTVType;
import smarttvsearch.utils.SearchUtils;
import smarttvsearch.utils.sounds.SamsungSound;


public class Search {

    private Move[] moveSeq;
    private MultiKeyboard keyboard;
    private LanguagePrior languagePrior;
    private String startKey;
    private SuboptimalMoveModel suboptimalModel;
    private KeyboardExtender keyboardExtender;
    private boolean doesEndWithDone;
    private boolean useDirections;
    private boolean shouldLinkKeys;
    private boolean doesSuggestDone;
    private int minCount;
    private SmartTVType tvType;

    private PriorityQueue<SearchState> frontier;
    private HashSet<VisitedState> visited;
    private HashSet<String> guessed;
    private boolean[] isMoveDeleted;

    private static int DONE_SUGGESTION_COUNT = 8;

    public Search(Move[] moveSeq, MultiKeyboard keyboard, LanguagePrior languagePrior, String startKey, SuboptimalMoveModel suboptimalModel, KeyboardExtender keyboardExtender, SmartTVType tvType, boolean useDirections, boolean shouldLinkKeys, boolean doesSuggestDone, int minCount) {
        this.moveSeq = moveSeq;
        this.keyboard = keyboard;
        this.languagePrior = languagePrior;
        this.startKey = startKey;
        this.suboptimalModel = suboptimalModel;
        this.keyboardExtender = keyboardExtender;
        this.useDirections = useDirections;
        this.shouldLinkKeys = shouldLinkKeys;
        this.tvType = tvType;
        this.minCount = minCount;
        this.doesSuggestDone = doesSuggestDone;

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

        // Get the moves that are deleted. We avoid scoring such moves to bias the prior (they don't make it in the final result anyways)
        this.isMoveDeleted = SearchUtils.markDeletedMoves(moveSeq);
    }

    public String next() {
        while (!this.frontier.isEmpty()) {
            // Get the top-ranked search state
            SearchState currentState = this.frontier.poll();
            int moveIdx = currentState.getMoveIdx();

            // Check if we are out of moves. If so, return the string if not guessed already.
            if (this.isFinished(moveIdx))  {
                String guess = currentState.toString();

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

                //boolean isFirstMoveAndZero = (moveIdx == 0) && (numMoves == 0);

                if (this.doesSuggestDone && (moveIdx >= DONE_SUGGESTION_COUNT) && (numMoves > 1)) {
                    numSuboptimal = Math.max(numSuboptimal, 1);
                }

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

                    // Users often make a single suboptimal move because the suggested key gets in the way
                    if (this.doesSuggestDone && (moveIdx >= DONE_SUGGESTION_COUNT) && (Math.abs(offset) == 1) && (numMoves > 1)) {
                        scoreFactor = 1.0;
                    }

                    Set<String> extendedNeighbors = this.keyboardExtender.getExtendedNeighbors(currentKey, adjustedNumMoves, currentState.getKeyboardName());
                    neighbors.addAll(extendedNeighbors);
   
                    for (String neighbor : neighbors) {
                        Double prevFactor = neighborKeys.get(neighbor);

                        if ((prevFactor == null) || (scoreFactor > prevFactor)) {
                            neighborKeys.put(neighbor, scoreFactor);
                        }
                    }
                }

                int nextMoveIdx = moveIdx + 1;

                for (String neighborKey : neighborKeys.keySet()) {
                    if (!this.isValidKey(neighborKey, nextMoveIdx, move.getEndSound(), keyboard, currentState.getKeyboardName()) && !this.isMoveDeleted[moveIdx]) {
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

                        if ((incrementalCount > this.minCount) || (this.isMoveDeleted[moveIdx]) || (this.isChangeKey(neighborKey))) {

                            double score = this.languagePrior.normalizeCount(incrementalCount);

                            if ((incrementalCount <= 0) || (this.isFinished(nextMoveIdx) && (SpecialKeys.DONE.equals(neighborKey))) || (this.isChangeKey(neighborKey)) || (this.isMoveDeleted[moveIdx])) {
                                score = 0.0;
                            } else {
                                score *= neighborKeys.get(neighborKey);
                                score = -1 * Math.log(score);
                            }

                            // Accumulate the score (in log space)
                            score = currentState.getScore() + score;

                            candidateState.setScore(score);
                            this.frontier.add(candidateState);
                            this.visited.add(visitedState);

                            if (this.shouldLinkKeys) {
                                List<KeyboardPosition> linkedStates = this.keyboard.getLinkedKeys(neighborKey, nextKeyboard);

                                for (KeyboardPosition position : linkedStates) {
                                    visitedState = new VisitedState(nextKeys, position.getKeyboardName());

                                    if (!this.visited.contains(visitedState)) {
                                        candidateState = new SearchState(position.getKey(), nextKeys, score, position.getKeyboardName(), nextMoveIdx);
                                        this.frontier.add(candidateState);
                                        this.visited.add(visitedState);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return null;
    }

    private boolean isValidKey(String key, int nextMoveIdx, SmartTVSound endSound, MultiKeyboard keyboard, String keyboardName) {
        if (this.isFinished(nextMoveIdx) && this.doesEndWithDone) {
            return key.equals(SpecialKeys.DONE);
        } else if (!keyboard.isClickable(key, keyboardName)) {
            return false;
        } else if (this.tvType == SmartTVType.SAMSUNG) {
            SamsungSound endSamsungSound = (SamsungSound) endSound;

            if (SpecialKeys.DONE.equals(key)) {
                return false;  // We are not at the end, so a DONE key makes no sense
            } else if (endSamsungSound.isSelect()) {
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

    private boolean isChangeKey(String key) {
        if (this.tvType == SmartTVType.SAMSUNG) {
            return key.equals(SpecialKeys.CHANGE) || key.equals(SpecialKeys.NEXT);
        }
        return false;
    }

    private boolean isFinished(int moveIdx) {
        return moveIdx >= this.moveSeq.length;
        //if (this.doesEndWithDone) {
        //    return moveIdx >= this.moveSeq.length;
        //} else {
        //    return moveIdx > this.moveSeq.length;
        //}
    }

}
