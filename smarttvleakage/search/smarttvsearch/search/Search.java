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
import smarttvsearch.prior.EnglishPrior;
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
import smarttvsearch.utils.sounds.AppleTVSound;


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
    private int maxNumCandidates;
    private boolean shouldUseSuggestions;

    private PriorityQueue<SearchState> frontier;
    private HashSet<VisitedState> visited;
    private HashSet<String> guessed;
    private boolean[] isMoveDeleted;
    private int numCandidates;

    private static final int DONE_SUGGESTION_COUNT = 8;
    private static final int SUGGESTIONS_MOVE_COUNT = 4;
    private static final int TOP_SUGGESTED_KEYS = 5;
    private static final double SCROLL_MISTAKE_FACTOR = 0.9;

    public Search(Move[] moveSeq, MultiKeyboard keyboard, LanguagePrior languagePrior, String startKey, SuboptimalMoveModel suboptimalModel, KeyboardExtender keyboardExtender, SmartTVType tvType, boolean useDirections, boolean shouldLinkKeys, boolean doesSuggestDone, int minCount, int maxNumCandidates, boolean shouldUseSuggestions) {
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
        this.maxNumCandidates = maxNumCandidates;
        this.shouldUseSuggestions = shouldUseSuggestions;

        this.frontier = new PriorityQueue<SearchState>();
        this.visited = new HashSet<VisitedState>();
        this.guessed = new HashSet<String>();
        this.numCandidates = 0;

        // Generate the initial state and place this on the queue
        SearchState initState = new SearchState(startKey, new ArrayList<String>(), 0.0, keyboard.getStartKeyboard(), 0);
        this.frontier.add(initState);

        // Check if the last move has a select sound. If so, then we end with a move to `done`
        this.doesEndWithDone = false;

        if (tvType == SmartTVType.SAMSUNG) {
            Move lastMove = moveSeq[moveSeq.length - 1];
            SamsungSound lastSound = (SamsungSound) lastMove.getEndSound();
            this.doesEndWithDone = lastSound.isSelect();
        } else if (tvType == SmartTVType.APPLE_TV) {
            Move lastMove = moveSeq[moveSeq.length - 1];
            AppleTVSound lastSound = (AppleTVSound) lastMove.getEndSound();
            this.doesEndWithDone = lastSound.isToolbarMove();
        }

        // Include the linked keys
        if (this.shouldLinkKeys) {
            List<KeyboardPosition> linkedStates = this.keyboard.getLinkedKeys(startKey, keyboard.getStartKeyboard());

            for (KeyboardPosition position : linkedStates) {
                SearchState candidateState = new SearchState(position.getKey(), new ArrayList<String>(), 0.0, position.getKeyboardName(), 0);
                this.frontier.add(candidateState);
            }
        }

        // Get the moves that are deleted. We avoid scoring such moves to bias the prior (they don't make it in the final result anyways)
        this.isMoveDeleted = SearchUtils.markDeletedMoves(moveSeq);
    }

    public String next() {
        while (!this.frontier.isEmpty()) {
            if ((this.maxNumCandidates > 0) && (this.numCandidates >= this.maxNumCandidates)) {
                return null;
            }

            // Get the top-ranked search state
            SearchState currentState = this.frontier.poll();
            int moveIdx = currentState.getMoveIdx();
            this.numCandidates += 1;

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
                HashSet<String> neighborsFromSuggestions = new HashSet<String>();

                // Password keyboards have a 'done' suggestion key which sometimes induces suboptimal moves in user
                // behavior (they have the explicitly clear the suggestion with a move)
                if (this.doesSuggestDone && (moveIdx >= DONE_SUGGESTION_COUNT) && (numMoves > 1)) {
                    numSuboptimal = Math.max(numSuboptimal, 1);
                } else if ((this.shouldUseSuggestions) && (moveIdx > 0)) {
                    numSuboptimal = Math.max(numSuboptimal, 1);  // Suggestions lead to an extra move to clear the result, so we expand the radius
                }

                // Each scroll has a mistake of +/- 1 based on challenges with audio parsing
                int scrollsBuffer = ((this.tvType == SmartTVType.APPLE_TV) && (!this.isFinished(moveIdx))) ? Math.max(2 * move.getNumScrolls(), 1) : move.getNumScrolls();
                //int scrollsBuffer = 0;
                numSuboptimal = Math.max(numSuboptimal, scrollsBuffer);

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

                    Set<String> neighbors = this.keyboard.getKeysForDistanceCumulative(currentKey, adjustedNumMoves, true, true, directions, currentState.getKeyboardName(), true);
                    double scoreFactor = this.suboptimalModel.getScoreFactor(offset);

                    // Users often make a single suboptimal move because the suggested key gets in the way
                    if (this.doesSuggestDone && (moveIdx >= DONE_SUGGESTION_COUNT) && (Math.abs(offset) == 1) && (numMoves > 1)) {
                        scoreFactor = 1.0;
                    } else if (this.shouldUseSuggestions && (Math.abs(offset) <= 1)) {
                        scoreFactor = 1.0; 
                    } else if ((offset != 0) && (Math.abs(offset) <= scrollsBuffer)) {
                        scoreFactor = SCROLL_MISTAKE_FACTOR;
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

                // Consider suggested keys if needed
                if ((this.shouldUseSuggestions) && (numMoves <= Search.SUGGESTIONS_MOVE_COUNT) && (moveIdx > 0)) {
                    EnglishPrior englishPrior = (EnglishPrior) this.languagePrior;  // Cast to english prior to get next most common letters
                    String[] nextMostCommon = englishPrior.nextMostCommon(currentState.toString(), Search.TOP_SUGGESTED_KEYS);

                    for (int commonIdx = 0; commonIdx < nextMostCommon.length; commonIdx++) {
                        String neighborKey = nextMostCommon[commonIdx];
                        neighborKeys.put(neighborKey, 1.0);
                        neighborsFromSuggestions.add(neighborKey);
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
                        
                        String keyLocation;
                        if (neighborsFromSuggestions.contains(neighborKey)) {
                            keyLocation = currentState.getCurrentKey();
                        } else {
                            keyLocation = neighborKey;
                        }

                        SearchState candidateState = new SearchState(keyLocation, nextKeys, 0.0, nextKeyboard, nextMoveIdx);
                        int incrementalCount = this.languagePrior.find(candidateState.toString());

                        if ((incrementalCount > this.minCount) || (this.isMoveDeleted[moveIdx]) || (this.isChangeKey(neighborKey))) {

                            double score = this.languagePrior.normalizeCount(incrementalCount);

                            if ((incrementalCount <= 0) || (this.isFinished(nextMoveIdx) && (SpecialKeys.DONE.equals(neighborKey))) || (this.isChangeKey(neighborKey)) || (this.isMoveDeleted[moveIdx])) {
                                score = 0.0;
                            } else {
                                score *= neighborKeys.get(neighborKey);
                                score = -1 * Math.log(score);
                            }

                            if (!this.shouldUseSuggestions) {
                                score = currentState.getScore() + score;  // Accumulate the score in log space
                            } else {
                                if (this.isFinished(nextMoveIdx)) {
                                    incrementalCount = this.languagePrior.find(candidateState.toString() + EnglishPrior.END_CHAR);
                                }

                                score = this.languagePrior.normalizeCount(incrementalCount) * neighborKeys.get(neighborKey);
                                score = -1 * Math.log(score);  // Mark the score as that of the string to this point
                            }

                            candidateState.setScore(score);
                            this.frontier.add(candidateState);
                            this.visited.add(visitedState);

                            if (this.shouldLinkKeys) {
                                List<KeyboardPosition> linkedStates = this.keyboard.getLinkedKeys(candidateState.getCurrentKey(), nextKeyboard);

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
        }  else if (this.tvType == SmartTVType.APPLE_TV) {
            AppleTVSound appleTVSound = (AppleTVSound) endSound;

            if (SpecialKeys.DONE.equals(key)) {
                return false;
            } else if (appleTVSound.isDelete()) {
                return KeyboardUtils.isAppleTVDeleteKey(key);
            } else {
                return this.languagePrior.isValidKey(key) && !KeyboardUtils.isAppleTVDeleteKey(key);
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

    private boolean isLastKeySelection(int moveIdx) {
        if (this.tvType == SmartTVType.APPLE_TV) {
            return (moveIdx == (this.moveSeq.length - 1));
        } else {
            Move lastMove = this.moveSeq[this.moveSeq.length - 1];
            SamsungSound endSound = (SamsungSound) lastMove.getEndSound();

            if (endSound.isSelect()) {
                return (moveIdx == (this.moveSeq.length - 2));
            } else {
                return (moveIdx == (this.moveSeq.length - 1));
            }
        }
    }

    private boolean isFinished(int moveIdx) {
        return moveIdx >= this.moveSeq.length;
    }

}
