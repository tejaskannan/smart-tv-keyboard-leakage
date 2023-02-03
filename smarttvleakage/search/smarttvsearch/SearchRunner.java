package smarttvsearch;

import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;

import org.json.JSONObject;
import org.json.JSONArray;
import org.json.JSONException;

import smarttvsearch.creditcard.CreditCardDetails;
import smarttvsearch.prior.LanguagePrior;
import smarttvsearch.prior.LanguagePriorFactory;
import smarttvsearch.keyboard.MultiKeyboard;
import smarttvsearch.keyboard.KeyboardExtender;
import smarttvsearch.keyboard.NumericKeyboardExtender;
import smarttvsearch.search.Search;
import smarttvsearch.suboptimal.SuboptimalMoveModel;
import smarttvsearch.suboptimal.SuboptimalMoveFactory;
import smarttvsearch.suboptimal.CreditCardMoveModel;
import smarttvsearch.utils.Direction;
import smarttvsearch.utils.FileUtils;
import smarttvsearch.utils.Move;
import smarttvsearch.utils.SmartTVType;
import smarttvsearch.utils.VectorUtils;
import smarttvsearch.utils.sounds.SmartTVSound;
import smarttvsearch.utils.sounds.SamsungSound;


public class SearchRunner {

    private static final int MAX_CCN_RANK = 100;
    private static final int MAX_EXPIRY_RANK = 5;
    private static final int MAX_CVV_RANK = 25;
    private static final int MAX_ZIP_RANK = 25;

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Must provide a path to (1) the move JSON file and (2) the folder containing the language prior.");
            return;
        }

        // Unpack the arguments
        String movePath = args[0];
        String priorFolder = args[1];

        JSONObject serializedMoves = FileUtils.readJsonObject(movePath);
        String seqType = serializedMoves.getString("seq_type");

        // Parse the TV Type
        String tvTypeName = serializedMoves.getString("tv_type").toUpperCase();
        SmartTVType tvType = SmartTVType.valueOf(tvTypeName);

        // Make the keyboard
        String keyboardFolder = FileUtils.joinPath("keyboard", tvTypeName.toLowerCase());
        MultiKeyboard keyboard = new MultiKeyboard(tvType, keyboardFolder);

        // Unpack the labels for reference. We do this for efficiency only, as we can stop the search when
        // we actually find the correct target string
        String labelsPath = movePath.substring(0, movePath.length() - 5) + "_labels.json";
        JSONObject serializedLabels = FileUtils.readJsonObject(labelsPath);

        // Determine the recovery process based on the sequence type
        if (seqType.equals("credit_card")) {
            JSONArray jsonMoveSequences = serializedMoves.getJSONArray("move_sequences");
            JSONArray creditCardLabels = serializedLabels.getJSONArray("labels");

            LanguagePrior ccnPrior = LanguagePriorFactory.makePrior("credit_card", null);
            LanguagePrior cvvPrior = LanguagePriorFactory.makePrior("numeric", null);
            LanguagePrior monthPrior = LanguagePriorFactory.makePrior("month", null);
            LanguagePrior yearPrior = LanguagePriorFactory.makePrior("year", null);
            LanguagePrior zipPrior = LanguagePriorFactory.makePrior("prefix", FileUtils.joinPath(priorFolder, "zip_codes.txt"));
 
            zipPrior.build(true);

            for (int idx = 0; idx < jsonMoveSequences.length(); idx++) {
                // Unpack the credit card record and parse each field as a proper move sequence
                JSONObject creditCardRecord = jsonMoveSequences.getJSONObject(idx);
 
                Move[] ccnSeq = parseMoveSeq(creditCardRecord.getJSONArray("credit_card"), tvType);
                Move[] zipSeq = parseMoveSeq(creditCardRecord.getJSONArray("zip_code"), tvType);
                Move[] monthSeq = parseMoveSeq(creditCardRecord.getJSONArray("exp_month"), tvType);
                Move[] yearSeq = parseMoveSeq(creditCardRecord.getJSONArray("exp_year"), tvType);
                Move[] cvvSeq = parseMoveSeq(creditCardRecord.getJSONArray("security_code"), tvType);

                // Get the labels for this index
                JSONObject labelsJson = creditCardLabels.getJSONObject(idx);

                for (int moveIdx = 0; moveIdx < ccnSeq.length; moveIdx++) {
                    System.out.printf("%d. %d\n", moveIdx + 1, ccnSeq[moveIdx].getNumMoves());
                }
                System.out.println();

                List<Integer> diffs = new ArrayList<Integer>();

                for (int moveIdx = 0; moveIdx < ccnSeq.length; moveIdx++) {
                    System.out.printf("%d. ", moveIdx + 1);

                    int[] moveTimes = ccnSeq[moveIdx].getMoveTimes();
                    List<Integer> moveDiffs = VectorUtils.getDiffs(moveTimes);

                    if (moveDiffs != null) {
                        diffs.addAll(moveDiffs);
                    }

                    for (int j = 1; j < moveTimes.length; j++) {
                        int diff = moveTimes[j] - moveTimes[j - 1];

                        System.out.printf("%d ", diff);
                    }

                    System.out.println();
                }

                // Recover each field
                List<String> ccnGuesses = recoverCreditCard(ccnSeq, keyboard, ccnPrior, keyboard.getStartKey(), tvType, labelsJson.getString("credit_card"), MAX_CCN_RANK);
                List<String> cvvGuesses = recoverString(cvvSeq, keyboard, cvvPrior, keyboard.getStartKey(), tvType, "standard", false, true, labelsJson.getString("security_code"), MAX_CVV_RANK);
                List<String> monthGuesses = recoverString(monthSeq, keyboard, monthPrior, keyboard.getStartKey(), tvType, "standard", false, true, labelsJson.getString("exp_month"), MAX_EXPIRY_RANK);
                List<String> yearGuesses = recoverString(yearSeq, keyboard, yearPrior, keyboard.getStartKey(), tvType, "standard", false, true, labelsJson.getString("exp_year"), MAX_EXPIRY_RANK);
                List<String> zipGuesses = recoverString(zipSeq, keyboard, zipPrior, keyboard.getStartKey(), tvType, "zip", false, true, labelsJson.getString("zip_code"), MAX_ZIP_RANK);

                CreditCardDetails truth = new CreditCardDetails(labelsJson.getString("credit_card"), labelsJson.getString("security_code"), labelsJson.getString("exp_month"), labelsJson.getString("exp_year"), labelsJson.getString("zip_code"));
                int rank = guessCreditCards(ccnGuesses, cvvGuesses, monthGuesses, yearGuesses, zipGuesses, truth);

                System.out.printf("Overall Rank: %d\n", rank);
                System.out.println("==========");
            }
        } else {
            throw new IllegalArgumentException("Invalid sequence type: " + seqType);
        }
    }

    private static List<String> recoverString(Move[] moveSeq, MultiKeyboard keyboard, LanguagePrior prior, String startKey, SmartTVType tvType, String suboptimalModelName, boolean useDirections, boolean shouldAccumulateScore, String target, int maxRank) {
        KeyboardExtender extender = new KeyboardExtender(keyboard);
        SuboptimalMoveModel suboptimalModel = SuboptimalMoveFactory.make(suboptimalModelName, moveSeq);
        Search searcher = new Search(moveSeq, keyboard, prior, startKey, suboptimalModel, extender, tvType, useDirections, shouldAccumulateScore);

        List<String> result = new ArrayList<String>();

        for (int rank = 1; rank <= maxRank; rank++) {
            String guess = searcher.next();

            if (guess == null) {
                break;
            }

            result.add(guess);

            if (guess.equals(target)) {
                System.out.printf("%d. %s (correct)\n", rank, guess);
            } else {
                System.out.printf("%d. %s\n", rank, guess);
            }
        }

        return result;
    }

    private static List<String> recoverCreditCard(Move[] moveSeq, MultiKeyboard keyboard, LanguagePrior prior, String startKey, SmartTVType tvType, String target, int maxRank) {
        HashSet<String> guessed = new HashSet<String>();
        KeyboardExtender extender = new NumericKeyboardExtender(keyboard);

        List<String> result = new ArrayList<String>();

        int rank = 1;
        for (int numSuboptimal = 0; numSuboptimal < moveSeq.length; numSuboptimal++) {
            CreditCardMoveModel suboptimalModel = new CreditCardMoveModel(moveSeq, numSuboptimal);
            Search searcher = new Search(moveSeq, keyboard, prior, startKey, suboptimalModel, extender, tvType, false, true);

            while (rank <= maxRank) {
                String guess = searcher.next();

                if (guess == null) {
                    break;
                }

                if (guessed.contains(guess)) {
                    continue;
                }

                result.add(guess);

                if (guess.equals(target)) {
                    System.out.printf("%d. %s (correct)\n", rank, guess);
                } else {
                    System.out.printf("%d. %s\n", rank, guess);
                }

                rank += 1;
                guessed.add(guess);
            }
        }

        return result;
    }

    private static int guessCreditCards(List<String> ccnGuesses, List<String> cvvGuesses, List<String> monthGuesses, List<String> yearGuesses, List<String> zipCodeGuesses, CreditCardDetails groundTruth) {
        // First, iterate through first 25 CCNs, first 5 CVVs, first 5 ZIPs, and first 2 months / years
        int ccnLimit = Math.min(25, ccnGuesses.size());
        int cvvLimit = Math.min(5, cvvGuesses.size());
        int zipLimit = Math.min(5, zipCodeGuesses.size());
        int monthLimit = Math.min(2, monthGuesses.size());
        int yearLimit = Math.min(2, yearGuesses.size());

        int rank = 1;
        CreditCardDetails guess;
        HashSet<CreditCardDetails> guesses = new HashSet<CreditCardDetails>();

        for (int yearIdx = 0; yearIdx < yearLimit; yearIdx++) {
            for (int monthIdx = 0; monthIdx < monthLimit; monthIdx++) {
                for (int zipIdx = 0; zipIdx < zipLimit; zipIdx++) {
                    for (int cvvIdx = 0; cvvIdx < cvvLimit; cvvIdx++) {
                        for (int ccnIdx = 0; ccnIdx < ccnLimit; ccnIdx++) {
                            guess = new CreditCardDetails(ccnGuesses.get(ccnIdx), cvvGuesses.get(cvvIdx), monthGuesses.get(monthIdx), yearGuesses.get(yearIdx), zipCodeGuesses.get(zipIdx));

                            if (guess.equals(groundTruth)) {
                                return rank;
                            }

                            guesses.add(guess);
                            rank += 1;
                        }
                    }
                } 
            }
        }

        // Second, search over the entire element set in a sequential manner, avoiding the same guess twice
        for (int yearIdx = 0; yearIdx < yearGuesses.size(); yearIdx++) {
            for (int monthIdx = 0; monthIdx < monthGuesses.size(); monthIdx++) {
                for (int zipIdx = 0; zipIdx < zipCodeGuesses.size(); zipIdx++) {
                    for (int cvvIdx = 0; cvvIdx < cvvGuesses.size(); cvvIdx++) {
                        for (int ccnIdx = 0; ccnIdx < ccnGuesses.size(); ccnIdx++) {
                            guess = new CreditCardDetails(ccnGuesses.get(ccnIdx), cvvGuesses.get(cvvIdx), monthGuesses.get(monthIdx), yearGuesses.get(yearIdx), zipCodeGuesses.get(zipIdx));

                            if (guesses.contains(guess)) {
                                continue;
                            }

                            if (guess.equals(groundTruth)) {
                                return rank;
                            }

                            rank += 1;
                            guesses.add(guess);
                        }
                    }
                } 
            }
        }

        return -1;
    }

    private static Move[] parseMoveSeq(JSONArray jsonMoveSeq, SmartTVType tvType) {
        Move[] moveSeq = new Move[jsonMoveSeq.length()];

        for (int idx = 0; idx < jsonMoveSeq.length(); idx++) {
            moveSeq[idx] = parseMove(jsonMoveSeq.getJSONObject(idx), tvType);
        }

        return moveSeq;
    }

    private static Move parseMove(JSONObject jsonMove, SmartTVType tvType) {
        int numMoves = jsonMove.getInt("num_moves");
        int startTime = jsonMove.getInt("start_time");
        int endTime = jsonMove.getInt("end_time");

        SmartTVSound endSound;
        if (tvType == SmartTVType.SAMSUNG) {
            endSound = new SamsungSound(jsonMove.getString("end_sound"));
        } else {
            throw new IllegalArgumentException("Cannot parse sound for tv: " + tvType.name());
        }

        int[] moveTimes = new int[numMoves];
        JSONArray jsonMoveTimes = jsonMove.getJSONArray("move_times");

        for (int timeIdx = 0; timeIdx < numMoves; timeIdx++) {
            moveTimes[timeIdx] = jsonMoveTimes.getInt(timeIdx);
        }

        try {
            JSONArray directionsArray = jsonMove.getJSONArray("directions");
            Direction[] directions = new Direction[directionsArray.length()];

            for (int idx = 0; idx < directionsArray.length(); idx++) {
                directions[idx] = Direction.valueOf(directionsArray.getString(idx).toUpperCase());
            }

            return new Move(numMoves, directions, endSound, startTime, endTime, moveTimes);
        } catch (JSONException ex) {
            return new Move(numMoves, endSound, startTime, endTime, moveTimes);
        }
    }

}
