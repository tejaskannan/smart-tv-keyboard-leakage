package smarttvsearch;

import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;

import org.json.JSONObject;
import org.json.JSONArray;
//import org.json.JSONException;

import smarttvsearch.creditcard.CreditCardDetails;
import smarttvsearch.prior.LanguagePrior;
import smarttvsearch.prior.LanguagePriorFactory;
import smarttvsearch.keyboard.MultiKeyboard;
import smarttvsearch.keyboard.KeyboardExtender;
import smarttvsearch.keyboard.NumericKeyboardExtender;
import smarttvsearch.search.Search;
import smarttvsearch.suboptimal.SuboptimalMoveModel;
import smarttvsearch.suboptimal.CreditCardMoveModel;
import smarttvsearch.utils.Direction;
import smarttvsearch.utils.FileUtils;
import smarttvsearch.utils.JsonUtils;
import smarttvsearch.utils.Move;
import smarttvsearch.utils.SmartTVType;
import smarttvsearch.utils.VectorUtils;
import smarttvsearch.utils.sounds.SmartTVSound;
import smarttvsearch.utils.sounds.SamsungSound;


public class SearchRunner {

    private static final int MAX_CCN_RANK = 100;
    private static final int MAX_EXPIRY_RANK = 5;
    private static final int MAX_CVV_RANK = 50;
    private static final int MAX_ZIP_RANK = 50;
    private static final int MAX_PASSWD_RANK = 500;

    public static void main(String[] args) {
        if (args.length != 3) {
            System.out.println("Must provide a path to (1) the move JSON file, (2) the folder containing the language prior, and (3) the name of the output file (json).");
            return;
        }

        // Unpack the arguments
        String movePath = args[0];
        String priorFolder = args[1];
        String outputPath = args[2];

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

        KeyboardExtender stndExtender = new KeyboardExtender(keyboard);

        // JSON Array to store the results
        JSONArray results = new JSONArray();

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

            KeyboardExtender numericExtender = new NumericKeyboardExtender(keyboard);

            for (int idx = 0; idx < jsonMoveSequences.length(); idx++) {
                // Unpack the credit card record and parse each field as a proper move sequence
                JSONObject creditCardRecord = jsonMoveSequences.getJSONObject(idx);
 
                Move[] ccnSeq = JsonUtils.parseMoveSeq(creditCardRecord.getJSONArray("credit_card"), tvType);
                Move[] zipSeq = JsonUtils.parseMoveSeq(creditCardRecord.getJSONArray("zip_code"), tvType);
                Move[] monthSeq = JsonUtils.parseMoveSeq(creditCardRecord.getJSONArray("exp_month"), tvType);
                Move[] yearSeq = JsonUtils.parseMoveSeq(creditCardRecord.getJSONArray("exp_year"), tvType);
                Move[] cvvSeq = JsonUtils.parseMoveSeq(creditCardRecord.getJSONArray("security_code"), tvType);

                // Get the labels for this index
                JSONObject labelsJson = creditCardLabels.getJSONObject(idx);

                for (int moveIdx = 0; moveIdx < ccnSeq.length; moveIdx++) {
                    System.out.printf("%d. %d (%s)\n", moveIdx + 1, ccnSeq[moveIdx].getNumMoves(), ccnSeq[moveIdx].getEndSound().getSoundName());
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
                List<String> ccnGuesses = recoverString(ccnSeq, keyboard, ccnPrior, keyboard.getStartKey(), tvType, false, false, false, numericExtender, 1e-3, 0, MAX_CCN_RANK);
                List<String> cvvGuesses = recoverString(cvvSeq, keyboard, cvvPrior, keyboard.getStartKey(), tvType, false, false, false, stndExtender, 0.5, 0, MAX_CVV_RANK);
                List<String> monthGuesses = recoverString(monthSeq, keyboard, monthPrior, keyboard.getStartKey(), tvType, false, false, false, stndExtender, 0.75, 0, MAX_EXPIRY_RANK);
                List<String> yearGuesses = recoverString(yearSeq, keyboard, yearPrior, keyboard.getStartKey(), tvType, false, false, false, stndExtender, 0.75, 0, MAX_EXPIRY_RANK);
                List<String> zipGuesses = recoverString(zipSeq, keyboard, zipPrior, keyboard.getStartKey(), tvType, false, false, false, stndExtender, 1e-3, 0, MAX_ZIP_RANK);

                CreditCardDetails truth = new CreditCardDetails(labelsJson.getString("credit_card"), labelsJson.getString("security_code"), labelsJson.getString("exp_month"), labelsJson.getString("exp_year"), labelsJson.getString("zip_code"));
                int rank = guessCreditCards(ccnGuesses, cvvGuesses, monthGuesses, yearGuesses, zipGuesses, truth);

                // Collect the output into a JSON Object
                JSONObject outputJson = new JSONObject();
                outputJson.put("ccn", JsonUtils.listToJsonArray(ccnGuesses));
                outputJson.put("cvv", JsonUtils.listToJsonArray(cvvGuesses));
                outputJson.put("exp_month", JsonUtils.listToJsonArray(monthGuesses));
                outputJson.put("exp_year", JsonUtils.listToJsonArray(yearGuesses));
                outputJson.put("zip", JsonUtils.listToJsonArray(zipGuesses));
                outputJson.put("rank", rank);

                results.put(outputJson);

                System.out.printf("Overall Rank: %d\n", rank);
                System.out.println("==========");
            }
        } else if (seqType.equals("standard")) {
            JSONArray jsonMoveSequences = serializedMoves.getJSONArray("move_sequences");
            JSONArray targetStrings = serializedLabels.getJSONArray("labels");

            String priorPath = FileUtils.joinPath(priorFolder, "rockyou-variable.db");
            LanguagePrior prior = LanguagePriorFactory.makePrior("ngram", priorPath);
            prior.build(false);

            for (int idx = 0; idx < jsonMoveSequences.length(); idx++) {
                Move[] moveSeq = JsonUtils.parseMoveSeq(jsonMoveSequences.getJSONArray(idx), tvType);

                for (Move move : moveSeq) {
                    System.out.println(move);
                }

                for (int moveIdx = 0; moveIdx < moveSeq.length; moveIdx++) {
                    System.out.printf("%d. ", moveIdx + 1);

                    int[] moveTimes = moveSeq[moveIdx].getMoveTimes();
                    List<Integer> moveDiffs = VectorUtils.getDiffs(moveTimes);

                    for (int j = 1; j < moveTimes.length; j++) {
                        int diff = moveTimes[j] - moveTimes[j - 1];
                        System.out.printf("%d ", diff);
                    }

                    System.out.println();
                }

                List<String> guesses = recoverString(moveSeq, keyboard, prior, keyboard.getStartKey(), tvType, true, true, true, stndExtender, 1e-2, 0, MAX_PASSWD_RANK);

                int rank = 1;
                boolean didFind = false;
                String groundTruth = targetStrings.getString(idx);

                for (String guess : guesses) {

                    if (guess.equals(groundTruth)) {
                        didFind = true;
                        System.out.printf("%d. %s (correct)\n", rank, guess);
                        break;
                    } else {
                        System.out.printf("%d. %s\n", rank, guess);
                    }

                    rank += 1;
                }

                if (didFind) {
                    System.out.printf("Correct Rank: %d\n", rank);
                }

                // Add the results to the output file
                JSONObject output = new JSONObject();
                output.put("guesses", JsonUtils.listToJsonArray(guesses));
                output.put("rank", rank);
                results.put(output);

                System.out.println("==========");
            }
        } else {
            throw new IllegalArgumentException("Invalid sequence type: " + seqType);
        }

        // Write the output to the provided JSON file
        FileUtils.writeJsonArray(results, outputPath);
    }

    //private static List<String> recoverString(Move[] moveSeq, MultiKeyboard keyboard, LanguagePrior prior, String startKey, SmartTVType tvType, boolean useDirections, int maxRank) {
    //    KeyboardExtender extender = new KeyboardExtender(keyboard);
    //    //SuboptimalMoveModel suboptimalModel = new SuboptimalMoveModel(moveSeq);
    //    SuboptimalMoveModel suboptimalModel = new CreditCardMoveModel(moveSeq, 1, 0.1);
    //    Search searcher = new Search(moveSeq, keyboard, prior, startKey, suboptimalModel, extender, tvType, useDirections, true, 5);

    //    List<String> result = new ArrayList<String>();

    //    for (int rank = 1; rank <= maxRank; rank++) {
    //        String guess = searcher.next();

    //        if (guess == null) {
    //            break;
    //        }

    //        result.add(guess);
    //    }

    //    return result;
    //}

    private static List<String> recoverString(Move[] moveSeq, MultiKeyboard keyboard, LanguagePrior prior, String startKey, SmartTVType tvType, boolean useDirections, boolean shouldLinkKeys, boolean doesSuggestDone, KeyboardExtender extender, double scoreFactor, int minCount, int maxRank) {
        HashSet<String> guessed = new HashSet<String>();

        List<String> result = new ArrayList<String>();

        int rank = 1;
        for (int numSuboptimal = 0; numSuboptimal < moveSeq.length; numSuboptimal++) {
            CreditCardMoveModel suboptimalModel = new CreditCardMoveModel(moveSeq, numSuboptimal, scoreFactor);
            Search searcher = new Search(moveSeq, keyboard, prior, startKey, suboptimalModel, extender, tvType, useDirections, shouldLinkKeys, doesSuggestDone, minCount);

            while (rank <= maxRank) {
                String guess = searcher.next();

                if (guess == null) {
                    break;
                }

                if (guessed.contains(guess)) {
                    continue;
                }

                result.add(guess);

                //System.out.printf("%d. %s\n", rank, guess);
                
                rank += 1;
                guessed.add(guess);
            }
        }

        return result;
    }

    private static int getCorrectRank(List<String> guesses, String groundTruth) {
        for (int rank = 0; rank < guesses.size(); rank++) {
            if (guesses.get(rank).equals(groundTruth)) {
                return rank + 1;
            }
        }

        return -1;
    }

    private static int guessCreditCards(List<String> ccnGuesses, List<String> cvvGuesses, List<String> monthGuesses, List<String> yearGuesses, List<String> zipCodeGuesses, CreditCardDetails groundTruth) {
        // Get the rank of each individual field for debugging
        int ccnRank = getCorrectRank(ccnGuesses, groundTruth.getCCN());
        int cvvRank = getCorrectRank(cvvGuesses, groundTruth.getCVV());
        int zipRank = getCorrectRank(zipCodeGuesses, groundTruth.getZipCode());
        int monthRank = getCorrectRank(monthGuesses, groundTruth.getExpMonth());
        int yearRank = getCorrectRank(yearGuesses, groundTruth.getExpYear());

        System.out.printf("CCN: %d / %d, CVV: %d / %d, Month: %d / %d, Year: %d / %d, Zip: %d / %d\n", ccnRank, ccnGuesses.size(), cvvRank, cvvGuesses.size(), monthRank, monthGuesses.size(), yearRank, yearGuesses.size(), zipRank, zipCodeGuesses.size());

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
}
