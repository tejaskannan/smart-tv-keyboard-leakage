package smarttvsearch;

import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;

import org.json.JSONObject;
import org.json.JSONArray;
//import org.json.JSONException;

import smarttvsearch.creditcard.CreditCardDetails;
import smarttvsearch.creditcard.CreditCardRank;
import smarttvsearch.creditcard.CreditCardUtils;
import smarttvsearch.prior.LanguagePrior;
import smarttvsearch.prior.LanguagePriorFactory;
import smarttvsearch.keyboard.MultiKeyboard;
import smarttvsearch.keyboard.KeyboardExtender;
import smarttvsearch.keyboard.NumericKeyboardExtender;
import smarttvsearch.search.Search;
import smarttvsearch.search.SearchResult;
import smarttvsearch.suboptimal.SuboptimalMoveModel;
import smarttvsearch.suboptimal.CreditCardMoveModel;
import smarttvsearch.utils.Direction;
import smarttvsearch.utils.FileUtils;
import smarttvsearch.utils.JsonUtils;
import smarttvsearch.utils.KeyboardType;
import smarttvsearch.utils.Move;
import smarttvsearch.utils.SearchArguments;
import smarttvsearch.utils.SpecialKeys;
import smarttvsearch.utils.SmartTVType;
import smarttvsearch.utils.VectorUtils;
import smarttvsearch.utils.sounds.SmartTVSound;
import smarttvsearch.utils.sounds.SamsungSound;


public class SearchRunner {

    private static final int MAX_CCN_RANK = 100;
    private static final int MAX_EXPIRY_RANK = 5;
    private static final int MAX_CVV_RANK = 50;
    private static final int MAX_ZIP_RANK = 50;
    private static final int MAX_PASSWD_RANK = 250;
    private static final int MAX_NUM_CANDIDATES = 500000;

    public static void main(String[] args) {
        // Parse the arguments
        SearchArguments searchArgs = new SearchArguments(args);

        // Read the move JSON file
        JSONObject serializedMoves = FileUtils.readJsonObject(searchArgs.getInputFile());
        String seqType = serializedMoves.getString("seq_type");

        // Parse the TV Type
        String tvTypeName = serializedMoves.getString("tv_type").toUpperCase();
        SmartTVType tvType = SmartTVType.valueOf(tvTypeName);

        // Parse the keyboard type
        String keyboardTypeName = serializedMoves.getString("keyboard_type");
        KeyboardType keyboardType;

        if (keyboardTypeName == null) {
            keyboardType = KeyboardType.valueOf(tvTypeName);
        } else {
            keyboardType = KeyboardType.valueOf(keyboardTypeName.toUpperCase());
        }

        // Make the keyboard
        String keyboardFolder = FileUtils.joinPath("keyboard", keyboardType.name().toLowerCase());
        MultiKeyboard keyboard = new MultiKeyboard(keyboardType, keyboardFolder);

        // Unpack the labels for reference. We do this for efficiency only, as we can stop the search when
        // we actually find the correct target string
        JSONObject serializedLabels = FileUtils.readJsonObject(searchArgs.getLabelsFile());

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

            LanguagePrior zipPrior = LanguagePriorFactory.makePrior("prefix", searchArgs.getZipPrior());
            zipPrior.build(true);

            KeyboardExtender numericExtender = new NumericKeyboardExtender(keyboard);
            int numFound = 0;

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

                // Recover each field
                List<SearchResult> ccnResults = recover(ccnSeq, keyboard, ccnPrior, keyboard.getStartKey(), tvType, false, false, false, numericExtender, 1e-3, 0, MAX_CCN_RANK, false, searchArgs.shouldUseSuboptimal(), false, MAX_NUM_CANDIDATES, "");
                //List<SearchResult> ccnResults = recoverExhaustive(ccnSeq, keyboard, ccnPrior, keyboard.getStartKey(), tvType, false, false, false, numericExtender, 1e-3, 0, MAX_CCN_RANK, false, searchArgs.shouldUseSuboptimal(), MAX_NUM_CANDIDATES, "");
                List<SearchResult> cvvResults = recover(cvvSeq, keyboard, cvvPrior, keyboard.getStartKey(), tvType, false, false, false, stndExtender, 0.5, 0, MAX_CVV_RANK, false, searchArgs.shouldUseSuboptimal(), false, -1, "");
                List<SearchResult> monthResults = recover(monthSeq, keyboard, monthPrior, keyboard.getStartKey(), tvType, false, false, false, stndExtender, 0.75, 0, MAX_EXPIRY_RANK, false, searchArgs.shouldUseSuboptimal(), false, -1, "");
                List<SearchResult> yearResults = recover(yearSeq, keyboard, yearPrior, keyboard.getStartKey(), tvType, false, false, false, stndExtender, 0.75, 0, MAX_EXPIRY_RANK, false, searchArgs.shouldUseSuboptimal(), false, -1, "");
                List<SearchResult> zipResults = recover(zipSeq, keyboard, zipPrior, keyboard.getStartKey(), tvType, false, false, false, stndExtender, 1e-3, 0, MAX_ZIP_RANK, false, searchArgs.shouldUseSuboptimal(), false, -1, "");

                // Unpack the guesses
                List<String> ccnGuesses = SearchResult.toGuessList(ccnResults);
                List<String> cvvGuesses = SearchResult.toGuessList(cvvResults);
                List<String> monthGuesses = SearchResult.toGuessList(monthResults);
                List<String> yearGuesses = SearchResult.toGuessList(yearResults);
                List<String> zipGuesses = SearchResult.toGuessList(zipResults);

                CreditCardDetails truth = new CreditCardDetails(labelsJson.getString("credit_card"), labelsJson.getString("security_code"), labelsJson.getString("exp_month"), labelsJson.getString("exp_year"), labelsJson.getString("zip_code"));
                int rank = guessCreditCards(ccnGuesses, cvvGuesses, monthGuesses, yearGuesses, zipGuesses, truth);

                int numPossibleCcns = 0;
                int numCcnCandidates = 0;
                if (!ccnGuesses.isEmpty()) {
                    SearchResult finalResult = ccnResults.get(ccnResults.size() - 1);
                    numPossibleCcns = finalResult.getNumPossibleGuesses();
                    numCcnCandidates = finalResult.getNumCandidates();
                }

                // Collect the output into a JSON Object
                JSONObject outputJson = new JSONObject();
                outputJson.put("ccn", JsonUtils.listToJsonArray(ccnGuesses));
                outputJson.put("cvv", JsonUtils.listToJsonArray(cvvGuesses));
                outputJson.put("exp_month", JsonUtils.listToJsonArray(monthGuesses));
                outputJson.put("exp_year", JsonUtils.listToJsonArray(yearGuesses));
                outputJson.put("zip", JsonUtils.listToJsonArray(zipGuesses));
                outputJson.put("rank", rank);
                outputJson.put("numPossibleCcns", numPossibleCcns);
                outputJson.put("numCcnCandidates", numCcnCandidates);

                results.put(outputJson);

                numFound += ((rank > 0) ? 1 : 0);
                System.out.printf("Completed %d / %d entries. Accuracy: %.4f\r", idx + 1, jsonMoveSequences.length(), ((double) numFound) / ((double) (idx + 1)));
            }
        } else if (seqType.equals("standard")) {
            JSONArray jsonMoveSequences = serializedMoves.getJSONArray("move_sequences");
            JSONArray jsonSuggestionsTypes = serializedMoves.getJSONArray("suggestions_types");
            JSONArray targetStrings = serializedLabels.getJSONArray("labels");

            LanguagePrior passwordPrior = LanguagePriorFactory.makePrior("ngram", searchArgs.getPasswordPrior());
            LanguagePrior englishPrior = LanguagePriorFactory.makePrior("english", searchArgs.getEnglishPrior());

            passwordPrior.build(false);
            englishPrior.build(false);

            double suboptimalFactor = (tvType == SmartTVType.APPLE_TV) ? 0.5 : 1e-2;
            int correctCount = 0;
            int totalCount = 0;

            for (int idx = 0; idx < jsonMoveSequences.length(); idx++) {
                Move[] moveSeq = JsonUtils.parseMoveSeq(jsonMoveSequences.getJSONArray(idx), tvType);

                // Determine the suggestions type
                String suggestionsType = jsonSuggestionsTypes.getString(idx);
                boolean shouldUseSuggestions = suggestionsType.equals("suggestions") || searchArgs.shouldForceSuggestions();
                LanguagePrior prior = shouldUseSuggestions ? englishPrior : passwordPrior;
                String groundTruth = targetStrings.getString(idx);

                List<SearchResult> searchResults = recover(moveSeq, keyboard, prior, keyboard.getStartKey(), tvType, searchArgs.shouldUseDirections(), true, true, stndExtender, suboptimalFactor, 0, MAX_PASSWD_RANK, shouldUseSuggestions, searchArgs.shouldUseSuboptimal(), false, MAX_NUM_CANDIDATES, groundTruth);
                List<String> guesses = SearchResult.toGuessList(searchResults);
                List<Double> scores = SearchResult.toScoreList(searchResults);

                int rank = 1;
                boolean didFind = false;

                for (String guess : guesses) {
                    if (guess.equals(groundTruth)) {
                        didFind = true;
                        break;
                    } 

                    rank += 1;
                }

                if (didFind) {
                    correctCount += 1;
                } else {
                    rank = -1;
                }

                totalCount += 1;

                int numCandidates = -1;
                if (searchResults.size() > 0) {
                    numCandidates = searchResults.get(searchResults.size() - 1).getNumCandidates();
                }

                if (!didFind) {
                    System.out.println(groundTruth);
                }

                // Add the results to the output file
                JSONObject output = new JSONObject();
                output.put("guesses", JsonUtils.listToJsonArray(guesses));
                output.put("scores", JsonUtils.listToJsonArray(scores));
                output.put("rank", rank);
                output.put("numCandidates", numCandidates);
                output.put("shouldUseSuggestions", shouldUseSuggestions);
                results.put(output);

                System.out.printf("Completed %d. Accuracy: %d / %d (%.4f)\r", totalCount, correctCount, totalCount, ((double) correctCount) / ((double) totalCount));
            }
        } else if (seqType.equals("credit_card_reverse")) {
            JSONArray jsonMoveSequences = serializedMoves.getJSONArray("move_sequences");
            JSONArray creditCardLabels = serializedLabels.getJSONArray("labels");

            LanguagePrior ccnPrior = LanguagePriorFactory.makePrior("reverse_credit_card", null);
            KeyboardExtender numericExtender = new NumericKeyboardExtender(keyboard);
            int numFound = 0;

            for (int idx = 0; idx < jsonMoveSequences.length(); idx++) {
                // Unpack the credit card record
                JSONObject creditCardRecord = jsonMoveSequences.getJSONObject(idx);
                Move[] ccnSeq = JsonUtils.parseMoveSeq(creditCardRecord.getJSONArray("credit_card"), tvType);
                
                // Reverse the sequence to perform search in opposite direction
                Move[] reversed = SearchRunner.reverseMoveSeq(ccnSeq);

                // Get the labels for this index
                JSONObject labelsJson = creditCardLabels.getJSONObject(idx);

                // Recover the credit card number field
                List<SearchResult> ccnResults = recover(reversed, keyboard, ccnPrior, SpecialKeys.DONE, tvType, false, false, false, numericExtender, 1e-3, 0, MAX_CCN_RANK, false, searchArgs.shouldUseSuboptimal(), true, MAX_NUM_CANDIDATES, "");

                // Unpack the guesses
                List<String> ccnGuesses = SearchResult.toGuessList(ccnResults);
                int rank = getCorrectRank(ccnGuesses, labelsJson.getString("credit_card"));

                // Collect the output into a JSON Object
                JSONObject outputJson = new JSONObject();
                outputJson.put("ccn", JsonUtils.listToJsonArray(ccnGuesses));
                outputJson.put("rank", rank);
                results.put(outputJson);

                numFound += ((rank > 0) ? 1 : 0);
                System.out.printf("Completed %d / %d entries. Accuracy: %.4f\r", idx + 1, jsonMoveSequences.length(), ((double) numFound) / ((double) (idx + 1)));
            }
        
        } else if (seqType.equals("credit_card_bubble")) {
            JSONArray jsonMoveSequences = serializedMoves.getJSONArray("move_sequences");
            JSONArray creditCardLabels = serializedLabels.getJSONArray("labels");

            LanguagePrior prior = LanguagePriorFactory.makePrior("numeric", null);
            KeyboardExtender numericExtender = new NumericKeyboardExtender(keyboard);
            int numFound = 0;

            for (int idx = 0; idx < jsonMoveSequences.length(); idx++) {
                // Unpack the credit card record
                JSONObject creditCardRecord = jsonMoveSequences.getJSONObject(idx);
                Move[] cvvSeq = JsonUtils.parseMoveSeq(creditCardRecord.getJSONArray("security_code"), tvType);

                // Get the labels for this index
                JSONObject labelsJson = creditCardLabels.getJSONObject(idx);
                String groundTruth = labelsJson.getString("security_code");

                // Recover the credit card number field
                List<SearchResult> cvvResults = recoverExhaustive(cvvSeq, keyboard, prior, SpecialKeys.DONE, tvType, false, false, false, numericExtender, 1e-3, 0, MAX_CCN_RANK, false, searchArgs.shouldUseSuboptimal(), MAX_NUM_CANDIDATES, groundTruth);

                // Unpack the guesses
                List<String> cvvGuesses = SearchResult.toGuessList(cvvResults);
                int rank = getCorrectRank(cvvGuesses, groundTruth);

                // Collect the output into a JSON Object
                JSONObject outputJson = new JSONObject();
                outputJson.put("security_code", JsonUtils.listToJsonArray(cvvGuesses));
                outputJson.put("rank", rank);
                results.put(outputJson);

                numFound += ((rank > 0) ? 1 : 0);
                System.out.printf("Completed %d / %d entries. Accuracy: %.4f\r", idx + 1, jsonMoveSequences.length(), ((double) numFound) / ((double) (idx + 1)));
            }
        } else {
            throw new IllegalArgumentException("Invalid sequence type: " + seqType);
        }

        // Write the output to the provided JSON file
        FileUtils.writeJsonArray(results, searchArgs.getOutputFile());
        System.out.println();
    }

    private static List<SearchResult> recover(Move[] moveSeq, MultiKeyboard keyboard, LanguagePrior prior, String startKey, SmartTVType tvType, boolean useDirections, boolean shouldLinkKeys, boolean doesSuggestDone, KeyboardExtender extender, double scoreFactor, int minCount, int maxRank, boolean shouldUseSuggestions, boolean shouldUseSuboptimal, boolean shouldReverse, int maxNumCandidates, String groundTruth) {
        HashSet<String> guessed = new HashSet<String>();

        List<SearchResult> configResults;
        List<SearchResult> results = new ArrayList<SearchResult>();

        int maxNumSuboptimal = Math.min(moveSeq.length, 10);
        if (!shouldUseSuboptimal) {
            maxNumSuboptimal = 1;
        }

        int rank = 1;
        int numCandidates = 0;

        if (tvType == SmartTVType.APPLE_TV) {
            CreditCardMoveModel suboptimalModel = new CreditCardMoveModel(moveSeq, 0, scoreFactor, tvType);
            Search searcher = new Search(moveSeq, keyboard, prior, startKey, suboptimalModel, extender, tvType, useDirections, shouldLinkKeys, doesSuggestDone, minCount, maxNumCandidates, shouldUseSuggestions, 0, shouldReverse);
            
            configResults = recoverForConfig(searcher, guessed, groundTruth, rank, maxRank);
            results.addAll(configResults);

            // Include the number of candidates we have seen up to this point
            for (SearchResult result : configResults) {
                result.addNumCandidates(numCandidates);
            }

            if (!configResults.isEmpty()) {
                SearchResult finalResult = configResults.get(configResults.size() - 1);
                
                if (finalResult.getGuess().equals(groundTruth)) {
                    return results;
                }

                numCandidates = finalResult.getNumCandidates();
            } else {
                System.out.println("Returned 0 results.");
            }
        }

        int minNumSuboptimal = (tvType == SmartTVType.APPLE_TV) ? 1 : 0;

        for (int numSuboptimal = 0; numSuboptimal < maxNumSuboptimal; numSuboptimal++) {
            CreditCardMoveModel suboptimalModel = new CreditCardMoveModel(moveSeq, numSuboptimal, scoreFactor, tvType);
            Search searcher = new Search(moveSeq, keyboard, prior, startKey, suboptimalModel, extender, tvType, useDirections, shouldLinkKeys, doesSuggestDone, minCount, maxNumCandidates, shouldUseSuggestions, minNumSuboptimal, shouldReverse);
            rank = results.size() + 1;

            configResults = recoverForConfig(searcher, guessed, groundTruth, rank, maxRank);
            results.addAll(configResults);

            if (!configResults.isEmpty()) {
                SearchResult finalResult = configResults.get(configResults.size() - 1);
                
                if (finalResult.getGuess().equals(groundTruth)) {
                    return results;
                }
            }
        }

        return results;
    }

    private static List<SearchResult> recoverExhaustive(Move[] moveSeq, MultiKeyboard keyboard, LanguagePrior prior, String startKey, SmartTVType tvType, boolean useDirections, boolean shouldLinkKeys, boolean doesSuggestDone, KeyboardExtender extender, double scoreFactor, int minCount, int maxRank, boolean shouldUseSuggestions, boolean shouldUseSuboptimal, int maxNumCandidates, String groundTruth) {
        HashSet<String> guessed = new HashSet<String>();

        int minNumSuboptimal = (tvType == SmartTVType.APPLE_TV) ? 1 : 0;

        SuboptimalMoveModel suboptimalModel = new SuboptimalMoveModel(moveSeq, tvType);
        Search searcher = new Search(moveSeq, keyboard, prior, startKey, suboptimalModel, extender, tvType, useDirections, shouldLinkKeys, doesSuggestDone, minCount, maxNumCandidates, shouldUseSuggestions, minNumSuboptimal, false);

        List<SearchResult> results = recoverForConfig(searcher, guessed, groundTruth, 1, maxRank);
        return results;
    }

    private static List<SearchResult> recoverForConfig(Search searcher, HashSet<String> guessed, String groundTruth, int rank, int maxRank) {
        List<SearchResult> results = new ArrayList<SearchResult>();

        while (rank <= maxRank) {
            SearchResult searchResult = searcher.next();

            if (searchResult == null) {
                break;
            }

            if (guessed.contains(searchResult.getGuess())) {
                continue;
            }

            results.add(searchResult);

            if (searchResult.getGuess().equals(groundTruth)) {
                return results;
            }

            rank += 1;
            guessed.add(searchResult.getGuess());
        }

        return results;
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

        //System.out.printf("CCN: %d / %d, CVV: %d / %d, Month: %d / %d, Year: %d / %d, Zip: %d / %d\n", ccnRank, ccnGuesses.size(), cvvRank, cvvGuesses.size(), monthRank, monthGuesses.size(), yearRank, yearGuesses.size(), zipRank, zipCodeGuesses.size());

        CreditCardRank targetRanks = new CreditCardRank(ccnRank, cvvRank, zipRank, monthRank, yearRank);
        CreditCardRank guessLimits = new CreditCardRank(ccnGuesses.size(), cvvGuesses.size(), zipCodeGuesses.size(), monthGuesses.size(), yearGuesses.size());

        int totalRank = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 10000);
        return totalRank;
    }

    private static Move[] reverseMoveSeq(Move[] moveSeq) {
        int numMoves = moveSeq.length;
        SamsungSound finalSound = (SamsungSound) moveSeq[numMoves - 1].getEndSound();

        if (!finalSound.isSelect()) {
            return null;
        }

        Move[] reversed = new Move[moveSeq.length - 1];
        int writeIdx = 0;

        for (int readIdx = numMoves - 1; readIdx >= 1; readIdx--) {
            Move move = moveSeq[readIdx];
            reversed[writeIdx] = new Move(move.getNumMoves(), new SamsungSound("key_select"), move.getStartTime(), move.getEndTime(), move.getMoveTimes(), move.getNumScrolls());
            writeIdx += 1;
        }

        return reversed;
    }
}
