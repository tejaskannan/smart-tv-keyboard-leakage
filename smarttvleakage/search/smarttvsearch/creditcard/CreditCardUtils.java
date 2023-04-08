package smarttvsearch.creditcard;

import java.util.HashSet;


public class CreditCardUtils {

    private static int[] CCN_CUTOFFS = { 10, 30, 100, 500 };
    private static int[] CVV_CUTOFFS = { 3, 5, 12, 25 };
    private static int[] ZIP_CUTOFFS = { 3, 5, 12, 25 };
    private static int[] MONTH_CUTOFFS = { 2, 2, 3, 5 };
    private static int[] YEAR_CUTOFFS = { 2, 2, 3, 5 };


    public static boolean verify(String creditCardNumber) {
        /**
         * Verifies the given credit card number using Luhn's algorithm.
         */
        int totalSum = 0;
        int shouldDouble = 0;
        char c;
        int digit;

        for (int idx = (creditCardNumber.length() - 1); idx >= 0; idx--) {
            c = creditCardNumber.charAt(idx);
            digit = ((int) (c - '0'));

            digit += (shouldDouble * digit);

            totalSum += (digit / 10) + (digit % 10);
            shouldDouble ^= 1;
        }

        return (totalSum % 10) == 0;
    }


    public static int computeTotalRank(CreditCardRank targetRanks, CreditCardRank guessLengths, int maxRank) {
        /**
         * Computes the overall rank for credit card guesses in which all fields have a perfect match.
         */
        // If we could not find one of the fields, then the overall match will fail (return -1)
        if (!targetRanks.didFind()) {
            return -1;
        }

        CreditCardRank guessLimits;
        CreditCardRankResult rankResult;
        HashSet<CreditCardRank> guessed = new HashSet<CreditCardRank>();
        int rank = 0;
        
        for (int limitIdx = 0; limitIdx < CCN_CUTOFFS.length; limitIdx++) {
            guessLimits = new CreditCardRank(Math.min(CCN_CUTOFFS[limitIdx], guessLengths.getCcn()),
                                             Math.min(CVV_CUTOFFS[limitIdx], guessLengths.getCvv()),
                                             Math.min(ZIP_CUTOFFS[limitIdx], guessLengths.getZip()),
                                             Math.min(MONTH_CUTOFFS[limitIdx], guessLengths.getMonth()),
                                             Math.min(YEAR_CUTOFFS[limitIdx], guessLengths.getYear()));

            rankResult = computeTotalRankForLimits(targetRanks, guessLimits, rank, guessed, maxRank);
            rank = rankResult.getRank();

            if (rankResult.didFind()) {
                return rank;
            }
        }

        return -1;
    }


    private static CreditCardRankResult computeTotalRankForLimits(CreditCardRank targetRanks, CreditCardRank guessLimits, int currentRank, HashSet<CreditCardRank> guessed, int maxRank) {
        for (int yearIdx = 1; yearIdx <= guessLimits.getYear(); yearIdx++) {
            for (int monthIdx = 1; monthIdx <= guessLimits.getMonth(); monthIdx++) {
                for (int zipIdx = 1; zipIdx <= guessLimits.getZip(); zipIdx++) {
                    for (int cvvIdx = 1; cvvIdx <= guessLimits.getCvv(); cvvIdx++) {
                        for (int ccnIdx = 1; ccnIdx <= guessLimits.getCcn(); ccnIdx++) {
                            if (currentRank >= maxRank) {
                                return new CreditCardRankResult(currentRank, false);
                            }

                            CreditCardRank guess = new CreditCardRank(ccnIdx, cvvIdx, zipIdx, monthIdx, yearIdx);

                            if (guessed.contains(guess)) {
                                continue;
                            }

                            currentRank += 1;
                            guessed.add(guess);

                            if (guess.equals(targetRanks)) {
                                return new CreditCardRankResult(currentRank, true);
                            }
                        }
                    }
                }
            }
        }

        return new CreditCardRankResult(currentRank, false);
    }

}
