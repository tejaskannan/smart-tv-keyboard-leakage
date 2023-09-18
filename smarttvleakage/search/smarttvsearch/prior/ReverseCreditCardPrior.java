package smarttvsearch.prior;

import smarttvsearch.creditcard.CreditCardUtils;


public class ReverseCreditCardPrior extends NumericPrior {

    private static final String[] CC_PREFIXES = new String[] { "34", "37", "22", "23", "24", "25", "4", "5" };

    public ReverseCreditCardPrior() {
        super();
        this.totalCount = 10;
    }

    @Override
    public int find(String word) {
        int length = word.length();

        if (length == 0) {
            return 0;
        } else if (super.isValid(word)) {
            return 1;
        } else {
            return 0;
        }
    }

    @Override
    public boolean isValid(String word) {
        boolean isNumeric = super.isValid(word);

        boolean hasValidStart = false;
        for (int prefixIdx = 0; prefixIdx < CC_PREFIXES.length; prefixIdx++) {
            if (word.startsWith(CC_PREFIXES[prefixIdx])) {
                hasValidStart = true;
                break;
            }
        }

        return ((word.length() == 15) || (word.length() == 16)) && hasValidStart && isNumeric && CreditCardUtils.verify(word);
    }
}
