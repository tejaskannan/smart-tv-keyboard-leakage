package smarttvsearch.prior;

import smarttvsearch.creditcard.CreditCardUtils;


public class CreditCardPrior extends NumericPrior {

    private static final String[] CC_PREFIXES = new String[] { "34", "37", "22", "23", "24", "25", "4", "5" };

    public CreditCardPrior() {
        super();
        this.totalCount = 10;
    }

    @Override
    public int find(String word) {
        int length = word.length();

        if (length == 0) {
            return 0;
        } else if (length == 1) {
            char firstChar = word.charAt(0);
            return ((firstChar >= '2') && (firstChar <= '5')) ? 1 : 0;
        } else {
            for (String prefix : CC_PREFIXES) {
                if (word.startsWith(prefix)) {
                    return 1;
                }
            }

            return 0;
        }
    }

    @Override
    public boolean isValid(String word) {
        boolean isNumeric = super.isValid(word);
        return ((word.length() == 15) || (word.length() == 16)) && isNumeric && CreditCardUtils.verify(word);
    }
}
