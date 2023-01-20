package smarttvsearch.prior;


public class MonthPrior extends NumericPrior {

    public MonthPrior() {
        super();
        this.totalCount = 12;
    }

    @Override
    public boolean isValid(String word) {
        if ((word.length() == 0) || (word.length() > 2)) {
            return false;
        }

        int firstChar = word.charAt(0);

        if ((firstChar != '0') && (firstChar != '1')) {
            return false;
        } else if (word.length() == 1) {
            return true;
        }

        int secondChar = word.charAt(1);

        if (firstChar == '0') {
            return (secondChar >= '0') && (secondChar <= '9');
        } else {
            return (secondChar >= '0') && (secondChar <= '2');
        }
    }
}
