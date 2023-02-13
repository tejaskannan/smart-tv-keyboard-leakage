package smarttvsearch.prior;


public class YearPrior extends NumericPrior {

    public YearPrior() {
        super();
        this.totalCount = 15;
    }

    @Override
    public boolean isValid(String word) {
        if ((word.length() == 0) || (word.length() > 2)) {
            return false;
        }

        int firstChar = word.charAt(0);

        if ((firstChar != '2') && (firstChar != '3')) {
            return false;
        } else if (word.length() == 1) {
            return true;
        }

        int secondChar = word.charAt(1);

        if (firstChar == '2') {
            return (secondChar >= '2') && (secondChar <= '9');
        } else {
            return (secondChar >= '0') && (secondChar <= '5');
        }
    }
}
