package smarttvsearch.prior;

public class CreditCardPrior extends LanguagePrior {

    @Override
    public String[] processWord(String word) {
        return new String[] { word };
    }

    @Override
    public void build(boolean hasCounts) {
    }

}
