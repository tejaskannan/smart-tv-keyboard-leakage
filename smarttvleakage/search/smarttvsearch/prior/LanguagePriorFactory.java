package smarttvsearch.prior;

import smarttvsearch.prior.LanguagePrior;
import smarttvsearch.prior.NGramPrior;
import smarttvsearch.prior.NumericPrior;
import smarttvsearch.prior.MonthPrior;
import smarttvsearch.prior.YearPrior;
import smarttvsearch.prior.CreditCardPrior;
import smarttvsearch.prior.ReverseCreditCardPrior;
import smarttvsearch.prior.PrefixPrior;
import smarttvsearch.prior.EnglishPrior;


public class LanguagePriorFactory {

    public static LanguagePrior makePrior(String name, String path) {
        name = name.toLowerCase();

        LanguagePrior result = null;

        if (name.equals("english")) {
            result = new EnglishPrior(path);
        } else if (name.equals("numeric")) {
            result = new NumericPrior();
        } else if (name.equals("ngram")) {
            result = new NGramPrior(path);
        } else if (name.equals("prefix")) {
            result = new PrefixPrior(path);  
        } else if (name.equals("month")) {
            result = new MonthPrior();  
        } else if (name.equals("year")) {
            result = new YearPrior();  
        } else if (name.equals("credit_card")) {
            result = new CreditCardPrior();  
        } else if (name.equals("reverse_credit_card")) {
            result = new ReverseCreditCardPrior();  
        } else {
            throw new IllegalArgumentException("No prior with name: " + name);
        }

        return result;
    }

}
