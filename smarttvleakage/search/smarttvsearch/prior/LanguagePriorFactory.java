package smarttvsearch.prior;

import smarttvsearch.prior.LanguagePrior;
import smarttvsearch.prior.NGramPrior;
import smarttvsearch.prior.NumericPrior;


public class LanguagePriorFactory {

    public static LanguagePrior makePrior(String name, String path) {
        name = name.toLowerCase();

        LanguagePrior result = null;

        if (name.equals("numeric")) {
            result = new NumericPrior();
        } else if (name.equals("ngram")) {
            result = new NGramPrior(path);
        } else {
            throw new IllegalArgumentException("No prior with name: " + name);
        }

        return result;
    }

}
