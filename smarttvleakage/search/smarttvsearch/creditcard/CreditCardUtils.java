package smarttvsearch.creditcard;


public class CreditCardUtils {

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

            //System.out.printf("%d ", digit);

            totalSum += (digit / 10) + (digit % 10);
            shouldDouble ^= 1;
        }

        //System.out.println();

        return (totalSum % 10) == 0;
    }

}
