package smarttvsearch.tests.creditcard;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import smarttvsearch.creditcard.CreditCardUtils;


public class CreditCardTests {

    @Test
    public void testValid0() {
        boolean result = CreditCardUtils.verify("79927398713");
        assertEquals(result, true);
    }

    @Test
    public void testValid1() {
        boolean result = CreditCardUtils.verify("3716820019271998");
        assertEquals(result, true);
    }

    @Test
    public void testValid2() {
        boolean result = CreditCardUtils.verify("6823119834248189");
        assertEquals(result, true);
    }

    @Test
    public void testInvalid0() {
        boolean result = CreditCardUtils.verify("5190990281925290");
        assertEquals(result, false);
    }

    @Test
    public void testInvalid1() {
        boolean result = CreditCardUtils.verify("37168200192719989");
        assertEquals(result, false);
    }

    @Test
    public void testInvalid2() {
        boolean result = CreditCardUtils.verify("8102966371298364");
        assertEquals(result, false);
    }

    @Test
    public void testAmex() {
        boolean result = CreditCardUtils.verify("379097448806314");
        assertEquals(result, true);
    }

    @Test
    public void testVisa() {
        boolean result = CreditCardUtils.verify("4716278591454565");
        assertEquals(result, true);
    }
}

