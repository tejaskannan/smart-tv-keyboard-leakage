package smarttvsearch.tests.creditcard;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import smarttvsearch.creditcard.CreditCardUtils;
import smarttvsearch.creditcard.CreditCardRank;


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

    @Test
    public void testRank1() {
        CreditCardRank guessLimits = new CreditCardRank(500, 50, 50, 5, 5);
        CreditCardRank targetRanks = new CreditCardRank(1, 1, 1, 1, 1);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 10000);
        assertEquals(1, result);
    }

    @Test
    public void testRank4() {
        CreditCardRank guessLimits = new CreditCardRank(500, 50, 50, 5, 5);
        CreditCardRank targetRanks = new CreditCardRank(4, 1, 1, 1, 1);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 10000);
        assertEquals(4, result);
    }

    @Test
    public void testRank11() {
        CreditCardRank guessLimits = new CreditCardRank(500, 50, 50, 5, 5);
        CreditCardRank targetRanks = new CreditCardRank(1, 2, 1, 1, 1);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 10000);
        assertEquals(11, result);
    }

    @Test
    public void testRank43() {
        CreditCardRank guessLimits = new CreditCardRank(500, 50, 50, 5, 5);
        CreditCardRank targetRanks = new CreditCardRank(3, 2, 2, 1, 1);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 10000);
        assertEquals(43, result);
    }

    @Test
    public void testRank251() {
        CreditCardRank guessLimits = new CreditCardRank(500, 50, 50, 1, 5);
        CreditCardRank targetRanks = new CreditCardRank(11, 4, 1, 1, 1);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 10000);
        assertEquals(251, result);
    }

    @Test
    public void testRank1527() {
        CreditCardRank guessLimits = new CreditCardRank(500, 50, 50, 1, 3);
        CreditCardRank targetRanks = new CreditCardRank(57, 1, 1, 1, 1);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 10000);
        assertEquals(1527, result);
    }

    @Test
    public void testRank3000() {
        CreditCardRank guessLimits = new CreditCardRank(500, 50, 50, 5, 5);
        CreditCardRank targetRanks = new CreditCardRank(30, 5, 5, 2, 2);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 10000);
        assertEquals(3000, result);
    }

    @Test
    public void testRank59266() {
        CreditCardRank guessLimits = new CreditCardRank(500, 36, 50, 2, 2);
        CreditCardRank targetRanks = new CreditCardRank(166, 5, 1, 1, 1);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 100000);
        assertEquals(59266, result);
    }

    @Test
    public void testRank37957() {
        CreditCardRank guessLimits = new CreditCardRank(500, 50, 50, 2, 3);
        CreditCardRank targetRanks = new CreditCardRank(7, 1, 8, 1, 2);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 100000);
        assertEquals(37957, result);
    }

    @Test
    public void testRankTooBig() {
        CreditCardRank guessLimits = new CreditCardRank(500, 50, 50, 2, 3);
        CreditCardRank targetRanks = new CreditCardRank(7, 1, 8, 1, 2);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 10000);
        assertEquals(-1, result);
    }

    @Test
    public void testRankNotFound() {
        CreditCardRank guessLimits = new CreditCardRank(500, 50, 50, 2, 3);
        CreditCardRank targetRanks = new CreditCardRank(-1, 1, 1, 1, 1);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits, 10000);
        assertEquals(-1, result);
    }

    @Test
    public void testRank129600() {
        CreditCardRank guessLimits = new CreditCardRank(500, 50, 50, 5, 5);
        CreditCardRank targetRanks = new CreditCardRank(100, 12, 12, 3, 3);
        int result = CreditCardUtils.computeTotalRank(targetRanks, guessLimits,  150000);
        assertEquals(129600, result);
    }
}
