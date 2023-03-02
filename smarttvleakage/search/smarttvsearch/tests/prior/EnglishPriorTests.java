package smarttvsearch.tests.prior;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import smarttvsearch.prior.EnglishPrior;


public class EnglishPriorTests {

    private EnglishPrior prior;

    public EnglishPriorTests() {
        this.prior = new EnglishPrior("/local/dictionaries/english/wikipedia.db");
        this.prior.build(false);
    }

    @Test
    public void testSimple() {
        assertEquals(this.prior.find("adjusted"), 19009);
        assertEquals(this.prior.find("sharks"), 23989);
    }

    @Test
    public void testIsValid() {
        assertEquals(this.prior.isValid("the"), true);
        assertEquals(this.prior.isValid("asdfjkluoriwe"), false);
    }

    @Test
    public void testNextChars() {
        char[] expected = { 'e', 'a' };
        char[] observed = this.prior.nextMostCommon("th", 2);
        arrayEquals(expected, observed);

        expected = new char[] { 'a', 'r', 'o' };
        observed = this.prior.nextMostCommon("footb", 3);
        arrayEquals(expected, observed);
    }

    private void arrayEquals(char[] expected, char[] observed) {
        assertEquals(expected.length, observed.length);

        for (int idx = 0; idx < expected.length; idx++) {
            assertEquals(expected[idx], observed[idx]);
        }
    }

}
