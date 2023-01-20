package smarttvsearch.tests.prior;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import smarttvsearch.prior.NumericPrior;


public class NumericPriorTests {

    @Test
    public void testValidKeys() {
        NumericPrior prior = new NumericPrior();
        assertTrue(prior.isValidKey("0"));
        assertTrue(prior.isValidKey("1"));
        assertTrue(prior.isValidKey("2"));
        assertTrue(prior.isValidKey("3"));
        assertTrue(prior.isValidKey("4"));
        assertTrue(prior.isValidKey("5"));
        assertTrue(prior.isValidKey("6"));
        assertTrue(prior.isValidKey("7"));
        assertTrue(prior.isValidKey("8"));
        assertTrue(prior.isValidKey("9"));
    }

    @Test
    public void testInvalidKeys() {
        NumericPrior prior = new NumericPrior();
        assertTrue(!prior.isValidKey("a"));
        assertTrue(!prior.isValidKey("g"));
        assertTrue(!prior.isValidKey("e"));
        assertTrue(!prior.isValidKey("<DONE>"));
        assertTrue(!prior.isValidKey(" "));
        assertTrue(!prior.isValidKey("<RETURN>"));
        assertTrue(!prior.isValidKey("k"));
        assertTrue(!prior.isValidKey("("));
        assertTrue(!prior.isValidKey("!"));
        assertTrue(!prior.isValidKey("$"));
    }
}
