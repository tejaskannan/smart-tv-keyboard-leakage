package smarttvsearch.tests.prior;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import smarttvsearch.prior.NGramPrior;


public class NGramPriorTests {

    private NGramPrior prior;

    public NGramPriorTests() {
        this.prior = new NGramPrior("/local/dictionaries/passwords/rockyou-variable.db");
        this.prior.build(false);
    }

    @Test
    public void testSimple() {
        System.out.println(this.prior.find("erik4125"));
        assertEquals(this.prior.find("passw"), 2799);
        assertEquals(this.prior.find("passt"), 77);
    }

    @Test
    public void testNGram() {
        assertEquals(this.prior.toNGram("password", 5), "sswor");
        assertEquals(this.prior.toNGram("1234", 5), "<S><S>123");
        assertEquals(this.prior.toNGram("f", 5), "<S><S><S><S><S>");
        assertEquals(this.prior.toNGram("", 5), "<S><S><S><S><S>");
    }
}
