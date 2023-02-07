package smarttvsearch.tests.prior;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import smarttvsearch.prior.NGramPrior;


public class NGramPriorTests {

    private NGramPrior prior;

    public NGramPriorTests() {
        this.prior = new NGramPrior("/local/dictionaries/passwords/rockyou.db");
        this.prior.build(false);
    }

    @Test
    public void testSimple() {
        assertEquals(this.prior.find("passw"), 2798);
        assertEquals(this.prior.find("passt"), 76);
        System.out.println(this.prior.getTotalCount());
    }

    @Test
    public void testNGram() {
        assertEquals(this.prior.toNGram("password"), "sswor");
        assertEquals(this.prior.toNGram("1234"), "<S><S>123");
        assertEquals(this.prior.toNGram("f"), "<S><S><S><S><S>");
        assertEquals(this.prior.toNGram(""), "<S><S><S><S><S>");
    }
}
