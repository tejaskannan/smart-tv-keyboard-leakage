package smarttvsearch.tests.prior;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import smarttvsearch.prior.PrefixPrior;


public class ZipPriorTests {

    private PrefixPrior prior;

    public ZipPriorTests() {
        this.prior = new PrefixPrior("/local/dictionaries/credit_cards/zip_codes.txt");
        this.prior.build(true);
    }

    @Test
    public void testFullZip() {
        assertEquals(this.prior.find("60804"), 82383);
        assertEquals(this.prior.find("60639"), 88204);
        assertEquals(this.prior.find("94306"), 27435);
        assertEquals(this.prior.find("60615"), 40590);
        assertEquals(this.prior.find("10451"), 48136);
    }

    @Test
    public void testZipPrefix() {
        assertEquals(this.prior.find("6080"), 124267);
        assertEquals(this.prior.find("6063"), 487787);
        assertEquals(this.prior.find("9430"), 112510);
    }
}
