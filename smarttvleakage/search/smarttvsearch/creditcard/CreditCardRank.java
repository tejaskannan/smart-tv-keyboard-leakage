package smarttvsearch.creditcard;


public class CreditCardRank {

    private int ccnRank;
    private int cvvRank;
    private int monthRank;
    private int yearRank;
    private int zipRank;

    public CreditCardRank(int ccnRank, int cvvRank, int zipRank, int monthRank, int yearRank) {
        this.ccnRank = ccnRank;
        this.cvvRank = cvvRank;
        this.zipRank = zipRank;
        this.monthRank = monthRank;
        this.yearRank = yearRank;
    }

    public int getCcn() {
        return this.ccnRank;
    }

    public int getCvv() {
        return this.cvvRank;
    }

    public int getZip() {
        return this.zipRank;
    }

    public int getMonth() {
        return this.monthRank;
    }

    public int getYear() {
        return this.yearRank;
    }

    public boolean didFind() {
        return (this.getCcn() > 0) && (this.getCvv() > 0) && (this.getZip() > 0) && (this.getMonth() > 0) && (this.getYear() > 0);
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof CreditCardRank) {
            CreditCardRank otherRank = (CreditCardRank) other;
            return (otherRank.getCcn() == this.getCcn()) && (otherRank.getCvv() == this.getCvv()) && (otherRank.getZip() == this.getZip()) && (otherRank.getMonth() == this.getMonth()) && (otherRank.getYear() == this.getYear());
        }

        return false;
    }

    @Override
    public int hashCode() {
        return this.getCcn() + 5 * this.getCvv() + 2 * this.getZip() + 7 * this.getMonth() + 3 * this.getYear();
    }

    @Override
    public String toString() {
        return String.format("CreditCardRank(ccn=%d, cvv=%d, zip=%d, month=%d, year=%d)", this.getCcn(), this.getCvv(), this.getZip(), this.getMonth(), this.getYear());
    }
}
