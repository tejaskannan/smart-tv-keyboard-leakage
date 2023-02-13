package smarttvsearch.creditcard;


public class CreditCardDetails {

    private String ccn;
    private String cvv;
    private String expMonth;
    private String expYear;
    private String zipCode;

    public CreditCardDetails(String ccn, String cvv, String expMonth, String expYear, String zipCode) {
        this.ccn = ccn;
        this.cvv = cvv;
        this.expMonth = expMonth;
        this.expYear = expYear;
        this.zipCode = zipCode;
    }

    public String getCCN() {
        return this.ccn;
    }

    public String getCVV() {
        return this.cvv;
    }

    public String getExpMonth() {
        return this.expMonth;
    }

    public String getExpYear() {
        return this.expYear;
    }

    public String getZipCode() {
        return this.zipCode;
    }

    @Override
    public int hashCode() {
        return this.getCCN().hashCode() + this.getCVV().hashCode() + this.getExpMonth().hashCode() + this.getExpYear().hashCode() + this.getZipCode().hashCode();
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof CreditCardDetails) {
            CreditCardDetails otherDetails = (CreditCardDetails) other;

            boolean isCCNEqual = this.getCCN().equals(otherDetails.getCCN());
            boolean isCVVEqual = this.getCVV().equals(otherDetails.getCVV());
            boolean isExpMonthEqual = this.getExpMonth().equals(otherDetails.getExpMonth());
            boolean isExpYearEqual = this.getExpYear().equals(otherDetails.getExpYear());
            boolean isZipCodeEqual = this.getZipCode().equals(otherDetails.getZipCode());

            return isCCNEqual && isCVVEqual && isExpMonthEqual && isExpYearEqual && isZipCodeEqual;
        }

        return false;
    }
}
