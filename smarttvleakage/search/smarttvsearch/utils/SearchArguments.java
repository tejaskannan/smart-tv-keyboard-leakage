package smarttvsearch.utils;


public class SearchArguments {

    private String inputFile;
    private String labelsFile;
    private String outputFile;
    private String passwordPrior;
    private String englishPrior;
    private String zipPrior;
    private boolean shouldUseDirections;
    private boolean shouldIncludeSuboptimal;
    private boolean shouldForceSuggestions;
    private boolean shouldUseExhaustive;

    private static final String INPUT_NAME = "--input-file";
    private static final String OUTPUT_NAME = "--output-file";
    private static final String PASSWORD_NAME = "--password-prior";
    private static final String ENGLISH_NAME = "--english-prior";
    private static final String ZIP_NAME = "--zip-prior";
    private static final String DIRECTIONS_NAME = "--ignore-directions";
    private static final String SUBOPTIMAL_NAME = "--ignore-suboptimal";
    private static final String SUGGESTIONS_NAME = "--force-suggestions";
    private static final String EXHAUSTIVE_NAME = "--use-exhaustive";

    public SearchArguments(String[] args) {
        if (args.length < 10) {
            throw new IllegalArgumentException("Must provide the (1) --input-file, (2) --output-file, (3) --password-prior, (4) --english-prior, and (5) --zip-prior");
        }

        this.inputFile = null;
        this.outputFile = null;
        this.passwordPrior = null;
        this.englishPrior = null;
        this.zipPrior = null;
        this.shouldUseDirections = true;
        this.shouldIncludeSuboptimal = true;
        this.shouldForceSuggestions = false;
        this.shouldUseExhaustive = false;

        for (int idx = 0; idx < args.length; idx += 2) {
            if (args[idx].equals(INPUT_NAME)) {
                if (this.inputFile != null) {
                    throw new IllegalArgumentException("Duplicate input files.");
                }
                this.inputFile = args[idx + 1];
            } else if (args[idx].equals(OUTPUT_NAME)) {
                if (this.outputFile != null) {
                    throw new IllegalArgumentException("Duplicate output files.");
                }
                this.outputFile = args[idx + 1];
            } else if (args[idx].equals(PASSWORD_NAME)) {
                if (this.passwordPrior != null) {
                    throw new IllegalArgumentException("Duplicate password priors.");
                }
                this.passwordPrior = args[idx + 1];
            } else if (args[idx].equals(ENGLISH_NAME)) {
                if (this.englishPrior != null) {
                    throw new IllegalArgumentException("Duplicate english priors.");
                }
                this.englishPrior = args[idx + 1];
            } else if (args[idx].equals(ZIP_NAME)) {
                if (this.zipPrior != null) {
                    throw new IllegalArgumentException("Duplicate zip priors.");
                }
                this.zipPrior = args[idx + 1];
            } else if (args[idx].equals(DIRECTIONS_NAME)) {
                this.shouldUseDirections = false;
            } else if (args[idx].equals(SUBOPTIMAL_NAME)) {
                this.shouldIncludeSuboptimal = false;
            } else if (args[idx].equals(SUGGESTIONS_NAME)) {
                this.shouldForceSuggestions = true;
            } else if (args[idx].equals(EXHAUSTIVE_NAME)) {
                this.shouldUseExhaustive = true;
            } else {
                throw new IllegalArgumentException(String.format("Unknown argument: %s", args[idx]));
            }
        }

        // Validate that we have the required arguments
        boolean areAllPresent = (this.inputFile != null) && (this.outputFile != null) && (this.passwordPrior != null) && (this.englishPrior != null) && (this.zipPrior != null);
        
        if (!areAllPresent) {
            StringBuilder errorMsg = new StringBuilder();
            errorMsg.append("Missing: ");

            if (this.inputFile == null) {
                errorMsg.append(INPUT_NAME);
                errorMsg.append(" ");
            }

            if (this.outputFile == null) {
                errorMsg.append(OUTPUT_NAME);
                errorMsg.append(" ");
            }

            if (this.passwordPrior == null) {
                errorMsg.append(PASSWORD_NAME);
                errorMsg.append(" ");
            }

            if (this.englishPrior == null) {
                errorMsg.append(ENGLISH_NAME);
                errorMsg.append(" ");
            }

            if (this.zipPrior == null) {
                errorMsg.append(ZIP_NAME);
                errorMsg.append(" ");
            }

            throw new IllegalArgumentException(errorMsg.toString());
        }

        this.labelsFile = this.inputFile.substring(0, this.inputFile.length() - 5) + "_labels.json";
    }

    public String getInputFile() {
        return this.inputFile;
    }

    public String getLabelsFile() {
        return this.labelsFile;
    }

    public String getOutputFile() {
        return this.outputFile;
    }

    public String getPasswordPrior() {
        return this.passwordPrior;
    }

    public String getEnglishPrior() {
        return this.englishPrior;
    }

    public String getZipPrior() {
        return this.zipPrior;
    }

    public boolean shouldUseDirections() {
        return this.shouldUseDirections;
    }

    public boolean shouldUseSuboptimal() {
        return this.shouldIncludeSuboptimal;
    }

    public boolean shouldForceSuggestions() {
        return this.shouldForceSuggestions;
    }

    public boolean shouldUseExhaustive() {
        return this.shouldUseExhaustive;
    }
}
