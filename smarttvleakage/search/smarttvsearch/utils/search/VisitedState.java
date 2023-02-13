package smarttvsearch.utils.search;

import java.util.List;


public class VisitedState {

    private List<String> keys;
    private String keyboardMode;

    public VisitedState(List<String> keys, String keyboardMode) {
        this.keys = keys;
        this.keyboardMode = keyboardMode;
    }

    public List<String> getKeys() {
        return this.keys;
    }

    public String getKeyboardMode() {
        return this.keyboardMode;
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof VisitedState) {
            VisitedState otherState = (VisitedState) other;

            List<String> otherKeys = otherState.getKeys();
            List<String> currKeys = this.getKeys();

            if ((currKeys.size() != otherKeys.size()) || !(this.getKeyboardMode().equals(otherState.getKeyboardMode()))) {
                return false;
            }

            for (int idx = 0; idx < currKeys.size(); idx++) {
                if (!currKeys.get(idx).equals(otherKeys.get(idx))) {
                    return false;
                }
            }

            return true;
        }

        return false;
    }

    @Override
    public int hashCode() {
        int hash = this.getKeyboardMode().hashCode();

        for (String key : this.getKeys()) {
            hash += key.hashCode();
        }

        return hash;
    }
}
