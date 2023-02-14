package smarttvsearch.keyboard;


public class KeyboardPosition {

    private String key;
    private String keyboardName;

    public KeyboardPosition(String key, String keyboardName) {
        this.key = key;
        this.keyboardName = keyboardName;
    }

    public String getKey() {
        return this.key;
    }

    public String getKeyboardName() {
        return this.keyboardName;
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof KeyboardPosition) {
            KeyboardPosition otherPos = (KeyboardPosition) other;
            return (otherPos.getKey().equals(this.getKey())) && (otherPos.getKeyboardName().equals(this.getKeyboardName()));
        }

        return false;
    }

    @Override
    public String toString() {
        return String.format("KeyboardPosition(key=%s, keyboard=%s)", this.getKey(), this.getKeyboardName());
    }
}
