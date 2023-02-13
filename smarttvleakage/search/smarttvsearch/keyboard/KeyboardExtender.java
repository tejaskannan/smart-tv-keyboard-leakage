package smarttvsearch.keyboard;

import java.util.Set;
import java.util.HashSet;


public class KeyboardExtender {

    protected MultiKeyboard keyboard;

    public KeyboardExtender(MultiKeyboard keyboard) {
        this.keyboard = keyboard;
    }

    public Set<String> getExtendedNeighbors(String key, int numMoves, String keyboardName) {
        return new HashSet<String>();
    }
}

