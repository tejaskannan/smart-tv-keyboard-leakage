package smarttvsearch.keyboard;

import java.util.HashSet;
import java.util.Set;
import smarttvsearch.utils.Direction;


public class NumericKeyboardExtender extends KeyboardExtender {

    public NumericKeyboardExtender(MultiKeyboard keyboard) {
        super(keyboard);
    }

    @Override
    public Set<String> getExtendedNeighbors(String key, int numMoves, String keyboardName) {
        if ((key.equals("0") || key.equals("1") || key.equals("2") || key.equals("9")) && (numMoves >= 5)) {
            // Get the neighbors through a forced wraparound
            Direction[] directions = new Direction[numMoves];
            Direction direction = (key.equals("1") || key.equals("2")) ? Direction.LEFT : Direction.RIGHT;

            for (int dirIdx = 0; dirIdx < numMoves; dirIdx++) {
                directions[dirIdx] = direction;
            }

            return this.keyboard.getKeysForDistanceCumulative(key, numMoves, true, true, directions, keyboardName);
        } else {
            return new HashSet<String>();
        }
    }
}
