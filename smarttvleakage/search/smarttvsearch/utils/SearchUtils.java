package smarttvsearch.utils;

import java.util.Stack;
import smarttvsearch.utils.sounds.SmartTVSound;


public class SearchUtils {

    public static boolean[] markDeletedMoves(Move[] moveSeq) {
        SmartTVSound[] moveSounds = new SmartTVSound[moveSeq.length];

        for (int idx = 0; idx < moveSeq.length; idx++) {
            moveSounds[idx] = moveSeq[idx].getEndSound();
        }

        return markDeletedMovesBySound(moveSounds);
    }

    public static boolean[] markDeletedMovesBySound(SmartTVSound[] moveSounds) {
        boolean[] result = new boolean[moveSounds.length];

        if (moveSounds.length <= 1) {
            return result;
        }

        Stack<Integer> moveIdxStack = new Stack<Integer>();

        for (int idx = 0; idx < moveSounds.length; idx++) {
            SmartTVSound endSound = moveSounds[idx];
            boolean isDelete = endSound.isDelete();

            if (isDelete) {
                if (!moveIdxStack.empty()) {
                    int deletedIdx = moveIdxStack.pop();
                    result[deletedIdx] = true;
                }
            } else {
                moveIdxStack.push(idx);
            }
        }

        return result;
    }

}
