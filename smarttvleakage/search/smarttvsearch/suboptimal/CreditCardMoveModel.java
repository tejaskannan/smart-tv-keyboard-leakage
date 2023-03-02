package smarttvsearch.suboptimal;

import java.util.List;

import smarttvsearch.utils.Move;
import smarttvsearch.utils.VectorUtils;
import smarttvsearch.utils.SmartTVType;


public class CreditCardMoveModel extends SuboptimalMoveModel {

    private int numSuboptimal;
    private int[] sortedIndices;

    public CreditCardMoveModel(Move[] moveSeq, int numSuboptimal, double scoreFactor, SmartTVType tvType) {
        super(moveSeq, tvType);
        this.scoreFactor = scoreFactor;
        this.numSuboptimal = numSuboptimal;

        // Sort the move sequences in descending order of average delay per move in a single sequence element
        int[] maxDelays = new int[moveSeq.length];
        for (int moveIdx = 0; moveIdx < moveSeq.length; moveIdx++) {
            List<Integer> diffs = VectorUtils.getDiffs(moveSeq[moveIdx].getMoveTimes());
            maxDelays[moveIdx] = VectorUtils.max(diffs);
        }

        this.sortedIndices = VectorUtils.argsortDescending(maxDelays);
    }

    public int getNumSuboptimal() {
        return this.numSuboptimal;
    }

    public int getLimit(int moveIdx) {
        for (int idx = 0; idx < this.getNumSuboptimal(); idx++) {
            if (this.sortedIndices[idx] == moveIdx) {
                return this.getMaxNumSuboptimalMoves();
            }
        }

        return 0;
    }
}
