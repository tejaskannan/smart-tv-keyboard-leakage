package smarttvsearch.suboptimal;

import java.util.List;

import smarttvsearch.utils.Move;
import smarttvsearch.utils.VectorUtils;


public class CreditCardMoveModel extends SuboptimalMoveModel {

    private static final double SUBOPTIMAL_MOVE_CUTOFF = 75.0;
    private static double avgDiff;
    private static double stdDiff;
    private static double mistakeFactor;

    public CreditCardMoveModel(Move[] moveSeq, double avgDiff, double stdDiff, double mistakeFactor) {
        super(moveSeq);
        this.scoreFactor = 0.75;
        this.avgDiff = avgDiff;
        this.stdDiff = stdDiff;
        this.mistakeFactor = mistakeFactor;
    }

    public double getCutoff() {
        return this.avgDiff + this.mistakeFactor * this.stdDiff;
    }

    public int getLimit(int moveIdx) {
        Move move = this.getMove(moveIdx);
        List<Integer> moveTimeDiffs = VectorUtils.getDiffs(move.getMoveTimes());

        if (moveTimeDiffs == null) {
            return 0;
        }

        for (int idx = 0; idx < moveTimeDiffs.size(); idx++) {
            if (((double) moveTimeDiffs.get(idx)) >= this.getCutoff()) {
                return SuboptimalMoveModel.MAX_NUM_SUBOPTIMAL_MOVES;
            }
        }

        return 0;
    }
}
