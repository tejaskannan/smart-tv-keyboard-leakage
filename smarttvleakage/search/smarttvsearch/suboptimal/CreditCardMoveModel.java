package smarttvsearch.suboptimal;

import smarttvsearch.utils.Move;


public class CreditCardMoveModel extends SuboptimalMoveModel {

    private static final double SUBOPTIMAL_MOVE_CUTOFF = 75.0;

    public CreditCardMoveModel(Move[] moveSeq) {
        super(moveSeq);
        this.scoreFactor = 0.75;
    }

    public int getLimit(int moveIdx) {
        double avgMoveTime = super.getAvgMoveTime(moveIdx);

        if (avgMoveTime > SUBOPTIMAL_MOVE_CUTOFF) {
            return SuboptimalMoveModel.MAX_NUM_SUBOPTIMAL_MOVES;
        }

        return 0;
    }
}
