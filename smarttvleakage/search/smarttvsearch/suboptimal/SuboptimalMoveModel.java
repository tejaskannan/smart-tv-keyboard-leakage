package smarttvsearch.suboptimal;

import smarttvsearch.utils.Move;


public class SuboptimalMoveModel {

    private Move[] moveSeq;
    protected double scoreFactor;
    public static final int MAX_NUM_SUBOPTIMAL_MOVES = 4;

    public SuboptimalMoveModel(Move[] moveSeq) {
        this.moveSeq = moveSeq;
        this.scoreFactor = 0.1;
    }
    
    public Move getMove(int moveIdx) {
        return this.moveSeq[moveIdx];
    }
    
    public int getLimit(int moveIdx) {
        return MAX_NUM_SUBOPTIMAL_MOVES;
    }

    public double getScoreFactor(int numSuboptimalMoves) {
        return Math.pow(this.scoreFactor, (double) Math.abs(numSuboptimalMoves));
    }
}
