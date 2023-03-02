package smarttvsearch.suboptimal;

import smarttvsearch.utils.Move;
import smarttvsearch.utils.SmartTVType;


public class SuboptimalMoveModel {

    private Move[] moveSeq;
    protected double scoreFactor;
    private int maxNumSuboptimalMoves;

    public SuboptimalMoveModel(Move[] moveSeq, SmartTVType tvType) {
        this.moveSeq = moveSeq;
        
        this.maxNumSuboptimalMoves = 0;
        if (tvType == SmartTVType.SAMSUNG) {
            this.maxNumSuboptimalMoves = 4;
            this.scoreFactor = 0.1;
        } else if (tvType == SmartTVType.APPLE_TV) {
            this.maxNumSuboptimalMoves = 6;  // Users make more suboptimal moves on the Apple TV
            this.scoreFactor = 0.5;
        } else {
            throw new IllegalArgumentException(String.format("Unknown suboptimal limit for tv: %s", tvType));
        }
    }

    public Move getMove(int moveIdx) {
        return this.moveSeq[moveIdx];
    }
    
    public int getLimit(int moveIdx) {
        return this.maxNumSuboptimalMoves;
    }

    public int getMaxNumSuboptimalMoves() {
        return this.maxNumSuboptimalMoves;
    }

    public double getScoreFactor(int numSuboptimalMoves) {
        return Math.pow(this.scoreFactor, (double) Math.abs(numSuboptimalMoves));
    }
}
