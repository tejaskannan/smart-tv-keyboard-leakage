package smarttvsearch.utils;

import smarttvsearch.utils.sounds.SmartTVSound;


public class Move {

    private int numMoves;
    private Direction[] directions;
    private SmartTVSound endSound;

    public Move(int numMoves, Direction[] directions, SmartTVSound endSound) {
        if (numMoves < 0) {
            throw new IllegalArgumentException("Most provide a non-negative number of moves.");
        }

        if (directions.length != numMoves) {
            throw new IllegalArgumentException(String.format("The number of moves (%d) must match the number of directions (%d).", numMoves, directions.length));
        }

        this.numMoves = numMoves;
        this.directions = directions;
        this.endSound = endSound;
    }

    public Move(int numMoves, SmartTVSound endSound) {
        if (numMoves < 0) {
            throw new IllegalArgumentException("Most provide a non-negative number of moves.");
        }

        Direction[] directions = new Direction[numMoves];
        for (int idx = 0; idx < numMoves; idx++) {
            directions[idx] = Direction.ANY;
        }

        this.numMoves = numMoves;
        this.directions = directions;
        this.endSound = endSound;
    }

    public int getNumMoves() {
        return this.numMoves;
    }

    public Direction getDirection(int moveIdx) {
        if ((moveIdx < 0) || (moveIdx >= this.getNumMoves())) {
            throw new IllegalArgumentException(String.format("The move index must be in the range [0, %d)", this.getNumMoves()));
        }

        return directions[moveIdx];
    }

    public SmartTVSound getEndSound() {
        return this.endSound;
    }
}
