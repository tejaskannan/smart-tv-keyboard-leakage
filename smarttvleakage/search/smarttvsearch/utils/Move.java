package smarttvsearch.utils;

import smarttvsearch.utils.sounds.SmartTVSound;


public class Move {

    private int numMoves;
    private Direction[] directions;
    private SmartTVSound endSound;
    private int startTime;
    private int endTime;

    public Move(int numMoves, Direction[] directions, SmartTVSound endSound, int startTime, int endTime) {
        if (numMoves < 0) {
            throw new IllegalArgumentException("Most provide a non-negative number of moves.");
        }

        if (directions.length != numMoves) {
            throw new IllegalArgumentException(String.format("The number of moves (%d) must match the number of directions (%d).", numMoves, directions.length));
        }

        this.numMoves = numMoves;
        this.directions = directions;
        this.endSound = endSound;
        this.startTime = startTime;
        this.endTime = endTime;
    }

    public Move(int numMoves, SmartTVSound endSound, int startTime, int endTime) {
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
        this.startTime = startTime;
        this.endTime = endTime;
    }

    public int getNumMoves() {
        return this.numMoves;
    }

    public int getStartTime() {
        return this.startTime;
    }

    public int getEndTime() {
        return this.endTime;
    }

    public double getAvgTimePerMove() {
        if (this.getNumMoves() == 0) {
            return 0.0;
        }

        return ((double) (this.getEndTime() - this.getStartTime())) / ((double) this.getNumMoves());
    }

    public Direction[] getDirections() {
        return this.directions;
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
