package smarttvsearch.utils;

import java.util.List;
import org.json.JSONObject;
import org.json.JSONArray;
import org.json.JSONException;

import smarttvsearch.utils.Move;
import smarttvsearch.utils.SmartTVType;
import smarttvsearch.utils.sounds.SmartTVSound;
import smarttvsearch.utils.sounds.SamsungSound;
import smarttvsearch.utils.sounds.AppleTVSound;


public class JsonUtils {

    public static <T> JSONArray listToJsonArray(List<T> list) {
        JSONArray result = new JSONArray();
        for (T value : list) {
            result.put(value);
        }

        return result;
    }

    //public static JSONArray doubleListToJsonArray(List<Double> list) {
    //    JSONArray result = new JSONArray();
    //    for (Double value : list) {
    //        result.put(value);
    //    }

    //    return result;
    //}

    public static Move[] parseMoveSeq(JSONArray jsonMoveSeq, SmartTVType tvType) {
        Move[] moveSeq = new Move[jsonMoveSeq.length()];

        for (int idx = 0; idx < jsonMoveSeq.length(); idx++) {
            moveSeq[idx] = parseMove(jsonMoveSeq.getJSONObject(idx), tvType);
        }

        return moveSeq;
    }

    public static Move parseMove(JSONObject jsonMove, SmartTVType tvType) {
        int numMoves = jsonMove.getInt("num_moves");
        int startTime = jsonMove.getInt("start_time");
        int endTime = jsonMove.getInt("end_time");
        int numScrolls = jsonMove.getInt("num_scrolls");

        SmartTVSound endSound;
        if (tvType == SmartTVType.SAMSUNG) {
            endSound = new SamsungSound(jsonMove.getString("end_sound"));
        } else if (tvType == SmartTVType.APPLE_TV) {
            endSound = new AppleTVSound(jsonMove.getString("end_sound"));  
        } else {
            throw new IllegalArgumentException("Cannot parse sound for tv: " + tvType.name());
        }

        int[] moveTimes = new int[numMoves];
        JSONArray jsonMoveTimes = jsonMove.getJSONArray("move_times");

        for (int timeIdx = 0; timeIdx < numMoves; timeIdx++) {
            moveTimes[timeIdx] = jsonMoveTimes.getInt(timeIdx);
        }

        try {
            JSONArray directionsArray = jsonMove.getJSONArray("directions");
            Direction[] directions = new Direction[directionsArray.length()];

            for (int idx = 0; idx < directionsArray.length(); idx++) {
                directions[idx] = Direction.valueOf(directionsArray.getString(idx).toUpperCase());
            }

            return new Move(numMoves, directions, endSound, startTime, endTime, moveTimes, numScrolls);
        } catch (JSONException ex) {
            return new Move(numMoves, endSound, startTime, endTime, moveTimes, numScrolls);
        }
    }
}
