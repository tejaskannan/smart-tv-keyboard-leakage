package smarttvsearch;

import org.json.JSONObject;
import org.json.JSONArray;
import org.json.JSONException;

import smarttvsearch.prior.LanguagePrior;
import smarttvsearch.prior.LanguagePriorFactory;
import smarttvsearch.keyboard.MultiKeyboard;
import smarttvsearch.utils.Direction;
import smarttvsearch.utils.FileUtils;
import smarttvsearch.utils.Move;
import smarttvsearch.utils.SmartTVType;
import smarttvsearch.utils.sounds.SmartTVSound;
import smarttvsearch.utils.sounds.SamsungSound;

public class SearchRunner {

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Must provide a path to (1) the move JSON file and (2) the folder containing the language prior.");
            return;
        }

        // Unpack the arguments
        String movePath = args[0];
        String priorFolder = args[1];

        JSONObject serializedMoves = FileUtils.readJsonObject(movePath);
        String seqType = serializedMoves.getString("seq_type");

        // Parse the TV Type
        String tvTypeName = serializedMoves.getString("tv_type").toUpperCase();
        SmartTVType tvType = SmartTVType.valueOf(tvTypeName);

        // Make the keyboard
        String keyboardFolder = FileUtils.joinPath("keyboard", tvTypeName.toLowerCase());
        MultiKeyboard keyboard = new MultiKeyboard(tvType, keyboardFolder);

        // Unpack the labels for reference. We do this for efficiency only, as we can stop the search when
        // we actually find the correct target string
        String labelsPath = movePath.substring(0, movePath.length() - 5) + "_labels.json";
        JSONObject serializedLabels = FileUtils.readJsonObject(labelsPath);

        // Determine the recovery process based on the sequence type
        if (seqType.equals("credit_card")) {
            JSONArray jsonMoveSequences = serializedMoves.getJSONArray("move_sequences");
            JSONArray creditCardLabels = serializedLabels.getJSONArray("labels");

            for (int idx = 0; idx < jsonMoveSequences.length(); idx++) {
                // Unpack the credit card record and parse each field as a proper move sequence
                JSONObject creditCardRecord = jsonMoveSequences.getJSONObject(idx);
 
                Move[] ccnSeq = parseMoveSeq(creditCardRecord.getJSONArray("credit_card"), tvType);
                Move[] zipSeq = parseMoveSeq(creditCardRecord.getJSONArray("zip_code"), tvType);
                Move[] monthSeq = parseMoveSeq(creditCardRecord.getJSONArray("exp_month"), tvType);
                Move[] yearSeq = parseMoveSeq(creditCardRecord.getJSONArray("exp_year"), tvType);
                Move[] cvvSeq = parseMoveSeq(creditCardRecord.getJSONArray("security_code"), tvType);

                // Get the labels for this index
                JSONObject labelsJson = creditCardLabels.getJSONObject(idx);
            }
        } else {
            throw new IllegalArgumentException("Invalid sequence type: " + seqType);
        }
    }

    private static Move[] parseMoveSeq(JSONArray jsonMoveSeq, SmartTVType tvType) {
        Move[] moveSeq = new Move[jsonMoveSeq.length()];

        for (int idx = 0; idx < jsonMoveSeq.length(); idx++) {
            moveSeq[idx] = parseMove(jsonMoveSeq.getJSONObject(idx), tvType);
        }

        return moveSeq;
    }

    private static Move parseMove(JSONObject jsonMove, SmartTVType tvType) {
        int numMoves = jsonMove.getInt("num_moves");

        SmartTVSound endSound;
        if (tvType == SmartTVType.SAMSUNG) {
            endSound = new SamsungSound(jsonMove.getString("end_sound"));
        } else {
            throw new IllegalArgumentException("Cannot parse sound for tv: " + tvType.name());
        }

        try {
            JSONArray directionsArray = jsonMove.getJSONArray("directions");
            Direction[] directions = new Direction[directionsArray.length()];

            for (int idx = 0; idx < directionsArray.length(); idx++) {
                directions[idx] = Direction.valueOf(directionsArray.getString(idx).toUpperCase());
            }

            return new Move(numMoves, directions, endSound);
        } catch (JSONException ex) {
            return new Move(numMoves, endSound);
        }
    }

}
