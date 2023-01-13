package smarttvsearch.keyboard;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.LinkedList;
import org.json.JSONObject;
import org.json.JSONArray;
import smarttvsearch.utils.FileUtils;
import smarttvsearch.utils.Direction;
import smarttvsearch.utils.search.BFSState;


public class Keyboard {

    private HashMap<String, String[]> adjacencyList;
    private HashMap<String, String[]> wraparoundMap;
    private ArrayList<HashMap<String, String[]>> shortcutList;

    private static String[] DIRECTIONS = { Direction.LEFT.name().toLowerCase(), Direction.RIGHT.name().toLowerCase(), Direction.UP.name().toLowerCase(), Direction.DOWN.name().toLowerCase() };
    private static HashMap<Direction, int[]> DIRECTION_INDICES = createDirectionMap();
    private static final int LEFT_INDEX = 0;
    private static final int RIGHT_INDEX = 1;
    private static final int UP_INDEX = 2;
    private static final int DOWN_INDEX = 3;

    public Keyboard(String path) {
        // Read in the JSON
        JSONObject jsonKeyboard = FileUtils.readJsonObject(path);
        JSONObject jsonAdjList = jsonKeyboard.getJSONObject("adjacency_list");
        
        // Read the adjacency list into a Map
        adjacencyList = new HashMap<String, String[]>();

        String[] adjacent;
        JSONObject record;
        String direction;
        String key;

        Iterator<String> keys = jsonAdjList.keys();

        while (keys.hasNext()) {
            key = keys.next();
            adjacent = new String[Keyboard.DIRECTIONS.length];  // Order left, right, up, down
            record = jsonAdjList.getJSONObject(key);  // Get the adjacent keys for the current key

            for (int idx = 0; idx < Keyboard.DIRECTIONS.length; idx++) {
                direction = Keyboard.DIRECTIONS[idx];

                if (!record.isNull(direction)) {
                    adjacent[idx] = record.getString(direction);
                }
            }

            adjacencyList.put(key, adjacent);
        }

        // Read the wraparound connections into a Map
        wraparoundMap = new HashMap<String, String[]>();

        JSONObject jsonWraparound = jsonKeyboard.getJSONObject("wraparound");
        keys = jsonWraparound.keys();

        while (keys.hasNext()) {
            key = keys.next();
            adjacent = new String[Keyboard.DIRECTIONS.length];  // Order left, right, up, down
            record = jsonWraparound.getJSONObject(key);  // Get the adjacent keys for the current key

            for (int idx = 0; idx < Keyboard.DIRECTIONS.length; idx++) {
                direction = Keyboard.DIRECTIONS[idx];

                if (!record.isNull(direction)) {
                    adjacent[idx] = record.getString(direction);
                }
            }

            wraparoundMap.put(key, adjacent);
        }

        // Read the shortcut connections into a List of Maps
        shortcutList = new ArrayList<HashMap<String, String[]>>();

        JSONArray jsonShortcutArray = jsonKeyboard.getJSONArray("shortcuts");
        int numShortcuts = jsonShortcutArray.length();

        for (int shortcutIdx = 0; shortcutIdx < numShortcuts; shortcutIdx++) {
            JSONObject jsonShortcuts = jsonShortcutArray.getJSONObject(shortcutIdx);
            keys = jsonShortcuts.keys();

            HashMap<String, String[]> shortcutMap = new HashMap<String, String[]>();

            while (keys.hasNext()) {
                key = keys.next();
                adjacent = new String[Keyboard.DIRECTIONS.length];  // Order left, right, up, down
                record = jsonShortcuts.getJSONObject(key);  // Get the adjacent keys for the current key

                for (int idx = 0; idx < Keyboard.DIRECTIONS.length; idx++) {
                    direction = Keyboard.DIRECTIONS[idx];

                    if (!record.isNull(direction)) {
                        adjacent[idx] = record.getString(direction);
                    }
                }

                shortcutMap.put(key, adjacent);
            }

            shortcutList.add(shortcutMap);
        }
    }

    public Set<String> getNeighbors(String key, boolean useWraparound, boolean useShortcuts, Direction direction) {
        Set<String> neighbors = new HashSet<String>();
        String[] adjacentKeys = this.adjacencyList.get(key);

        if (adjacentKeys == null) {
            return neighbors;
        }

        int[] directionIndices = Keyboard.DIRECTION_INDICES.get(direction);

        String[] wraparoundKeys = this.wraparoundMap.get(key);
        String[] shortcutKeys;

        String neighborKey;
        String wraparoundKey;
        String shortcutKey;

        for (int idx : directionIndices) {
            // Add neighbors from the standard adjacency list
            neighborKey = adjacentKeys[idx];
            if (neighborKey != null) {
                neighbors.add(neighborKey);
            }

            // Add neighbors from the wraparound list
            if (useWraparound && (wraparoundKeys != null)) {
                wraparoundKey = wraparoundKeys[idx];
                if (wraparoundKey != null) {
                    neighbors.add(wraparoundKey);
                }
            }

            // Add neighbors from the possible list of shortcuts
            if (useShortcuts) {
                for (HashMap<String, String[]> shortcutMap : this.shortcutList) {
                    shortcutKeys = shortcutMap.get(key);

                    if (shortcutKeys != null) {
                        shortcutKey = shortcutKeys[idx];
                        if (shortcutKey != null) {
                            neighbors.add(shortcutKey);
                        }
                    }
                }
            }
        }

        return neighbors;
    }

    public Set<String> getKeysForDistance(String startKey, int distance, boolean useWraparound, boolean useShortcuts, Direction[] directions) {
        LinkedList<BFSState> frontier = new LinkedList<BFSState>();

        BFSState initState = new BFSState(startKey, 0);
        frontier.push(initState);

        Set<String> visited = new HashSet<String>();
        Set<String> result = new HashSet<String>();
        int currentDist;

        visited.add(startKey);

        while (!frontier.isEmpty()) {
            BFSState current = frontier.pop();
            currentDist = current.getDistance();

            if (currentDist == distance) {
                result.add(current.getKey());
            } else {
                Set<String> neighbors = this.getNeighbors(current.getKey(), useWraparound, useShortcuts, directions[currentDist]);

                for (String neighbor : neighbors) {
                    if (!visited.contains(neighbor)) {
                        BFSState nextState = new BFSState(neighbor, currentDist + 1);
                        frontier.add(nextState);
                        visited.add(neighbor);
                    }
                }
            }
        }

        return result;
    }

    public Set<String> getKeysForDistanceCumulative(String startKey, int distance, boolean useWraparound, boolean useShortcuts, Direction[] directions) {
        Set<String> result = this.getKeysForDistance(startKey, distance, false, false, directions);

        if (useWraparound) {
            result.addAll(this.getKeysForDistance(startKey, distance, true, false, directions));
        }

        if (useShortcuts) {
            result.addAll(this.getKeysForDistance(startKey, distance, false, true, directions));
        }

        if (useWraparound && useShortcuts) {
            result.addAll(this.getKeysForDistance(startKey, distance, true, true, directions));
        }

        return result;
    }

    private static HashMap<Direction, int[]> createDirectionMap() {
        HashMap<Direction, int[]> result = new HashMap<Direction, int[]>();

        int[] anyDirections = { LEFT_INDEX, RIGHT_INDEX, UP_INDEX, DOWN_INDEX };
        result.put(Direction.ANY, anyDirections);

        int[] horizontalDirections = { LEFT_INDEX, RIGHT_INDEX };
        result.put(Direction.HORIZONTAL, horizontalDirections);

        int[] verticalDirections = { UP_INDEX, DOWN_INDEX };
        result.put(Direction.VERTICAL, verticalDirections);

        int[] leftDirections = { LEFT_INDEX };
        result.put(Direction.LEFT, leftDirections);

        int[] rightDirections = { RIGHT_INDEX };
        result.put(Direction.RIGHT, rightDirections);

        int[] downDirections = { DOWN_INDEX };
        result.put(Direction.DOWN, downDirections);

        int[] upDirections = { UP_INDEX };
        result.put(Direction.UP, upDirections);

        return result;
    }
}
