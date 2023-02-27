package smarttvsearch.utils.sounds;


public class AppleTVSound implements SmartTVSound {

    private static final String KEYBOARD_SELECT = "keyboard_select";
    private static final String KEYBOARD_DELETE = "keyboard_delete";
    private static final String KEYBOARD_MOVE = "keyboard_move";
    private static final String TOOLBAR_MOVE = "toolbar_move";

    private String sound;

    public AppleTVSound(String sound) {
        if (!(sound.equals(AppleTVSound.KEYBOARD_SELECT)) && !(sound.equals(AppleTVSound.KEYBOARD_DELETE)) && !(sound.equals(AppleTVSound.KEYBOARD_MOVE)) && !(sound.equals(AppleTVSound.TOOLBAR_MOVE))) {
            throw new IllegalArgumentException("Unknown sound name: " + sound);
        }

        this.sound = sound;
    }

    @Override
    public String getSoundName() {
        return this.sound;
    }

    public boolean isMove() {
        return this.getSoundName().equals(AppleTVSound.KEYBOARD_MOVE);
    }

    public boolean isKeySelect() {
        return this.getSoundName().equals(AppleTVSound.KEYBOARD_SELECT);
    }

    public boolean isToolbarMove() {
        return this.getSoundName().equals(AppleTVSound.TOOLBAR_MOVE);
    }

    @Override
    public boolean isDelete() {
        return this.getSoundName().equals(AppleTVSound.KEYBOARD_DELETE);
    }

}
