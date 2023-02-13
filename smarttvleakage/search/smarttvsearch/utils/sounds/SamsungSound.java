package smarttvsearch.utils.sounds;


public class SamsungSound implements SmartTVSound {

    private static final String SELECT = "select";
    private static final String KEY_SELECT = "key_select";
    private static final String MOVE = "move";
    private static final String DELETE = "delete";

    private String sound;

    public SamsungSound(String sound) {
        if (!(sound.equals(SamsungSound.SELECT)) && !(sound.equals(SamsungSound.KEY_SELECT)) && !(sound.equals(SamsungSound.MOVE)) && !(sound.equals(SamsungSound.DELETE))) {
            throw new IllegalArgumentException("Unknown sound name: " + sound);
        }

        this.sound = sound;
    }

    @Override
    public String getSoundName() {
        return this.sound;
    }

    public boolean isMove() {
        return this.getSoundName().equals(SamsungSound.MOVE);
    }

    public boolean isKeySelect() {
        return this.getSoundName().equals(SamsungSound.KEY_SELECT);
    }

    public boolean isSelect() {
        return this.getSoundName().equals(SamsungSound.SELECT);
    }

    public boolean isDelete() {
        return this.getSoundName().equals(SamsungSound.DELETE);
    }

}
