public class Consts {

    // serial communication
    public static final String PORT_NAME = "cu.usbmodem";

    // save util
    public final static boolean SAVE_ENABLED = true;
    public final static String CSV_PATH = "./TestEnvironmentGUI/data/";
    public final static String DATE_FORMAT_TODAY = "yyyy-MM-dd";

    // game loop
    public static final int DELAY = 25; // 25 ms delay between ticks
    public static final int MAX_ROUNDS = 5;

    // radii
    public static final int ACTIVATION_RADIUS = 300;
    public static final int MAGNET_RADIUS = 100;
    public static final int TARGET_RADIUS = 50;
    public static final int GUESS_RADIUS = 25;

    // bit strings
    public static final String RESET_COILS = "00000000000000000000000";
    public static final String CALIBRATE_COILS = "00000000000100000000000";
    
}
