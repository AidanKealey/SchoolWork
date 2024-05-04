import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;

import com.opencsv.CSVWriter;

public class SaveUtil {

    public static void saveToCsv(String username, ArrayList<Integer> scores, ArrayList<Double> times) {
        System.out.println("Saving score data for user: \""+username+"\"...");

        // convert scores into string array
        int csvColumns = scores.size() + times.size() + 1;
        String[] data = new String[csvColumns];
        data[0] = username;
        for (int i=1; i<csvColumns; i++) {
            data[i] = (i < 6) ? scores.get(i-1).toString() : times.get(i-6).toString();
        }

        // write data to csv file
        String path = Consts.CSV_PATH + getTodayDate() + ".csv";
        File file = new File(path);
        try {
            CSVWriter writer = new CSVWriter(new FileWriter(file, true));
            writer.writeNext(data);
            writer.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String getTodayDate() {
        Calendar c = Calendar.getInstance();
        SimpleDateFormat sdf = new SimpleDateFormat(Consts.DATE_FORMAT_TODAY);
        return sdf.format(c.getTime());
    }

}
