package tweet;

import java.util.ArrayList;
import java.util.Arrays;

public class TrainTweet extends AbstractTweet {
    /*
    csv fields:
    0 date
    1 hashtags
    */

    public TrainTweet() {
    }

    public TrainTweet(String[] values) {
        String h = values[1].replaceAll("[\\[\\] \\']+", "");
        if (!h.equals(""))
            hashtags = Arrays.asList(h.split(","));
        else
            hashtags = new ArrayList<>();
        time = Long.parseLong(values[0].split("\\.")[0]);
    }

    @Override
    public TrainTweet CreateTweet(Object s) {
        return new TrainTweet((String[]) s);
    }
}
