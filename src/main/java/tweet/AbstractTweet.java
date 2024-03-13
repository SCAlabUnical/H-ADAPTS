package tweet;

import java.io.Serializable;
import java.util.Date;
import java.util.List;

public abstract class AbstractTweet implements Tweet, Serializable {
    protected long time;
    protected String text;
    protected List<String> hashtags;
    protected boolean isRetweet;

    @Override
    public long getTime() {
        return time;
    }

    @Override
    public String getText() {
        return text;
    }

    @Override
    public List<String> getHashtags() {
        return hashtags;
    }

    @Override
    public boolean isRetweet() {
        return isRetweet;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Date: " + new Date(time) + "\n");
        sb.append("isRetweet: " + isRetweet + "\n");
        sb.append(text + "\n");
        if (!hashtags.isEmpty()) {
            for (String h : hashtags)
                sb.append(h + ", ");
            sb.delete(sb.length() - 2, sb.length());
            sb.append("\n");
        }
        return sb.toString();
    }
}
