package tweet;

import java.util.List;

/**
 * Interface for tweet object in case of other spouts
 */
public interface Tweet {
    /**
     * Returns the time the tweet was created in unix time
     *
     * @return the time the tweet was created in unix time
     */
    long getTime();

    /**
     * Returns the text of the tweet
     *
     * @return the the text of the tweet
     */
    String getText();

    /**
     * Returns the tweet's hashtags
     *
     * @return a list containing the tweet's hashtags
     */
    List<String> getHashtags();


    /**
     * Returns the retweet status of the tweet
     *
     * @return true if the tweet is a retweet
     */
    boolean isRetweet();

    /**
     * Factory method to create tweets
     *
     * @return a Tweet instance
     */
    Tweet CreateTweet(Object o);
}