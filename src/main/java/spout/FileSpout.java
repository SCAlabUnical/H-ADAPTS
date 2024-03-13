package spout;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Map;

import com.opencsv.CSVReader;
import org.apache.storm.Config;
import org.apache.storm.spout.SpoutOutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichSpout;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Values;
import tweet.Tweet;
import org.apache.storm.utils.Utils;

/**
 * Spout for files that are a list of tweet jsons
 */
@SuppressWarnings("serial")
public class FileSpout extends BaseRichSpout {
    SpoutOutputCollector collector;
    File file;
    CSVReader reader;
    Tweet tweetTemplate;
    Boolean end = false;

    public FileSpout(String path, Tweet template) {
        file = new File(path);
        tweetTemplate = template;
    }

    public void open(Map<String, Object> conf, TopologyContext context, SpoutOutputCollector collector) {
        this.collector = collector;
        try {
            reader = new CSVReader(new FileReader(file));
            reader.readNext();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void nextTuple() {
        String[] tweet = null;
        try {
            tweet = reader.readNext();
        } catch (IOException e) {
        }
        if (tweet != null) {
            Utils.sleep(2);
            collector.emit(new Values(tweetTemplate.CreateTweet(tweet)));
        } else if (!end) {
            System.out.println("SPOUT EXHAUSTED");
            end = true;
        }
    }

    public void close() {
        try {
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Map<String, Object> getComponentConfiguration() {
        Config ret = new Config();
        ret.setMaxTaskParallelism(1);
        return ret;
    }

    public void ack(Object id) {
    }

    public void fail(Object id) {
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("tweet"));
    }
}