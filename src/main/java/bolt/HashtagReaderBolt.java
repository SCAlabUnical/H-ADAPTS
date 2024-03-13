package bolt;

import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.IRichBolt;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.tuple.*;
import tweet.Tweet;

import java.util.Map;

public class HashtagReaderBolt implements IRichBolt {
    private OutputCollector collector;

    @Override
    public void prepare(Map<String, Object> conf, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
    }

    @Override
    public void execute(Tuple tuple) {
        Tweet tweet = (Tweet) tuple.getValueByField("tweet");
        long date = tweet.getTime();
        long time = date * 1000;
        for (String hashtag : tweet.getHashtags()) {
            this.collector.emit(new Values(hashtag, date, time));
        }
    }

    @Override
    public void cleanup() {
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("hashtag", "date", "time"));
    }

    @Override
    public Map<String, Object> getComponentConfiguration() {
        return null;
    }
}
