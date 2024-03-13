package topology;

import bolt.CounterBolt;
import bolt.HashtagReaderBolt;
import bolt.SocketBolt;
import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.topology.TopologyBuilder;
import org.apache.storm.topology.base.BaseWindowedBolt;
import org.apache.storm.utils.Utils;
import spout.FileSpout;
import tweet.TrainTweet;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

public class Topology {
    private static final TimeUnit unit = TimeUnit.DAYS;
    private static int tumblingWindowLength;
    private static int k;
    private static double threshold;
    public static String[] startTrending;

    public static void main(String[] args) throws Exception {
        try {
            File myObj = new File("../../resources/java_constants.txt");
            Scanner myReader = new Scanner(myObj);
            tumblingWindowLength = Integer.parseInt(myReader.nextLine());
            k = Integer.parseInt(myReader.nextLine());
            threshold = Double.parseDouble(myReader.nextLine());
            startTrending = myReader.nextLine().split(" ");
            myReader.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        TopologyBuilder builder = new TopologyBuilder();

        builder.setSpout("file-spout", new FileSpout("[PUT YOUR DATASET HERE]", new TrainTweet()), 1);
        builder.setBolt("hashtag-reader-bolt", new HashtagReaderBolt()).shuffleGrouping("file-spout");
        builder.setBolt("counter-bolt", new CounterBolt(k, threshold, startTrending).withTumblingWindow(new BaseWindowedBolt.Duration(tumblingWindowLength, unit)).withTimestampField("time")).shuffleGrouping("hashtag-reader-bolt");
        builder.setBolt("socket-bolt", new SocketBolt(), 1).shuffleGrouping("counter-bolt");

        Config conf = new Config();
        conf.setDebug(false);
        conf.setMessageTimeoutSecs((int) unit.toSeconds(tumblingWindowLength * 2));

        String topologyName = "topology";

        try (LocalCluster cluster = new LocalCluster();
             LocalCluster.LocalTopology topo = cluster.submitTopology(topologyName, conf, builder.createTopology())) {
            Utils.sleep(Integer.MAX_VALUE);
            cluster.shutdown();
        }
    }
}
