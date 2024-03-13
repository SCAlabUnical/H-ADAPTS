package bolt;

import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseWindowedBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;
import org.apache.storm.windowing.TupleWindow;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class CounterBolt extends BaseWindowedBolt {

    private List<String> oldTrending;

    private OutputCollector collector;

    private int k;
    private double threshold;

    public CounterBolt(int k, double threshold, String[] startTrending) {
        this.k = k;
        this.threshold = threshold;
        oldTrending = new LinkedList<>(Arrays.asList(startTrending));
    }

    @Override
    public void prepare(Map<String, Object> stormConf, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
    }

    public static <K, V extends Comparable<? super V>> List<K> topK(Map<K, V> map, int k) {
        List<Entry<K, V>> list = new ArrayList<>(map.entrySet());
        list.sort(Entry.comparingByValue(new Comparator<V>() {
            @Override
            public int compare(V o1, V o2) {
                return -o1.compareTo(o2);
            }
        }));
        LinkedList<K> result = new LinkedList<>();
        for (Entry<K, V> entry : list) {
            result.add(entry.getKey());
            if (result.size() == k)
                break;
        }
        return result;
    }

    @Override
    public void execute(TupleWindow inputWindow) {
        HashMap<String, Integer> trendingMap = new HashMap<>();
        for (Tuple tuple : inputWindow.get()) {
            String h = tuple.getStringByField("hashtag");
            if (!trendingMap.containsKey(h))
                trendingMap.put(h, 1);
            else
                trendingMap.put(h, trendingMap.get(h) + 1);
        }
        List<String> newTrending = topK(trendingMap, k);

        double intersection = 0;
        for (int i = 0; i < oldTrending.size(); i++) {
            if (newTrending.contains(oldTrending.get(i)))
                intersection += Math.min(k - i, k - newTrending.indexOf(oldTrending.get(i)));
        }
        try {
            BufferedWriter out = new BufferedWriter(
                    new FileWriter("topics.txt", true));
            PrintWriter pw = new PrintWriter(out);
            pw.println(newTrending);
            pw.println("");
            pw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        double union = k * (k + 1) / 2.0;
        double jd = 1 - intersection / union;
        boolean train = false;
        if (jd >= threshold) {
            oldTrending = newTrending;
            train = true;
        }
        collector.emit(new Values(inputWindow.getEndTimestamp() / 1000, train));
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("endTimestamp", "train"));
    }
}
