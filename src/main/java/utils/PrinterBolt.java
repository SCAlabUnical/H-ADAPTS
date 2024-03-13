package utils;

import org.apache.storm.topology.BasicOutputCollector;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseBasicBolt;
import org.apache.storm.tuple.Tuple;

public class PrinterBolt extends BaseBasicBolt {

    public void execute(Tuple input, BasicOutputCollector collector) {
        System.out.println("TUPLE:");
        for (String field : input.getFields())
            System.out.println(field + ": " + input.getValueByField(field));
        System.out.println("\n");
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
    }
}
