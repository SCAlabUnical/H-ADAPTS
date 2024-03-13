package bolt;

import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.IRichBolt;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.tuple.*;

import java.net.*;
import java.io.*;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

public class SocketBolt implements IRichBolt {
    private OutputCollector collector;
    private static Socket clientSocket;
    private static PrintWriter printWriter;
    private static ObjectMapper mapper;

    @Override
    public void prepare(Map<String, Object> conf, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
        try {
            clientSocket = new Socket("localhost", 8082);
            printWriter = new PrintWriter(clientSocket.getOutputStream(), true);
            mapper = new ObjectMapper();
        } catch (Exception e) {
        }
    }

    @Override
    public void execute(Tuple tuple) {
        try {
            String ret = mapper.writeValueAsString(tuple.getValues());
            printWriter.println(ret);
        } catch (Exception e) {
        }
    }

    @Override
    public void cleanup() {
        try {
            printWriter.close();
            clientSocket.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
    }

    @Override
    public Map<String, Object> getComponentConfiguration() {
        return null;
    }

}
