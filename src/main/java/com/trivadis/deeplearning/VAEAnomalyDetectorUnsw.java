package com.trivadis.deeplearning;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.*;


public class VAEAnomalyDetectorUnsw {

    private static final Logger log = LoggerFactory.getLogger(VAEAnomalyDetectorUnsw.class);
    private int seed = 777;
    File modelFile = new File("VAEUnsw.zip");
    boolean saveUpdater = false;
    boolean modelExists = true;

    private int minibatchSize = 128;
    //private double learningRate = 0.0001;
    private double learningRate = 0.01;
    private int numEpochs = 500;
    private int reconstructionNumSamples = 32;

    private int inputSize = 194; // number of features
    private int[] encoderSizes = new int[]{32,8};
    private int[] decoderSizes = new int[]{8,32};

    private int latentSize = 2;
    private Activation latentActivation = Activation.IDENTITY;

    private String trainFile = "X_train_with_label.csv";
    private String testFile = "X_test_with_label.csv";

    private double plotMin = -5;
    private double plotMax = 5;

    static Map<Integer, String> attackClasses = new HashMap<Integer, String>() {

        {
            put(1, "Normal");
            put(2, "Analysis");
            put(3, "Backdoor");
            put(4, "DoS");
            put(5, "Exploits");
            put(6, "Fuzzers");
            put(7, "Generic");
            put(8, "Reconnaissance");
            put(9, "Shellcode");
            put(10, "Worms");
        }

        ;
    };



    public static void main(String[] args) throws IOException, InterruptedException {
        new VAEAnomalyDetectorUnsw().run();
    }

    public void run() throws IOException, InterruptedException {

        RecordReader trainReader = new CSVRecordReader(1);
        trainReader.initialize(new FileSplit(new File(trainFile)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, 100, 194, 10);

        RecordReader testReader = new CSVRecordReader(1);
        testReader.initialize(new FileSplit(new File(testFile)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, 100, 194, 10);

        RecordReader testReader2 = new CSVRecordReader(1);
        testReader2.initialize(new FileSplit(new File(testFile)));
        DataSet testdata = new RecordReaderDataSetIterator(testReader2, 175342, 194, 10).next();
        INDArray testFeatures = testdata.getFeatures();
        INDArray testLabels = testdata.getLabels();


        MultiLayerNetwork net;

       if(modelExists) {
           net = ModelSerializer.restoreMultiLayerNetwork(modelFile);
       } else {
           Nd4j.getRandom().setSeed(seed);

           MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                   .learningRate(learningRate)
                   .updater(Updater.ADAM)
                   .weightInit(WeightInit.XAVIER)
                   .list()
                   .layer(0, new VariationalAutoencoder.Builder()
                           .activation(Activation.RELU)
                           .encoderLayerSizes(encoderSizes)
                           .decoderLayerSizes(decoderSizes)
                           .pzxActivationFunction(latentActivation)     //p(z|data) activation function
                           //.reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID)) // for modelling binary data (or data in range 0 to 1)
                           .reconstructionDistribution(new GaussianReconstructionDistribution(Activation.TANH))
                           .nIn(inputSize)
                           .nOut(latentSize)
                           .build())
                   .pretrain(true).backprop(false).build();

           net = new MultiLayerNetwork(conf);
           net.init();
           net.setListeners(new ScoreIterationListener(20));

           for (int i = 0; i < numEpochs; i++) {
               net.fit(trainIter);
               System.out.println("Finished epoch " + (i + 1) + " of " + numEpochs);
           }
           ModelSerializer.writeModel(net, modelFile, saveUpdater);

       }

        log.debug(net.summary());

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);
        INDArray latentSpaceValues = vae.activate(testFeatures, false);

        Map<Integer, List<Pair<Double, INDArray>>> listsByDigit = new HashMap<>();
        for (int i = 0; i < 10; i++) listsByDigit.put(i, new ArrayList<>());

        int q = 0;
        while (testIter.hasNext()) {

            DataSet ds = testIter.next();
            INDArray features = ds.getFeatures();
            INDArray labels = Nd4j.argMax(ds.getLabels(), 1);
            if(q % 1000 == 0) {
                log.debug(String.valueOf(labels.getInt(0)));
                log.debug(String.valueOf(features.getRow(0)));
            }

            int nRows = features.rows();

            INDArray reconstructionErrorEachExample = vae.reconstructionLogProbability(features, reconstructionNumSamples);    //Shape: [minibatchSize, 1]

            for (int j = 0; j < nRows; j++) {
                INDArray example = features.getRow(j);
                int label = (int) labels.getDouble(j);
                double score = reconstructionErrorEachExample.getDouble(j);
                listsByDigit.get(label).add(new Pair<>(score, example));
            }
            q++;
        }

        Evaluation eval = new Evaluation();

        Comparator<Pair<Double, INDArray>> c = new Comparator<Pair<Double, INDArray>>() {
            @Override
            public int compare(Pair<Double, INDArray> o1, Pair<Double, INDArray> o2) {
                //Negative: return highest reconstruction probabilities first -> sorted from best to worst
                return -Double.compare(o1.getFirst(), o2.getFirst());
            }
        };

        for (List<Pair<Double, INDArray>> list : listsByDigit.values()) {
            Collections.sort(list, c);
        }

        List<DescriptiveStatistics> statsList = new ArrayList<DescriptiveStatistics>();


        // summary statistics per class
        for (int i = 0; i < 10; i++) {
            DescriptiveStatistics stats = new DescriptiveStatistics();
            List<Pair<Double, INDArray>> list = listsByDigit.get(i);
            for( int j = 0; j < list.size(); j++) {
                stats.addValue(list.get(j).getFirst());
            }
            statsList.add(stats);
        }

        for (int i = 0; i < 10; i++) {
            System.out.println("Reconstruction probability summary stats for class: " + attackClasses.get(i+1));
            System.out.println("Min: " + String.valueOf((double) Math.round(statsList.get(i).getMin() * 100)/100));
            System.out.println("Median: " + String.valueOf((double)Math.round(statsList.get(i).getPercentile(50) * 100)/100));
            System.out.println("Max: " + String.valueOf((double)Math.round(statsList.get(i).getMax() * 100)/100));
            System.out.println("Mean: " + String.valueOf((double)Math.round(statsList.get(i).getMean() * 100)/100));
            System.out.println("SD: " + String.valueOf((double)Math.round(statsList.get(i).getStandardDeviation() * 100)/100));
            System.out.println();
        }

        // best and worst scores per class
        for (int i = 0; i < 10; i++) {
            List<Pair<Double, INDArray>> list = listsByDigit.get(i);
            System.out.println("Best/worst reconstruction probabilities for class: " + attackClasses.get(i+1));
            StringBuilder best = new StringBuilder("Best: ");
            for (int j = 0; j < 5; j++) {
                Double score = list.get(j).getFirst();
                best.append(String.valueOf((double)Math.round(score * 100)/100));
                if (j != 4) best.append(", ");
            }
            System.out.println(best.toString());

            StringBuilder worst = new StringBuilder("Worst: ");
            for (int j = 0; j < 5; j++) {
                Double score = list.get(list.size() - j - 1).getFirst();
                worst.append(String.valueOf((double)Math.round(score * 100)/100));
                if (j != 4) worst.append(", ");
            }
            System.out.println(worst.toString());
            System.out.println();
        }

    VAEAnomalyDetectorUnsw.PlotUtil.plotData(latentSpaceValues, testLabels, plotMin, plotMax);



    }

    public static class PlotUtil {



        private static JFreeChart createChart(INDArray features, INDArray labels, double axisMin, double axisMax) {
            return createChart(features, labels, axisMin, axisMax, "Variational Autoencoder Latent Space - UNSW Test Set");
        }

        private static JFreeChart createChart(INDArray features, INDArray labels, double axisMin, double axisMax, String title ) {

            XYDataset dataset = createDataSet(features, labels);

            JFreeChart chart = ChartFactory.createScatterPlot(title,
                    "X", "Y", dataset, PlotOrientation.VERTICAL, true, true, false);

            XYPlot plot = (XYPlot) chart.getPlot();
            plot.getRenderer().setBaseOutlineStroke(new BasicStroke(0));
            plot.setNoDataMessage("NO DATA");

            plot.setDomainPannable(false);
            plot.setRangePannable(false);
            plot.setDomainZeroBaselineVisible(true);
            plot.setRangeZeroBaselineVisible(true);

            plot.setDomainGridlineStroke(new BasicStroke(0.0f));
            plot.setDomainMinorGridlineStroke(new BasicStroke(0.0f));
            plot.setDomainGridlinePaint(Color.blue);
            plot.setRangeGridlineStroke(new BasicStroke(0.0f));
            plot.setRangeMinorGridlineStroke(new BasicStroke(0.0f));
            plot.setRangeGridlinePaint(Color.blue);

            plot.setDomainMinorGridlinesVisible(true);
            plot.setRangeMinorGridlinesVisible(true);

            XYLineAndShapeRenderer renderer
                    = (XYLineAndShapeRenderer) plot.getRenderer();
            renderer.setSeriesOutlinePaint(0, Color.black);
            renderer.setUseOutlinePaint(true);
            NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
            domainAxis.setAutoRangeIncludesZero(false);
            domainAxis.setRange(axisMin, axisMax);

            domainAxis.setTickMarkInsideLength(2.0f);
            domainAxis.setTickMarkOutsideLength(2.0f);

            domainAxis.setMinorTickCount(2);
            domainAxis.setMinorTickMarksVisible(true);

            NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
            rangeAxis.setTickMarkInsideLength(2.0f);
            rangeAxis.setTickMarkOutsideLength(2.0f);
            rangeAxis.setMinorTickCount(2);
            rangeAxis.setMinorTickMarksVisible(true);
            rangeAxis.setRange(axisMin, axisMax);
            return chart;
        }

        private static XYDataset createDataSet(INDArray features, INDArray labelsOneHot){
            int nRows = features.rows();

            int nClasses = labelsOneHot.columns();

            XYSeries[] series = new XYSeries[nClasses];
            for( int i=0; i< nClasses; i++){
                series[i] = new XYSeries(attackClasses.get(i+1));
            }
            INDArray classIdx = Nd4j.argMax(labelsOneHot, 1);
            for( int i=0; i<nRows; i++ ){
                int idx = classIdx.getInt(i);
                series[idx].add(features.getDouble(i, 0), features.getDouble(i, 1));
            }

            XYSeriesCollection c = new XYSeriesCollection();
            for( XYSeries s : series) c.addSeries(s);
            return c;
        }


        public static void plotData(INDArray values, INDArray labels, double axisMin, double axisMax){

            JPanel panel = new ChartPanel(createChart(values, labels, axisMin, axisMax));
            final JFrame f = new JFrame();
            f.add(panel, BorderLayout.CENTER);
            f.setLayout(new BorderLayout());
            f.add(panel, BorderLayout.CENTER);
            f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            f.pack();
            f.setVisible(true);
        }


    }



}
