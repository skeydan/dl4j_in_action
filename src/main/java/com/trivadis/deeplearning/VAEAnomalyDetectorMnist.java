package com.trivadis.deeplearning;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;
import java.util.List;

public class VAEAnomalyDetectorMnist extends VAEAnomalyDetector {

    private static final Logger log = LoggerFactory.getLogger(VAEAnomalyDetectorMnist.class);
    private int seed = 777;

    private static int minibatchSize = 128;
    private static int numEpochs = 50;
    private static int reconstructionNumSamples = 16;

    private static int inputSize = 28 * 28; // number of features
    private static int[] encoderSizes = new int[]{256, 32};
    private static int[] decoderSizes = new int[]{32, 256};
    private int latentSize = 2;

    private Activation latentActivation = Activation.IDENTITY;

    private double plotMin = -5;
    private double plotMax = 5;


    public static void main(String[] args) throws IOException, InterruptedException {
        new VAEAnomalyDetectorMnist().run();
    }

    private void run() throws IOException, InterruptedException {

        DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, true, seed);
        DataSet next = trainIter.next();
        log.debug(next.getLabels().shapeInfoToString());
        log.debug(next.getFeatures().shapeInfoToString());

        // for evaluation
        DataSetIterator testIter = new MnistDataSetIterator(minibatchSize, false, seed);
        // for plotting
        DataSet testdata = new MnistDataSetIterator(10000, false, seed).next();
        INDArray testFeatures = testdata.getFeatures();
        INDArray testLabels = testdata.getLabels();

        Nd4j.getRandom().setSeed(seed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.05)
                .updater(Updater.ADAM)
                .weightInit(WeightInit.XAVIER)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new VariationalAutoencoder.Builder()
                        .activation(Activation.LEAKYRELU)
                        .encoderLayerSizes(encoderSizes)
                        .decoderLayerSizes(decoderSizes)
                        .pzxActivationFunction(latentActivation)     //p(z|data) activation function
                        //Bernoulli reconstruction distribution + sigmoid activation - for modelling binary data (or data in range 0 to 1)
                        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                        .nIn(inputSize)
                        .nOut(latentSize)
                        .build())
                .pretrain(true).backprop(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);

        net.setListeners(new ScoreIterationListener(100));

        for (int i = 0; i < numEpochs; i++) {
            net.fit(trainIter);
            log.info("Finished epoch " + (i + 1) + " of " + numEpochs);
        }

        INDArray latentSpaceValues = vae.activate(testFeatures, false);


        Map<Integer, List<Pair<Double, INDArray>>> listsByDigit = new HashMap<>();
        for (int i = 0; i < 10; i++) listsByDigit.put(i, new ArrayList<>());

         while (testIter.hasNext()) {
            DataSet ds = testIter.next();
            INDArray features = ds.getFeatures();
            INDArray labels = Nd4j.argMax(ds.getLabels(), 1);
            int nRows = features.rows();

            INDArray reconstructionErrorEachExample = vae.reconstructionLogProbability(features, reconstructionNumSamples);    //Shape: [minibatchSize, 1]

            for (int j = 0; j < nRows; j++) {
                INDArray example = features.getRow(j);
                int label = (int) labels.getDouble(j);
                double score = reconstructionErrorEachExample.getDouble(j);
                listsByDigit.get(label).add(new Pair<>(score, example));
            }
        }


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


        // best and worst scores per class
        for (int i = 0; i < 10; i++) {
            List<Pair<Double, INDArray>> list = listsByDigit.get(i);
            System.out.println("Reconstruction probabilities for class: " + i);
            System.out.println("Best:");
            for (int j = 0; j < 5; j++) {
                // the original number's pixels
                Double score = list.get(j).getFirst();
                System.out.println(score);
            }
            System.out.println("Worst:");
            for (int j = 0; j < 5; j++) {
                // the original number's pixels
                Double score = list.get(list.size() - j - 1).getFirst();
                System.out.println(score);
            }
        }

        //Select the 5 best and 5 worst numbers (by reconstruction probability) for each digit
        List<INDArray> best = new ArrayList<>(50);
        List<INDArray> worst = new ArrayList<>(50);

        List<INDArray> bestReconstruction = new ArrayList<>(50);
        List<INDArray> worstReconstruction = new ArrayList<>(50);

        for (int i = 0; i < 10; i++) {
            List<Pair<Double, INDArray>> list = listsByDigit.get(i);
            for (int j = 0; j < 5; j++) {
                // the original number's pixels
                INDArray b = list.get(j).getSecond();
                INDArray w = list.get(list.size() - j - 1).getSecond();

                // latent representation
                INDArray pzxMeanBest = vae.preOutput(b);
                INDArray reconstructionBest = vae.generateAtMeanGivenZ(pzxMeanBest);

                INDArray pzxMeanWorst = vae.preOutput(w);
                INDArray reconstructionWorst = vae.generateAtMeanGivenZ(pzxMeanWorst);

                best.add(b);
                bestReconstruction.add(reconstructionBest);
                worst.add(w);
                worstReconstruction.add(reconstructionWorst);
            }
        }


        MNISTVisualizer bestVisualizer = new MNISTVisualizer(2.0, best, "Best (Highest Rec. Prob)");
        bestVisualizer.visualize();

        MNISTVisualizer bestReconstructions = new MNISTVisualizer(2.0, bestReconstruction, "Best - Reconstructions");
        bestReconstructions.visualize();

        MNISTVisualizer worstVisualizer = new MNISTVisualizer(2.0, worst, "Worst (Lowest Rec. Prob)");
        worstVisualizer.visualize();

        MNISTVisualizer worstReconstructions = new MNISTVisualizer(2.0, worstReconstruction, "Worst - Reconstructions");
        worstReconstructions.visualize();

        PlotUtil.plotData(latentSpaceValues, testLabels, plotMin, plotMax);



    }
    public static class MNISTVisualizer {
        private double imageScale;
        private List<INDArray> digits;  //Digits (as row vectors), one per INDArray
        private String title;
        private int gridWidth;

        public MNISTVisualizer(double imageScale, List<INDArray> digits, String title ) {
            this(imageScale, digits, title, 5);
        }

        public MNISTVisualizer(double imageScale, List<INDArray> digits, String title, int gridWidth ) {
            this.imageScale = imageScale;
            this.digits = digits;
            this.title = title;
            this.gridWidth = gridWidth;
        }

        public void visualize(){
            JFrame frame = new JFrame();
            frame.setTitle(title);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            JPanel panel = new JPanel();
            panel.setLayout(new GridLayout(0,gridWidth));

            List<JLabel> list = getComponents();
            for(JLabel image : list){
                panel.add(image);
            }

            frame.add(panel);
            frame.setVisible(true);
            frame.pack();
        }

        private List<JLabel> getComponents(){
            List<JLabel> images = new ArrayList<>();
            for( INDArray arr : digits ){
                BufferedImage bi = new BufferedImage(28,28,BufferedImage.TYPE_BYTE_GRAY);
                for( int i=0; i<784; i++ ){
                    bi.getRaster().setSample(i % 28, i / 28, 0, (int)(255*arr.getDouble(i)));
                }
                ImageIcon orig = new ImageIcon(bi);
                Image imageScaled = orig.getImage().getScaledInstance((int)(imageScale*28),(int)(imageScale*28),Image.SCALE_REPLICATE);
                ImageIcon scaled = new ImageIcon(imageScaled);
                images.add(new JLabel(scaled));
            }
            return images;
        }
    }

    public static class PlotUtil {



        private static JFreeChart createChart(INDArray features, INDArray labels, double axisMin, double axisMax) {
            return createChart(features, labels, axisMin, axisMax, "Variational Autoencoder Latent Space - MNIST Test Set");
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
            for( int i=0; i<nClasses; i++){
                series[i] = new XYSeries(String.valueOf(i));
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
