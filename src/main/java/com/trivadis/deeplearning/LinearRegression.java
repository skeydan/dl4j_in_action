package com.trivadis.deeplearning;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class LinearRegression {

    public static void main(String[] args) {

        int seed = 777;
        int numSamples = 64;
        int numFeatures = 1000;
        int hiddenDim = 100;
        int outputDim = 10;

        int numEpochs = 100;

        INDArray X = Nd4j.randn(numSamples, numFeatures);
        INDArray Y = Nd4j.randn(numSamples, outputDim);

        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(numEpochs)
                .miniBatch(false)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .weightInit(WeightInit.XAVIER) // As per Glorot and Bengio 2010: Gaussian distribution with mean 0, variance 2.0/(fanIn + fanOut)
                .updater(Updater.SGD)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numFeatures).nOut(hiddenDim)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(hiddenDim).nOut(outputDim).build())
                .build());

        net.init();
        System.out.println(net.summary());
        net.setListeners(new ScoreIterationListener(10));

        net.fit(X, Y);

    }
}
