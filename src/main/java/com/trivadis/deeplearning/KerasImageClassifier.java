package com.trivadis.deeplearning;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;
import java.io.IOException;

public class KerasImageClassifier {

    public static void main(String[] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        //String path = args[0];
        String path = "/home/key/pics/claude.jpeg";
        String modelPath = "/home/key/code/R/cracks/model_filter323264_kernel3_epochs20_lr001.h5";

        MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights(modelPath);
        //network.setUpdater(); if we wanted to further train the model
        network.summary();

        File file = new File(path);
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(file);

        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);

        //INDArray[] output = network.output(false, image);
    }
}
