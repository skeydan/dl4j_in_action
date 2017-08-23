package com.trivadis.deeplearning;


import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;
import java.io.IOException;

public class PretrainedImageClassifier {

    public static void main(String[] args) throws IOException {

        //String path = args[0];
        String path = "/home/key/Downloads/26525687115_420ed7de89_o.jpg";
        ZooModel vgg16 = new VGG16();
        ComputationGraph pretrainedNet = (ComputationGraph) vgg16.initPretrained(PretrainedType.IMAGENET);

        File file = new File(path);
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(file);

        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);

        INDArray[] output = pretrainedNet.output(false,image);
        System.out.println(TrainedModels.VGG16.decodePredictions(output[0]));





    }

}
