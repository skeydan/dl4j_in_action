package com.trivadis.deeplearning;

import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ScaleImageTransform;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class KerasImageClassifier {

    private static final Logger log = LoggerFactory.getLogger(VAEAnomalyDetectorUnsw.class);
    private int seed = 777;

    public static void main(String[] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        //String path = "/home/key/code/R/cracks/data/test/crack/img_10_1121_449.png";
        String path = "/home/key/code/R/cracks/data/test/nocrack/img_10_225_673.png";
        String modelPath = "/home/key/code/R/cracks/model_filter323264_kernel3_epochs20_lr001.h5";

        MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights(modelPath);
        //network.setUpdater(...); if we wanted to further train the model
        log.info(network.summary());

        File file = new File(path);
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(file);
        image.divi(255);
        //log.debug(String.valueOf(image));

        System.out.println(network.output(image));

        BufferedImage img = ImageIO.read(new File(path));
        JFrame f = new JFrame();
        JLabel picLabel = new JLabel(new ImageIcon(img));
        f.add(picLabel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setVisible(true);

    }
}
