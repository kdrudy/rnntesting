package com.kyrutech.testing;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Created by Kyle Rudy on 3/11/2017.
 */
public class W2VTesting {
    public static void main(String args[]) {
        int totalWords = 100;
        int nearestWords = 3;
        W2VTesting w = new W2VTesting();
        w.train();

        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("timecubeWord2Vec.txt");

        int words = word2Vec.getVocab().numWords();
        System.out.println("Num of words: " + word2Vec.getVocab().numWords());
        int randWord = (int) (Math.random()*words);

        String word = word2Vec.getVocab().wordAtIndex(randWord);
        ArrayList<String> lst = (ArrayList<String>) word2Vec.wordsNearest(word, nearestWords);
        StringBuilder sb = new StringBuilder();
        sb.append(word);
        sb.append(" ");
        for(int i = 1;i<totalWords;i++) {
            int newWordIndex = (int) (Math.random()*nearestWords);
            word = lst.get(newWordIndex);
            sb.append(word);
            sb.append(" ");
            lst = (ArrayList<String>) word2Vec.wordsNearest(word, nearestWords);
        }
        System.out.println(sb.toString());

    }

    public void train() {
        SentenceIterator iter = new LineSentenceIterator(new File("./src/main/resources/timecube.txt"));

        TokenizerFactory t = new DefaultTokenizerFactory();
        //t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .iterations(50)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        vec.fit();

        WordVectorSerializer.writeWord2VecModel(vec, "timecubeWord2Vec.txt");
    }
}
