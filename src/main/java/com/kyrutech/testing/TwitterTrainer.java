package com.kyrutech.testing;

import org.deeplearning4j.nn.api.Layer;
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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import twitter4j.Paging;
import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by Kyle on 3/12/2017.
 */
public class TwitterTrainer {
    public static void main( String[] args ) throws Exception {
        int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
        int miniBatchSize = 32;						//Size of mini batch to use when  training
        int exampleLength = 1000;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 100;						//Total number of training epochs
        int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
        int nSamplesToGenerate = 2;					//Number of samples to generate after each training epoch
        int nCharactersToSample = 300;				//Length of each sample to generate
        String generationInitialization = null;		//Optional character initialization; a random character is used if null
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
        Random rng = new Random(12345);

        File savedNetwork = new File("TwitterSavedRNN.zip");

        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our GravesLSTM network.
        CharacterIterator iter = getTwitterIterator(miniBatchSize,exampleLength);

        //StringWordIterator iter = new StringWordIterator("./src/main/resources/timecube.txt", miniBatchSize, exampleLength, rng);

        int nOut = iter.totalOutcomes();

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.1)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                        .nIn(lstmLayerSize).nOut(nOut).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        if(savedNetwork.exists()) {
            net = ModelSerializer.restoreMultiLayerNetwork(savedNetwork, true);
        }

        //Print the  number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for( int i=0; i<layers.length; i++ ){
            int nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        //Do training, and then generate and print samples from network
        int miniBatchNumber = 0;

        for( int i=0; i<numEpochs; i++ ){
            long start = System.currentTimeMillis();
            System.out.print("Epoch: " + i);
            while(iter.hasNext()){
                DataSet ds = iter.next();
                net.fit(ds);
            }
            long now = System.currentTimeMillis();
            double seconds = (now - start)/1000d;
            System.out.println("  " + seconds + " secs");

            if(i % 20 == 0) {
                generateSample(nSamplesToGenerate, nCharactersToSample, generationInitialization, rng, iter, net);
            }

            iter.reset();	//Reset iterator for another epoch
        }

        generateSample(nSamplesToGenerate, nCharactersToSample, generationInitialization, rng, iter, net);

        ModelSerializer.writeModel(net, savedNetwork, true);
        System.out.println("\n\nExample complete");
    }

    private static void generateSample(int nSamplesToGenerate, int nCharactersToSample, String generationInitialization, Random rng, CharacterIterator iter, MultiLayerNetwork net) {
        System.out.println("--------------------");
        System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
        String[] samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate);
        for (int j = 0; j < samples.length; j++) {
            System.out.println("----- Sample " + j + " -----");
            System.out.println(samples[j]);

            ArrayList<String> sentences = extractSentences(samples[j]);

            for (int k = 0; k < sentences.size(); k++) {
                System.out.println("--Sentence " + k + ": " + sentences.get(k));
            }

            System.out.println();
        }
    }


    private static ArrayList<String> extractSentences(String dp) {
        ArrayList<String> sentences = new ArrayList<>();
        Pattern pattern = Pattern.compile("([?.]\\s+[A-Za-z*@])");
        Matcher m = pattern.matcher(dp);

        if(m.find()) {
            int startIndex = m.start() + 1;
            int endIndex = dp.length();
            while (m.find()) {
                endIndex = m.end() - 1;

                sentences.add(dp.substring(startIndex, endIndex));

                startIndex = m.start() + 1;
            }
        }

        return sentences;
    }

    public static CharacterIterator getTwitterIterator(int miniBatchSize, int sequenceLength) throws Exception {
        File f = new File("twitter.txt");

        Twitter twitter = TwitterFactory.getSingleton();
        System.out.println(twitter.getRateLimitStatus().get("/statuses/user_timeline"));
        Paging p = new Paging();
        p.setCount(200);
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i < 20; i++) {
            p.setPage(i);
            List<Status> statuses = twitter.getUserTimeline("kyru", p);

            for (Status s : statuses) {
                if (!s.getText().startsWith("RT") && !s.getText().startsWith("@")) {
                    sb.append(" " + s.getText());
                }

            }
        }

        FileWriter fw = new FileWriter(f);
        fw.write(sb.toString());
        fw.close();

        if(!f.exists()) throw new IOException("File does not exist: " + f.getAbsolutePath());	//Download problem?

        char[] validCharacters = CharacterIterator.getDefaultCharacterSet();	//Which characters are allowed? Others will be removed
        return new CharacterIterator(f.getAbsolutePath(), Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, new Random(12345));
    }

    private static String[] sampleWordsFromNetwork(String initialization, MultiLayerNetwork net,
                                                   StringWordIterator iter, Random rng, int wordsToSample, int numSamples ) {
        //Set up initialization. If no initialization: use a random character
        if( initialization == null ){
            initialization = iter.getRandomWord();
        }


        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
        String[] init = initialization.split(" ");
        for( int i=0; i<init.length; i++ ){
            int idx = iter.convertWordToIndex(init[i]);
            for( int j=0; j<numSamples; j++ ){
                initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization + " ");

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output

        for( int i=0; i<wordsToSample; i++ ){
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for( int s=0; s<numSamples; s++ ){
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);

                nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
                sb[s].append(iter.convertIndexToWord(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
                sb[s].append(" ");
            }

            output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
        return out;
    }
    /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initalization is used for all samples
     * @param initialization String, may be null. If null, select a random character as initialization for all samples
     * @param charactersToSample Number of characters to sample from network (excluding initialization)
     * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
     * @param iter CharacterIterator. Used for going from indexes back to characters
     */
    private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                        CharacterIterator iter, Random rng, int charactersToSample, int numSamples ){
        //Set up initialization. If no initialization: use a random character
        if( initialization == null ){
            initialization = String.valueOf(iter.getRandomCharacter());
        }

        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();
        for( int i=0; i<init.length; i++ ){
            int idx = iter.convertCharacterToIndex(init[i]);
            for( int j=0; j<numSamples; j++ ){
                initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output

        for( int i=0; i<charactersToSample; i++ ){
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for( int s=0; s<numSamples; s++ ){
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);

                nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
                sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
        return out;
    }

    /** Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    public static int sampleFromDistribution( double[] distribution, Random rng ){
        double d = 0.0;
        double sum = 0.0;
        for( int t=0; t<10; t++ ) {
            d = rng.nextDouble();
            sum = 0.0;
            for( int i=0; i<distribution.length; i++ ){
                sum += distribution[i];
                if( d <= sum ) return i;
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
        //Should be extremely unlikely to happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
    }

}
