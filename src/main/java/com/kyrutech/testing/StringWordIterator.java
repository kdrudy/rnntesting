package com.kyrutech.testing;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;

/**
 * Created by Kyle Rudy on 3/11/2017.
 */
public class StringWordIterator implements DataSetIterator {

    private String[] validWords;
    //Maps each character to an index ind the input/output
    private Map<String,Integer> stringToIdxMap;
    //Array of all words in the file
    private String[] fileWords;
    //Length of each example/minibatch (number of words)
    private int exampleLength;
    //Size of each minibatch (number of examples)
    private int miniBatchSize;
    //Offsets for the start of each example
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    private Random rng;

    public StringWordIterator(String textFilePath, int miniBatchSize, int exampleLength, Random rng ) throws IOException {
        if( !new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);

        this.exampleLength = exampleLength;
        this.miniBatchSize = miniBatchSize;
        this.rng = rng;

        stringToIdxMap = new HashMap<>();
        List<String> lines = Files.readAllLines(new File(textFilePath).toPath());
        int maxSize = 0;
        for( String s : lines ) maxSize += s.split(" ").length;
        String[] words = new String[maxSize];
        //validWords = new String[maxSize];
        ArrayList<String> validWordsList = new ArrayList<>();
        int currIdx = 0;
        int validWordsIdx = 0;
        for( String s : lines ){
            String[] thisLine = s.split(" ");
            for (String aThisLine : thisLine) {
                if(!stringToIdxMap.containsKey(aThisLine)) {
                    stringToIdxMap.put(aThisLine, validWordsIdx);
                    validWordsList.add(validWordsIdx++, aThisLine);
                }
                words[currIdx++] = aThisLine;
            }
            //if(newLineValid) characters[currIdx++] = '\n';
        }

        validWords = validWordsList.toArray(new String[validWordsList.size()]);

        if( currIdx == words.length ){
            fileWords = words;
        } else {
            fileWords = Arrays.copyOfRange(words, 0, currIdx);
        }

        initializeOffsets();
    }

    private void initializeOffsets() {
        //This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (fileWords.length - 1) / exampleLength - 2;   //-2: for end index, and for partial example
        for (int i = 0; i < nMinibatchesPerEpoch; i++) {
            exampleStartOffsets.add(i * exampleLength);
        }
        Collections.shuffle(exampleStartOffsets, rng);
    }

    public String convertIndexToWord( int idx ){
        return validWords[idx];
    }

    public int convertWordToIndex( String c ){
        return stringToIdxMap.get(c);
    }

    public String getRandomWord(){
        return validWords[(int) (rng.nextDouble()*validWords.length)];
    }

    public int getRandomWordIdx() { return (int) rng.nextDouble()*validWords.length; }


    @Override
    public DataSet next(int num) {
        if( exampleStartOffsets.size() == 0 ) throw new NoSuchElementException();

        int currMinibatchSize = Math.min(num, exampleStartOffsets.size());
        INDArray input = Nd4j.create(new int[]{currMinibatchSize,validWords.length,exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize,validWords.length,exampleLength}, 'f');

        for( int i=0; i<currMinibatchSize; i++ ){
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            int currCharIdx = stringToIdxMap.get(fileWords[startIdx]);	//Current input
            int c=0;
            for( int j=startIdx+1; j<endIdx; j++, c++ ){
                int nextCharIdx = stringToIdxMap.get(fileWords[j]);		//Next character to predict
                input.putScalar(new int[]{i,currCharIdx,c}, 1.0);
                labels.putScalar(new int[]{i,nextCharIdx,c}, 1.0);
                currCharIdx = nextCharIdx;
            }
        }

        return new DataSet(input,labels);
    }

    @Override
    public int totalExamples() {
        return (fileWords.length-1) / miniBatchSize - 2;
    }

    @Override
    public int inputColumns() {
        return validWords.length;
    }

    @Override
    public int totalOutcomes() {
        return validWords.length;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        exampleStartOffsets.clear();
        initializeOffsets();
    }

    @Override
    public int batch() {
        return miniBatchSize;
    }

    @Override
    public int cursor() {
        return totalExamples() - exampleStartOffsets.size();
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public boolean hasNext() {
        return exampleStartOffsets.size() > 0;
    }

    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
