package core;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.meta.Prediction;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * predict download time or number using LSTM
 *
 */
public class LSTMPredictor 
{
	
    private static final Logger LOGGER = LoggerFactory.getLogger(LSTMPredictor.class);
    
    /**
     * @param train_feature_folder location of folder for training set features
     * @param train_label_folder location of folder for training set labels
     * @param test_feature_folder location of folder for test set features
     * @param test_label_folder location of folder for test set labels
     * @param max_index equals to (training/testing instance number - 1)
     * @param ind_output location of individual outputs from prediction
     * @param numClasses number of classes to predict
     * @param miniBatchSize mini batch size
     * @param nEpochs number of epochs
     * @param lstmLayerSize size of LSTM layer
     * @param learning_rate learning rate
     * @param input_size corresponds to number of features in LSTM input
     * @throws IOException
     * @throws InterruptedException
     */
    public static void main(String train_feature_folder, String train_label_folder, String test_feature_folder, String test_label_folder, int max_index, String ind_output, int numClasses, int miniBatchSize, int nEpochs, int lstmLayerSize, double learning_rate, int input_size) throws IOException, InterruptedException
    {    	

        // ----- Load the training data -----
    	SequenceRecordReader featureReader = new CSVSequenceRecordReader(0, ",");
    	SequenceRecordReader labelReader = new CSVSequenceRecordReader(0, ",");
    	
        featureReader.initialize(new NumberedFileInputSplit(train_feature_folder + "/train_features_%d", 0, max_index));
        labelReader.initialize(new NumberedFileInputSplit(train_label_folder + "/train_labels_%d", 0, max_index));
        //For regression, numPossibleLabels is not used. Setting it to -1 here
        DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, miniBatchSize, numClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainIter);              //Collect training data statistics
        trainIter.reset();

        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
        trainIter.setPreProcessor(normalizer);
        
        // ----- Load the testing data -----
        SequenceRecordReader featureReader2 = new CSVSequenceRecordReader(0, ",");
    	SequenceRecordReader labelReader2 = new CSVSequenceRecordReader(0, ",");
    	featureReader2.initialize(new NumberedFileInputSplit(test_feature_folder + "/test_features_%d", 0, max_index));
        labelReader2.initialize(new NumberedFileInputSplit(test_label_folder + "/test_labels_%d", 0, max_index));
        DataSetIterator testIter = new SequenceRecordReaderDataSetIterator(featureReader2, labelReader2, miniBatchSize, numClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        testIter.setPreProcessor(normalizer);  

        // ----- Configure the network -----
        // for momentum, the updater is called NESTEROVS
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(140)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .learningRate(learning_rate)
            .list()
            .layer(0, new GravesLSTM.Builder().forgetGateBiasInit(1).activation(Activation.TANH).nIn(input_size).nOut(lstmLayerSize)
                .build())
            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX).nIn(lstmLayerSize).nOut(numClasses).build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(20));
        
        // ----- Train the network, evaluating the test set performance at each epoch -----       
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        double maxAccu = 0;
        double maxF = 0;
        double weightedF1 = 0;
        
        //run predictions
        ArrayList<Integer> predicted = null;
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIter);
            //Evaluate on the test set:
            Evaluation evaluation = net.evaluate(testIter);
     
            weightedF1 = evaluation.f1();
            LOGGER.info(String.format(str, i, evaluation.accuracy(), weightedF1));            
            double accuracy = evaluation.accuracy();
           
            testIter.reset();
            trainIter.reset();
            
            //get max metrics
            if(accuracy > maxAccu){
            	maxF = weightedF1;
            	maxAccu = accuracy;
            	predicted = getPredictions(net, testIter);
            }
            
            testIter.reset();
            
            //output evaluation metrics
            System.out.println(evaluation.stats());
            System.out.println("max performance: " + maxAccu + " " + maxF);
            
        }
        
        //output individual predictions for use in significance test
        BufferedWriter eval_writer = new BufferedWriter(new FileWriter(ind_output));
        for(int num: predicted){
        	eval_writer.write(num + "\n");
        	eval_writer.flush();
        }
        System.out.println(predicted);
        eval_writer.close();
    }

    /**
     * get a list of predictions by class label
     * @param net
     * @param testIter
     * @return
     */
    private static ArrayList<Integer> getPredictions(MultiLayerNetwork net, DataSetIterator testIter) {
    	int totalOutputExamples = 0;
    	ArrayList<Integer> predictions = new ArrayList<Integer>();
    	while(testIter.hasNext()){
        	//data in one epoch
            DataSet next = testIter.next();
            
            INDArray outputMask = next.getLabelsMaskArray();
            INDArray predict2 = net.output(next.getFeatures(),false,next.getFeaturesMaskArray(),next.getLabelsMaskArray());
            INDArray labels = next.getLabels();
            
            totalOutputExamples = outputMask.sumNumber().intValue();
            int outSize = labels.size(1);            
            INDArray predicted2d = Nd4j.create(totalOutputExamples, outSize);

            int rowCount = 0;
            //put results in the current epoch
            for (int ex = 0; ex < outputMask.size(0); ex++) {
                for (int t = 0; t < outputMask.size(1); t++) {
                    if (outputMask.getDouble(ex, t) == 0.0) 
                    	continue;

                    predicted2d.putRow(rowCount, predict2.get(NDArrayIndex.point(ex), NDArrayIndex.all(), NDArrayIndex.point(t)));
                    rowCount++;
                }
            }
            
            //add results to predictions
            for (int i = 0; i < totalOutputExamples; i++){
            	INDArray probabilities = predicted2d.getRow(i);
            	//determines the label based on probabilities
            	double max = (Double) probabilities.maxNumber();
            	for(int j = 0; j < probabilities.length(); j++){
            		if(probabilities.getDouble(j) == max){
            			predictions.add(j);
            			break;
            		}
            	}            	
            }
            
        }
        testIter.reset();
		return predictions;
	}

	/**
     * 
     * @param evaluation
     * @return weighted F1
     */
	private static double getWeightedF1(Evaluation evaluation) {
		double weightedF1 = 0;
    	int totalCount = 0;
    	
    	//get total instances
    	for(int i = 0; i < 5; i++){
    		totalCount += evaluation.classCount(i);
    	}
    	//get label
    	for(int i = 0; i < 5; i++){
    		int count = evaluation.classCount(i);
    		double f1 = evaluation.f1(i);
    		weightedF1 += f1 * (double) count/ (double) totalCount;
    	}
    	return weightedF1;
	}
}
