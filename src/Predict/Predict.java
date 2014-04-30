package Predict;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;


public class Predict {

	public static void main(String[] args) throws IOException {
        if (args.length < 3)
        {
            System.out.println("Usage: Predict.exe test_file model_file result_file");
            return;
        }

        String testFile = args[0];
        String modelFile = args[1];
        String resultFile = args[2];

        boolean hasLabel = true;
        if (args.length == 4) hasLabel =  Boolean.parseBoolean(args[3]);

        LoadModelNew(modelFile);
        PredictNew(testFile, resultFile,hasLabel);		
	}	

	static Map<Long,Double> modelFeaturesWeightsNew = null;
    private static void LoadModelNew(String modelFile)
    {
		long start = System.currentTimeMillis();
        try {
    		BufferedReader r = new BufferedReader(new FileReader(modelFile));   
			String solverType = r.readLine();
			String classNumber = r.readLine();
			String labels = r.readLine();
			String featureAmount = r.readLine();
			String bias = r.readLine();
			r.readLine();
			
			int featuresAmount = Integer.parseInt(featureAmount.replace("nr_feature", "").trim());
			++featuresAmount; //The feature program start with index 1 to supprt SVMLight format
			modelFeaturesWeightsNew = new TreeMap<Long,Double>();
			String line;
			while ((line = r.readLine()) != null)
			{
				String[] fields = line.split("\t");
				long idx = Long.parseLong(fields[0]);
				double weight = Double.parseDouble(fields[1]);
				modelFeaturesWeightsNew.put(idx,weight);
				}
			r.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        long end = System.currentTimeMillis();
        System.out.println("Loading model took " + (end-start) + " milliseconds.");
    }

    private static void PredictNew(String testFile, String resultFile, boolean hasLabel) throws IOException
    {
    	long start = System.currentTimeMillis();
    	BufferedReader r = new BufferedReader(new FileReader(testFile));   
    	BufferedWriter  sw = new BufferedWriter (new FileWriter(resultFile));
        int instanceAmount = 0;

        int true_positives  = 0;  // positive intances that were classified as positive
        int false_negatives = 0;  // positive intances that were classified as positive
        int true_negatives  = 0;  // negative intances that were classified as negative
        int false_positives = 0;  // negative intances that were classified as negative


        int true_positives2 = 0;  // positive intances that were classified as positive
        int false_negatives2 = 0;  // positive intances that were classified as positive
        int true_negatives2 = 0;  // negative intances that were classified as negative
        int false_positives2 = 0;  // negative intances that were classified as negative

		String line;
		while ((line = r.readLine()) != null)
        {
            ++instanceAmount;
            String[] values = line.split(" ");
            String label = "N/A";
            if (hasLabel) label = values[0];

            double sum = 0.0;
            int instanceFeaturesAmount = values.length - 1; // -1 is because the train/test creation function add extra space at the end 
            int i = 0;
            if (hasLabel) i = 1;
            for (; i < values.length; i++)
            {
            	String value = values[i];
                if (value.trim().length() == 0) continue; // The last line might be empty
                String[] fields = value.split(":");
                long featureIndex = Long.parseLong(fields[0]);
                if (modelFeaturesWeightsNew.containsKey(featureIndex))
                {
                    double instanceFeatureValue = Double.parseDouble(fields[1]);
                    sum += instanceFeatureValue * modelFeaturesWeightsNew.get(featureIndex);
                }
            }
            if (label.equals("1") && sum > 0.0) ++true_positives;
            if (label.equals("1") && sum <= 0.0) ++false_negatives;
            if (!label.equals("1") && sum > 0.0) ++false_positives;
            if (!label.equals("1") && sum <= 0.0) ++true_negatives;

            if (label == "1" && sum >= 0) ++true_positives2;
            if (label == "1" && sum < 0) ++false_negatives2;
            if (label != "1" && sum >= 0) ++false_positives2;
            if (label != "1" && sum < 0) ++true_negatives2;

            sw.append(label + "\t" + sum);
            sw.newLine();
        }
        r.close();
        sw.close();
        long end = System.currentTimeMillis();
        System.out.println("Prediction of " + instanceAmount + " took " + (end-start) + " milliseconds. (Average of " + (double)((end-start) / instanceAmount) + " milliseconds per instance).");
        if (hasLabel)
        {
        	Double Precision = (double)true_positives / (true_positives + false_positives);
            Double Negative_predictive_value = (double)true_negatives / (true_negatives + false_negatives);
            Double Sensitivity = (double)true_positives / (true_positives + false_negatives);
            Double Spcecifity = (double)true_negatives / (true_negatives + false_positives);
            Double Accuracy = (double)(true_positives + true_negatives) / (true_positives + false_negatives + true_negatives + false_positives);
            Double Recall = Sensitivity;
            Double F1 = 2 * Precision * Recall / (Precision + Recall);
            Double accPer = Accuracy*100;
            System.out.println("Accuracy = "+accPer+"% (" +(true_positives+true_negatives)+"/"+(true_positives+false_negatives+true_negatives + false_positives)+")");
            System.out.println("accuracy="+Accuracy +
            		           ", Precision="+Precision+
            		           ",Recall="+Recall+
            		           ",NegativePredictiveValue="+Negative_predictive_value+
            		           ",Sensitivity="+Sensitivity+
            		           ",Specifity="+Spcecifity);
        }
    }

}
