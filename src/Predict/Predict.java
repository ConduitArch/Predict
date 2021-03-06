package Predict;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.Console;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;


public class Predict {

	private static final int BELOW_THRESHOLD_GROUP = 0;
	private static final int ABOVE_THRESHOLD_GROUP = 1;
	private static final int POSITIVE_GROUP = 1;
	private static final int NEGATIVE_GROUP = 0;
	private static final long GROUP_IDENTIFIER = 0L;

	public static void main(String[] args) throws IOException {
        if (args.length < 2)
        {
            System.out.println("Usage: Predict.exe test_file model_file result_file");
            return;
        }

        String testFile = args[0];
        String modelFile = args[1];
        //String resultFile = args[2];

//        boolean hasLabel = true;
//        if (args.length == 4) hasLabel =  Boolean.parseBoolean(args[3]);

//        Map<Long, Double> model = loadModel(modelFile);
//        predictAccuracy(testFile, resultFile, model);
        System.out.println(getAccuracy(testFile, modelFile));
	}	

    private static Map<Long,Double> loadModel(String modelFile) throws IOException
    {
    	BufferedReader r = null;
        try {
        	r = new BufferedReader(new FileReader(modelFile));
			r.readLine(); // solverType
			r.readLine(); // classNumber
			r.readLine(); // labels
			r.readLine(); // featureAmount
			r.readLine(); // bias
			r.readLine();
			
			Map<Long,Double> modelFeatureWeights = new TreeMap<Long,Double>();
			String line;
			while ((line = r.readLine()) != null)
			{
				String[] fields = line.split("\t");
				long idx = Long.parseLong(fields[0]);
				double weight = Double.parseDouble(fields[1]);
				modelFeatureWeights.put(idx,weight);
			}
			return modelFeatureWeights;
		} finally {
			if (r != null) {
				r.close();
			}
		}
    }
    
    private static <T> double multiplyVectors(Map<T, Double> shortVector, Map<T, Double> longVector) {
    	if (shortVector.size() > longVector.size()) return multiplyVectors(longVector, shortVector);
    	
    	double sum = 0;
    	
    	for (T feature : shortVector.keySet()) {
    		if (longVector.containsKey(feature)) {
    			sum += shortVector.get(feature) * longVector.get(feature);
    		}
    	}
    	
    	return sum;
    }
    
    private static List<Map<Long, Double>> readFeatureVectors(String testFile) throws IOException {
    	List<Map<Long, Double>> result = new LinkedList<>();
    	BufferedReader r = null;
    	try {
    		r = new BufferedReader(new FileReader(testFile));
    		String line;
			while ((line = r.readLine()) != null) {
    		    Map<Long, Double> vector = new HashMap<>();
    		    String[] values = line.split(" ");
    		    for (String value : values) {
    		    	if (value == values[0]) {
    		    		vector.put(GROUP_IDENTIFIER, Double.parseDouble(value));
    		    		continue;
    		    	}
    		    	String[] fields = value.split(":");
    		    	vector.put(Long.parseLong(fields[0]), Double.parseDouble(fields[1]));
    		    }
    		    result.add(vector);
    		}
    	} finally {
    		if (r != null) {
    			r.close();
    		}
    	}
    	return result;
    }
    
    public static int[][] mapPrediction(List<Map<Long, Double>> featureVectors, Map<Long, Double> model) {
    	int[][] results = new int[2][2];
    	for (Map<Long, Double> vector : featureVectors) {
    		double score = multiplyVectors(vector, model);
    		results[score > 0 ? ABOVE_THRESHOLD_GROUP : BELOW_THRESHOLD_GROUP][vector.get(GROUP_IDENTIFIER) > 0 ? POSITIVE_GROUP : NEGATIVE_GROUP] += 1;
    	}
    	return results;
    }
    
    public static double getAccuracy(int[][] prediction) {
    	System.out.println("" + (prediction[BELOW_THRESHOLD_GROUP][NEGATIVE_GROUP] + prediction[ABOVE_THRESHOLD_GROUP][POSITIVE_GROUP]) + "/" + (prediction[BELOW_THRESHOLD_GROUP][NEGATIVE_GROUP] + prediction[ABOVE_THRESHOLD_GROUP][POSITIVE_GROUP] + 
    					prediction[BELOW_THRESHOLD_GROUP][POSITIVE_GROUP] + prediction[ABOVE_THRESHOLD_GROUP][NEGATIVE_GROUP]));
    	return (0.0 + prediction[BELOW_THRESHOLD_GROUP][NEGATIVE_GROUP] + prediction[ABOVE_THRESHOLD_GROUP][POSITIVE_GROUP]) / 
    			(prediction[BELOW_THRESHOLD_GROUP][NEGATIVE_GROUP] + prediction[ABOVE_THRESHOLD_GROUP][POSITIVE_GROUP] + 
    					prediction[BELOW_THRESHOLD_GROUP][POSITIVE_GROUP] + prediction[ABOVE_THRESHOLD_GROUP][NEGATIVE_GROUP]);
    }
    
    public static double getSpecificity(int[][] prediction) {
    	return 1.0 * prediction[BELOW_THRESHOLD_GROUP][NEGATIVE_GROUP] / (prediction[BELOW_THRESHOLD_GROUP][NEGATIVE_GROUP] + prediction[ABOVE_THRESHOLD_GROUP][NEGATIVE_GROUP]);
    }
    
    public static double getSensitivity(int[][] prediction) {
    	return 1.0 * prediction[ABOVE_THRESHOLD_GROUP][POSITIVE_GROUP] / (prediction[ABOVE_THRESHOLD_GROUP][POSITIVE_GROUP] + prediction[BELOW_THRESHOLD_GROUP][POSITIVE_GROUP]);
    }
    
    public static double getAccuracy(String testFile, String modelFile) throws IOException {
    	return getAccuracy(mapPrediction(readFeatureVectors(testFile), loadModel(modelFile)));
    }

    private static double predictAccuracy(String testFile, String resultFile, Map<Long, Double> model) throws IOException
    {
    	BufferedReader r = null;
    	BufferedWriter sw = null;
    	try {
    		r = new BufferedReader(new FileReader(testFile));   
    	    sw = new BufferedWriter (new FileWriter(resultFile));
    		int truePositives  = 0;  // positive instances that were classified as positive
	        int falseNegatives = 0;  // positive instances that were classified as positive
	        int trueNegatives  = 0;  // negative instances that were classified as negative
	        int falsePositives = 0;  // negative instances that were classified as negative
	
	
	        String line;
			while ((line = r.readLine()) != null)
	        {
	            String[] values = line.split(" ");
	            String label = "N/A";
	            label = values[0];
	
	            double sum = 0.0;
	            for (String value : values)
	            {
	                if (value.trim().length() == 0) continue; // The last line might be empty
	                String[] fields = value.split(":");
	                long featureIndex = Long.parseLong(fields[0]);
	                if (model.containsKey(featureIndex))
	                {
	                    double instanceFeatureValue = Double.parseDouble(fields[1]);
	                    sum += instanceFeatureValue * model.get(featureIndex);
	                }
	            }
	            if (label.equals("1") && sum > 0.0) ++truePositives;
	            if (label.equals("1") && sum <= 0.0) ++falseNegatives;
	            if (!label.equals("1") && sum > 0.0) ++falsePositives;
	            if (!label.equals("1") && sum <= 0.0) ++trueNegatives;
	
	            sw.append(label + "\t" + sum);
	            sw.newLine();
	        }

//	        double precision = truePositives / (truePositives + falsePositives);
//			double negativePredictiveValue = trueNegatives / (trueNegatives + falseNegatives);
//			double sensitivity = truePositives / (truePositives + falseNegatives);
//			double specificity = trueNegatives / (trueNegatives + falsePositives);
//			double accuracy = (truePositives + trueNegatives) / (truePositives + falseNegatives + trueNegatives + falsePositives);
//			double recall = sensitivity;
			// Double F1 = 2 * Precision * Recall / (Precision + Recall);
//			double accPer = accuracy*100;
//			System.out.println("Accuracy = "+accPer+"% (" +(truePositives+trueNegatives)+"/"+(truePositives+falseNegatives+trueNegatives + falsePositives)+")");
//			System.out.println("accuracy="+accuracy +
//					           ", Precision="+precision+
//					           ",Recall="+recall+
//					           ",NegativePredictiveValue="+negativePredictiveValue+
//					           ",Sensitivity="+sensitivity+
//					           ",Specifity="+specificity);
			return (truePositives + trueNegatives) / (truePositives + falseNegatives + trueNegatives + falsePositives);
		} finally {
			if (r != null) {
				r.close();
			}
			if (sw != null) {
				sw.close();
			}
		}
    	
    }

}
