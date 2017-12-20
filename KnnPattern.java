import java.io.BufferedWriter;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.File;
import java.io.FileWriter;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


/**
* Written by: Andres Cremisini 12/7/2017
* Some comments by Oswaldo Artiles, 12/10/2017
* CAP5768: Introduction to Data Science
* Final Project: MapReduce Project
* Optical character recognition
* E). Map-Reduce K-Nearest Neighbor (k-NN)Classifier
* 
*******************************************
* TO RUN:
* 1) Move the files zip_1000.train, zip_3000.train, zip_5000.train, zip_7291.train and zip.test to your hdfs input folder.
* 2) hadoop jar {/local/dir/}KnnPattern.jar KnnPattern {/local/dir} {/hdfs/input/dir} {/hdfs/output/dir}
* 
* USAGE EXAMPLE:
* hadoop fs -rm -r /user
* hadoop fs -mkdir -p /user/artiles
* hadoop fs -mkdir /user/artiles/input
* 
* hadoop fs -copyFromLocal /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/zip_1000.train /user/artiles/input
* hadoop fs -copyFromLocal /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/zip_3000.train /user/artiles/input
* hadoop fs -copyFromLocal /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/zip_5000.train /user/artiles/input
* hadoop fs -copyFromLocal /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/zip_7291.train /user/artiles/input
* hadoop fs -copyFromLocal /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/zip.test /user/artiles/input
*
* hadoop jar /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/KnnPattern.jar KnnPattern /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN /user/artiles/input /user/artiles/output
* 
* DISCUSSION:
* This program creates  a k-NN classifier based on
* the MapReduce computation methods implemented in
* the open-source implementation Hadoop.

* The mapper class (KnnMap):
* 1. Computes  the standard Euclidean distance
* between each train point and all the 
* test points.
* 2. Maps the key/value (distance/num) pairs
* to a set of intermediate key/values (test_point_key/<distance, num>) 
* corresponding to each test point, that after 
* sorting and shuffling are the inputs of the 
* reducer class.

* The reducer class (KnnReducer):
* 1. Accepts the key/<distance, num> pairs from the mapper 
* 2. Finds the k-smallest distances
* 3. The k-smallest distances are used to classify
*  the test point using a voting scheme based
*  on the majority class of the neighbors.  
* 4. The majority class is determined by the maximum 
* number  of votes obtained by comparing each one of 
* the classes in the neighbor list with the class
* of each row in the neighbor list. If the confidence
* is below .90 (ie. the majority class has < 90% of
* the votes), the average distance of the runner_up
* class and the majority class scaled by their number of votes 
* are used to determine the winner.
* 
* The number of train points are: 1000, 3000, 5000, and 7291.
* For each of these train points, the classifier runs for 
* all the odd values of k from 1 to 25. 
*  
*
output:   For each set of train points, the value of k,
*         and the corresponding classifier error fraction.
*         The CPU time for each set of train points.
*         The CPU time for each value of k when the 
*         set of train points is 7291.
*/

public class KnnPattern
{	
	/*
	 * WritableComparable class for a paired Double and Double (representing distance and number_class)
	 */
	public static class DoubleTuple implements WritableComparable<DoubleTuple>
	{
		private Double distance = 0.0;
		private Double num = 0.0;

		public void set(Double key, Double val)
		{
			distance = key;
			num = val;
		}
		
		public Double getDistance()
		{
			return distance;
		}
		
		public Double getNum()
		{
			return num;
		}
		
		@Override
		public void readFields(DataInput in) throws IOException
		{
			distance = in.readDouble();
			num = in.readDouble();
		}
		
		@Override
		public void write(DataOutput out) throws IOException
		{
			out.writeDouble(distance);
			out.writeDouble(num);
		}
		
		@Override
		public int compareTo(DoubleTuple o)
		{
			return (this.num).compareTo(o.num);
		}
	}
	
	/*
	 * The mapper class accepts an object and text (row identifier and row contents) and outputs
	 * a Text (test vector identifier) and DoubleTuple (distance and number_class ie. <15.4, 9.0>)
	 * 
	 */
	public static class KnnMapper extends Mapper<Object, Text, Text, DoubleTuple>
	{
		DoubleTuple distanceAndNum = new DoubleTuple();
		//Map<test_vector_id, Map<dist_from_train_to_test, train_class>>
		HashMap<Integer, TreeMap<Double, Double>> test_maps = new HashMap<Integer, TreeMap<Double, Double>>();
		
		int K;
		final int TRAIN_ROWS = 7291;
        final int COLS = 257;
        final int TEST_ROWS = 2007;
		double [][] test_vector = new double[TEST_ROWS][COLS];
		
	    /**
	     * Computes the Euclidean distance between two vectors,  
	     * represented as 1D arrays.  
	     * Returns the distance.
	     * 
	     * @param v1 the first vector
	     * @param v2 the second vector
	     * @return the distance  
	     * 
	     */
	    public static double getEuclDistance (double [] v1, double [] v2)   
	    {
	        double dist_sqr = 0.0; 
	        int len = v1.length;
	    
	        for (int i = 1; i < len; i++)
	        {
	            dist_sqr = dist_sqr + Math.pow((v1[i]-v2[i]),2);
	        }   
	        return Math.sqrt (dist_sqr);
	    }

		@Override
		/*
		 * The setup() method is run once at the start of the mapper and is supplied with MapReduce's
		 * context object
		 * 
		 * @param context object used throughout program to access configuration details and write
		 * result of calculations
		 * 
		 * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException
		{
			//input file is not empty
			if (context.getCacheFiles() != null && context.getCacheFiles().length > 0)
			{
				//set K from main
				Configuration conf = context.getConfiguration();
				String[] k_ = conf.getStrings("k");
				K = Integer.parseInt(k_[0]);
				
				// Read parameter file using alias established in main(), parse it
				String knnParams = FileUtils.readFileToString(new File("./test_file"));
				List<String> vals = new ArrayList<String>(Arrays.asList(knnParams.split(" |\\\n")));
				int i = 0; int j = 0;
				for(String v : vals){
			    	test_vector[i][j] = Double.parseDouble(v);
			    	j++;
			    	if (j % COLS == 0){
			    		test_maps.put(i, new TreeMap<Double, Double>());
						i++;
						j = 0;
			    	}
				}
			}
		}
				
		@Override
		/*
		 * The map() method is run by MapReduce once for each row supplied as the input data
		 * 
		 * @param key A key corresponding to a training vector (belonging to some mapper). Since we are interested
		 * later in aggregating all distances to a given *test* vector, the keys for the training vector are not used.
		 * @param value A Text representation of a line from the training data containing the number_class and the 256
		 * grayscale values
		 * @param context context object described in setup
		 * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
		 */
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException
		{
			//Tokenize the input line (presented as 'value' by MapReduce) from the train file, 
			//place it in an array
			String rLine = value.toString();
			List<String> vals = new ArrayList<String>(Arrays.asList(rLine.split(" |\\\n")));
			int i = 0;
			double [] train_vector = new double[COLS];
			for (String v : vals){
				train_vector[i] = Double.parseDouble(v);
				i++;
			}
			
			//get the distance from the current training vector to *every* test point. store that value in 
			//the TreeMap corresponding to each test vector accessed in the loop.
			for(int j = 0; j < TEST_ROWS; j++){
				Double dist = getEuclDistance(train_vector, test_vector[j]);
				Double num = train_vector[0];
				test_maps.get(j).put(dist, num);
				
				// Only K distances are required, so if the TreeMap contains over K entries, remove the last one,
				// which is the highest distance number.
				if (test_maps.get(j).size() > K)
				{
					test_maps.get(j).remove(test_maps.get(j).lastKey());
				}
			}
		}

		@Override
		/*
		 * The cleanup() method is run once after map() has run for every row. So at this point, we have
		 * the distance from one training vector to every test vector, and we store that information 
		 * in the tree map belonging to each test vector. Each TreeMap is identified by the place where
		 * the test vector appears in the test file (ie {1,....,num_test_vectors})
		 * 
		 * @param context same as above
		 * 
		 * @see org.apache.hadoop.mapreduce.Mapper#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
		 */
		protected void cleanup(Context context) throws IOException, InterruptedException
		{		
			Text label = new Text();
			// Loop through the K key:values in the TreeMap
			for (int i = 0; i < TEST_ROWS; i++){
				for(Entry<Double, Double> entry : test_maps.get(i).entrySet())
				{
					  Double dist = entry.getKey();
					  Double num = entry.getValue();
					  //distanceAndModel is the instance of DoubleTuple declared earlier. here we
					  //change the members and pass-by-value to the context.
					  distanceAndNum.set(dist, num);
					  //key corresponding to the current test vector as accessed by the loop
					  label.set(Integer.toString(i));
					  // Write to context a Text as key (for the test vector) and DoubleTuple (ie. distance
					  // from current train vector) as value
					  context.write(label, distanceAndNum);
				}
			}

		}
	}

	/*
	 * The reducer class accepts the Text and DoubleTuple objects just supplied to context and
	 * outputs two Text objects, key (for a test vector) and prediction, for the final classification.
	 */
	public static class KnnReducer extends Reducer<Text, DoubleTuple, Text, Text>
	{
		TreeMap<Integer, Double> test_preds = new TreeMap<Integer, Double>();
		int K;
		
		@Override
		/*
		 * setup() again is run before the main reduce() method
		 * 
		 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException
		{
			if (context.getCacheFiles() != null && context.getCacheFiles().length > 0)
			{
				Configuration conf = context.getConfiguration();
				String[] k_ = conf.getStrings("k");
				K = Integer.parseInt(k_[0]);
			}
		}
		
		@Override
		/*
		 * The reduce() method accepts the objects the mapper wrote to context: a Text and a DoubleTuple
		 * 
		 * @param key A key corresponding to a test vector (its distance to all the training points have now been aggregated)
		 * @param values All of the distances to the key test vector
		 * @param context The same context object
		 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		public void reduce(Text key, Iterable<DoubleTuple> values, Context context) throws IOException, InterruptedException
		{
			TreeMap<Double, Double> KnnMap = new TreeMap<Double, Double>();
			
			// values are the K-nearest DoubleTuple objects which the mapper wrote to context
			for (DoubleTuple val : values)
			{
				Double trainDist = val.getDistance();
				Double trainNum = val.getNum();

				//leave only k
				KnnMap.put(trainDist, trainNum);
				if (KnnMap.size() > K)
				{
					KnnMap.remove(KnnMap.lastKey());
				}
			}	

			//count ocurrence of each class in neighbors
			Map<String, Integer> counts = new HashMap<String, Integer>();
			for(Double n : KnnMap.values()){
				String k = Integer.toString(n.intValue());
				if(counts.get(k) == null)
					counts.put(k, 1);
				else
					counts.put(k, counts.get(k) + 1);
			}

		    String most_common_neigh = null;
		    int max_freq = -1;
		    for(Map.Entry<String, Integer> entry: counts.entrySet())
		    {
		        if(entry.getValue() > max_freq)
		        {
		            most_common_neigh = entry.getKey();
		            max_freq = entry.getValue();
		        }
		    }
		    //using this when the confidence in the prediction isn't very high. essentially it allows the opportunity
		    //for closer yet less numerous neighbors (but only less numerous to 10% of K) to win the classification vote.
		    if (max_freq / K < .90){
		    	String runner_up_neigh = null;
		    	int runner_up_freq = -1;
		    	for(Map.Entry<String, Integer> entry: counts.entrySet()){
		    		if(entry.getKey().equals(most_common_neigh)){
		    			continue;
		    		}
		    		else{
		    			if(entry.getValue() > runner_up_freq)
				        {
				            runner_up_neigh = entry.getKey();
				            runner_up_freq = entry.getValue();
				        }
		    		}	
		    	}
		    	Double most_common_dist = 0.0;
		    	Double runner_up_dist = 0.0;
		    	for(Map.Entry<Double, Double> entry: KnnMap.entrySet()){
		    		if (Double.toString(entry.getValue()).equals(most_common_neigh)){
		    			most_common_dist += entry.getKey();
		    		}
		    		if(Double.toString(entry.getValue()).equals(runner_up_neigh)){
		    			runner_up_dist += entry.getKey();
		    		}
		    	}
		    	most_common_dist = most_common_dist / max_freq;
		    	runner_up_dist = runner_up_dist / runner_up_freq;
		    	if(runner_up_dist < most_common_dist && (max_freq - runner_up_freq)/K < .10 ){
		    		most_common_neigh = runner_up_neigh;
		    	}
		    }
			//add predictions for test vector in TreeMap that will be used to write to context in cleanup()   
			test_preds.put(Integer.parseInt(key.toString()), Double.parseDouble(most_common_neigh));

		}
		@Override
		/*
		 * The cleanup() method is run once after map() has run for every row
		 * 
		 * @param context Same as above
		 * 
		 * @see org.apache.hadoop.mapreduce.Reducer#cleanup(org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void cleanup(Context context) throws IOException, InterruptedException
		{
			Text label = new Text();

			for (int i = 0; i < test_preds.size(); i++){
				String test_vector_key = Integer.toString(i);
				Double pred = test_preds.get(i);
				
				label.set(test_vector_key);
				
				context.write(label, new Text(Double.toString(pred)));
			}

		}
	}
	
	/*
	 * helper class to write results to a file called knn_mapRed_results.txt in the local dir
	 */
	public static class Writer
	{
		/*
		 * @param path local dir
		 * @param line the text to write
		 */
		public static void writeLine(String path, String line)
		{
			try(FileWriter fw = new FileWriter
		    		(path + "/knn_mapRed_results.txt", true);
		    	BufferedWriter bw = new BufferedWriter(fw);
		    	PrintWriter out_ = new PrintWriter(bw))
		    {
		    	out_.println(line);
		    	
		    } catch(IOException e){
		    	System.out.println("Error in .txt IO");
		    }
		}
	}
	
	//TODO: get different dirs as parameters, to make grader's life easier 
	
	/*
	 * Main program to run: By calling MapReduce's 'job' API it configures and submits the MapReduce job.
	 * 
	 * args[0] local: path to the dir where you unzipped our project
	 * args[1] hdfs: path to your input directory in hdfs is (ie. where zip_x.train and zip.test are located)
	 * args[2] hdfs: path to output directory in hdfs (one output folder for every test will be written here)
	 * 
	 */
	public static void main(String[] args) throws Exception
	{
		int[] train_list = {1000, 3000, 5000, 7291};
		int[] k_list = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25};
		String line = "";
		
		for(int t : train_list){
			line = "train points = " + t + "\nk classifier error fraction CPU time (msec)";
			Writer.writeLine(args[0], line);
			
			double total_time = 0.0;
			for(int k_ : k_list){
				
				if (args.length != 3)
				{
					System.err.println("Usage: KnnPattern </path/to/local/dir> </path/to/hdfs/input_dir> </path/to/hdfs/output_dir>");
					System.exit(2);
				}
				long startTime = System.currentTimeMillis();
				
				// Create configuration, set current k
				Configuration conf = new Configuration();
				conf.set("k", Integer.toString(k_));
				
				// Create job
				Job job = Job.getInstance(conf, "Find K-Nearest Neighbor");
				job.setJarByClass(KnnPattern.class);
				// Set the third parameter when running the job to be the parameter file and give it an alias
				job.addCacheFile(new URI(args[1] + "/zip.test" + "#test_file")); // Parameter file containing test data
				
				//This is the file that is split and sent to the mappers. Since generally we can expect the training data
				//to be larger than the testing data, the training data is what we want to split and send to the mappers.
				//String p_ = "/user/acremisini/input/zip" + "_" + t + ".train";
				FileInputFormat.addInputPath(job, new Path(args[1] + "/zip_" + t + ".train"));
				
				// Setup MapReduce job
				job.setMapperClass(KnnMapper.class);
				job.setReducerClass(KnnReducer.class);
				job.setNumReduceTasks(1); // Only one reducer in this design

				// Specify key / value
				job.setMapOutputKeyClass(Text.class);
				job.setMapOutputValueClass(DoubleTuple.class);
				job.setOutputKeyClass(Text.class);
				job.setOutputValueClass(Text.class);
						
				// Input (the data file) and Output (the resulting classification)
				//String o_ = "/user/acremisini/test_out/output_" + t + "_" + k_;
				FileOutputFormat.setOutputPath(job, new Path(args[2] + "/" + t + "_" + k_));
				
				//if job is successful, get accuracy and write results
				if(job.waitForCompletion(true)){
					long stopTime = System.currentTimeMillis();
				    long elapsedTime = stopTime - startTime;
				    
				    //steps to get file with predictions
				    String hdfsuri = conf.get("fs.defaultFS");
					String path= args[2] + "/" +  t + "_" + k_ + "/";
				    String fileName="part-r-00000";
				
				    // Set FileSystem URI
				    conf.set("fs.defaultFS", hdfsuri);
				    // Because of Maven
				    conf.set("fs.hdfs.impl", org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());
				    conf.set("fs.file.impl", org.apache.hadoop.fs.LocalFileSystem.class.getName());
				    
				    //Get the filesystem - HDFS
				    FileSystem fs = FileSystem.get(conf);

					//Read file
				    //Create a path
			        Path hdfsreadpath = new Path(path + fileName);
			        //Init input stream
				    FSDataInputStream inputStream = fs.open(hdfsreadpath);
				    //Classical input stream usage
				    String out = IOUtils.toString(inputStream, "UTF-8");
				    
				    //get predictions, put them in a list
					List<String> preds = new ArrayList<String>(Arrays.asList(out.split("\n")));
					String test_classes = FileUtils.readFileToString(new File
							(args[0] + "/test_classes.txt"));
					
					//get anwsers, put them in a list
					List<String> test_list = new ArrayList<String>(Arrays.asList(test_classes.split("\n")));
					
					//calculate error
					double count = 0;
					int total = 0;
					for(String pr : preds){
						String p = pr.split("\t")[1];
						if(p.equals(test_list.get(total))){
							count++;
						}
						total++;
					}
					
					total_time += elapsedTime;
				    line = String.format("%d %15.2f %15d", k_, (1-count/total) * 100, elapsedTime);
				    inputStream.close();
				    fs.close();
				    
				    //write out results for this k
				    Writer.writeLine(args[0], line);
				}
				else{
					System.out.println("ERROR ON JOB " + t + " ," + k_);
				}
			}//end k loop
			//record time for this k
			Writer.writeLine(args[0], "\nCPU time of the MapReduce kNN algorithm = " + total_time + " miliseconds.\n");
		}//end train partition loop
	}//end main
}//end KnnPattern