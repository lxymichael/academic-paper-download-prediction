package download_prediction;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.apporiented.algorithm.clustering.AverageLinkageStrategy;
import com.apporiented.algorithm.clustering.Cluster;
import com.apporiented.algorithm.clustering.ClusteringAlgorithm;
import com.apporiented.algorithm.clustering.DefaultClusteringAlgorithm;
import com.apporiented.algorithm.clustering.SingleLinkageStrategy;
import com.fastdtw.dtw.FastDTW;
import com.fastdtw.timeseries.TimeSeries;
import com.fastdtw.timeseries.TimeSeriesBase;
import com.fastdtw.timeseries.TimeSeriesItem;
import com.fastdtw.timeseries.TimeSeriesPoint;
import com.fastdtw.util.Distances;

/**
 * cluster users by DTW distance using hierarchical clustering (one link, nearest neighbor)
 * load time-series data, calculate distance, clustering, output different sizes of clusters to file
 *  *
 */
public class DTWUserCluster {

	/**
	 * @param user_stats_input input of user statics csv file, each line in format [user_index, feature_1, feature_2.....], one user takes several lines in sequential order
	 * @param num_users number of users
	 * @param user_dim number of fields in each line of user_stats_input file
	 * @param top_level_cluster_output output location for 1st level cluster
	 * @param second_level_cluster_output output location for 2nd level cluster
	 * @throws NumberFormatException
	 * @throws IOException
	 */
	public static void main(String user_stats_input, int num_users, int user_dim, String top_level_cluster_output, String second_level_cluster_output) throws NumberFormatException, IOException {
		
		double[][] distances = new double[num_users][num_users];
		ArrayList<TimeSeries> users = new ArrayList<TimeSeries> ();
		
		//1. read user data into distance matrix
		BufferedReader reader = new BufferedReader(new FileReader(user_stats_input));
		String line;
		String part[];	
		String prevUser = null;
		ArrayList<String> stats = new ArrayList<String>();
		while ((line = reader.readLine()) != null) {
			String user = line.split(",")[0];
			
			//add previous user's stats
			if(prevUser != null && !prevUser.equals(user)){		
				List<TimeSeriesItem> list = new ArrayList<TimeSeriesItem>();
				//iterate over session stats for a user
				for(int i = 0; i < stats.size(); i++){
					part = stats.get(i).split(",");
					double[]  data = new double[user_dim];		
					for(int j = 0; j < user_dim; j++)
						data[j] = Double.parseDouble(part[j + 1]);
					//add series
					TimeSeriesPoint tp = new TimeSeriesPoint (data);
					list.add(new TimeSeriesItem(i, tp));
				}
				
				TimeSeries ts  = new TimeSeriesBase(list);
				users.add(ts);
				stats.clear();
				prevUser = user;
				continue;				
			}
			//add records	
			stats.add(line);
			prevUser = user;			
		}
		reader.close();
		
		//2. add last user
		List<TimeSeriesItem> list = new ArrayList<TimeSeriesItem>();
		for(int i = 0; i < stats.size(); i++){
			part = stats.get(i).split(",");
			double[]  data = new double[user_dim];	
			for(int j = 0; j < user_dim; j++)
				data[j] = Double.parseDouble(part[j + 1]);
			//add series
			TimeSeriesPoint tp = new TimeSeriesPoint (data);
			list.add(new TimeSeriesItem(i, tp));
		}
		TimeSeries ts  = new TimeSeriesBase(list);
		users.add(ts);
		
		//3. calculate distance using dtw for distance matrix
		for(int i = 0; i < num_users; i++){
			for(int j = i; j < num_users; j++){
				double distance = FastDTW.compare(users.get(i), users.get(j), 5, Distances.EUCLIDEAN_DISTANCE).getDistance();
				distances[i][j] = distance;
				distances[j][i] = distance;
			}
			System.out.println(i + "");
		}	
		
		//4. Hierarchical clustering using average linkage
		String[] names = new String[num_users];
		for(int i = 0; i < num_users; i++)
			names[i] = i + "";
		ClusteringAlgorithm alg = new DefaultClusteringAlgorithm();
		Cluster cluster = alg.performClustering(distances, names, new AverageLinkageStrategy());
		
		//output clusters at different levels of hierarchy, level 1 and 2 in separate files,
		BufferedWriter bw = new BufferedWriter(new FileWriter(top_level_cluster_output));
		
		//output level 1 children
		List<Cluster> level1 = cluster.getChildren();
		for(Cluster child : level1){
			ArrayList<String> children = getChildren(child);
			for(String kid : children){
				bw.write(kid + ",");
			}
			bw.write("\n");
		}
		bw.close();
		
		//output level 2 clusters
		bw = new BufferedWriter(new FileWriter(second_level_cluster_output));
		for(Cluster child : level1){
			List<Cluster> level2 = child.getChildren();
			for(Cluster kid: level2){
				ArrayList<String> children = getChildren(kid);
				for(String temp : children){
					bw.write(temp + ",");
				}
				bw.write("\n");
			}
		}
		bw.close();
	}
	
	/**
	 * returns ALL children of a cluster
	 * @param cluster
	 * @return
	 */
	public static ArrayList<String> getChildren (Cluster cluster){
		ArrayList<String> temp = new ArrayList<String>();
		if(cluster.isLeaf())
			temp.add(cluster.getName());
		else{
			List<Cluster> list = cluster.getChildren();
			for(Cluster child : list){
				temp.addAll(getChildren(child));
			}
		}
		return temp;
	}
}
