package mqo_chimera.solver;

import java.io.PrintWriter;

import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.testcases.ChimeraMqoProblem;
import mqo_chimera.testcases.MqoSolution;

/**
 * Represents a software solver for MQO problems.
 * 
 * @author immanueltrummer
 *
 */
public abstract class Solver {
	/**
	 * The cost of the best solution found during the last run until certain
	 * checkpoint times.
	 */
	public double[] lastRunCheckpointCost = new double[BenchmarkConfiguration.nrBenchmarkTimes];
	/**
	 * Updates cost statistics after a plan with a certain execution cost has been
	 * generated after a certain number of milliseconds since optimization start.
	 * 
	 * @param cost				the cost of a generated plan
	 * @param elapsedMillis		the number of milliseconds since optimization start
	 */
	protected void updateStats(double cost, long elapsedMillis) {
		for (int intervalCtr=0; intervalCtr<BenchmarkConfiguration.nrBenchmarkTimes; ++intervalCtr) {
			long intervalBound = BenchmarkConfiguration.benchmarkTimes[intervalCtr];
			if (elapsedMillis <= intervalBound) {
				lastRunCheckpointCost[intervalCtr] = Math.min(lastRunCheckpointCost[intervalCtr], cost);
			}
		}
	}
	/**
	 * Writes the optimization time-solution quality curve seen in the last run to a file on disc.
	 * 
	 * @param path	path to the file to create
	 */
	public void writeLastCurve(String path) throws Exception {
		// open file
		PrintWriter writer = new PrintWriter(path);
		// write header
		writer.println("elapsedMillis,bestCost");
		// write curve
		for (int intervalCtr=0; intervalCtr<BenchmarkConfiguration.nrBenchmarkTimes; ++intervalCtr) {
			long millis = BenchmarkConfiguration.benchmarkTimes[intervalCtr];
			double cost = lastRunCheckpointCost[intervalCtr];
			writer.println(millis + "," + cost);
		}
		// close file
		writer.close();
	}
	/**
	 * Generates solutions of increasing quality for a MQO problem and stores
	 * statistics about how solution quality (i.e., execution cost) evolves
	 * over time. 
	 * 
	 * @param problem	an MQO problem instance
	 * @return			the best solution that was found for the MQO problem instance
	 */
	public abstract MqoSolution solve(ChimeraMqoProblem problem) throws Exception;
	/**
	 * Returns a short solver ID describing the solver and its configuration - this
	 * ID will be inserted into result file names and must not contain any special
	 * characters. 
	 * @return
	 */
	public abstract String solverID();
}
