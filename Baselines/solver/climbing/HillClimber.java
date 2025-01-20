package mqo_chimera.solver.climbing;

import java.util.Arrays;

import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.solver.Solver;
import mqo_chimera.testcases.ChimeraFactory;
import mqo_chimera.testcases.ChimeraMqoProblem;
import mqo_chimera.testcases.MqoSolution;

/**
 * Simple hill climbing heuristic for MQO.
 * 
 * @author immanueltrummer
 *
 */
public class HillClimber extends Solver {
	/**
	 * Try to reduce execution cost by changing one plan selection. 
	 * 
	 * @param problem		an MQO problem to select optimal plans for
	 * @param solution		stores for each query a plan selection, 
	 * 						contains improved solution after invocation
	 * @param startTime		start time of optimization in milliseconds
	 * @param timeoutMillis	number of milliseconds for timeout
	 * @return 				true if it was possible to improve the last solution
	 */
	static boolean improveSolution(ChimeraMqoProblem problem, int[] solution, long startTime, long timeoutMillis) {
		// Extract problem dimensions
		int nrQueries = problem.nrQueries;
		int maxPlanIndex = problem.maxPlanIndex();
		// Calculate current execution cost - we will try to improve it
		double curCost = problem.executionCost(solution);
		// Iterate over queries and try all possible plans for each
		for (int query=0; query<nrQueries; ++query) {
			// Store plan selection in current solution
			int curPlan = solution[query];
			for (int plan=0; plan<=maxPlanIndex; ++plan) {
				solution[query] = plan;
				double newCost = problem.executionCost(solution);
				if (newCost < curCost) {
					return true;
				}
			}
			// Restore previously selected plan
			solution[query] = curPlan;
			// Check for timeout
			if (System.currentTimeMillis() - startTime > timeoutMillis) {
				return false;
			}
		}
		// If we arrive here then the solution could not be improved
		return false;
	}
	/**
	 * Generates in each iteration a random solution (meaning a plan selection for each query)
	 * and improves it via hill climbing until the timeout is reached or the cost reaches
	 * the optimum (which is known in advance as we generate problems with planted solutions).
	 */
	@Override
	public MqoSolution solve(ChimeraMqoProblem problem) throws Exception {
		// Start timer
		long startTime = System.currentTimeMillis();
		// Initialize variables
		boolean allowIndependentProcessing = problem.allowIndependentProcessing;
		Arrays.fill(lastRunCheckpointCost, Double.POSITIVE_INFINITY);
		long elapsedMillis = 0;
		int[] solution;
		double cost;
		int[] bestSolution = null;
		double bestCost = Double.POSITIVE_INFINITY;
		// While optimal solution not found and time remaining
		do {
			// Generate solution randomly
			solution = ChimeraFactory.pickRandomPlans(problem, allowIndependentProcessing);
			// Find local optimum
			while (improveSolution(problem, solution, startTime, BenchmarkConfiguration.timeoutMillis)) {
				// Calculate locally optimal execution cost
				cost = problem.executionCost(solution);
				// Update best plan found so far
				if (cost < bestCost) {
					bestCost = cost;
					bestSolution = solution;
				}
				// Calculate elapsed time
				elapsedMillis = System.currentTimeMillis() - startTime;
				// Update cost statistics
				updateStats(cost, elapsedMillis);				
			};
		} while (elapsedMillis < BenchmarkConfiguration.timeoutMillis);
		// Generate solution for best plan selections found
		return new MqoSolution(problem, bestSolution);
	}
	@Override
	public String solverID() {
		return "CLIMB";
	}
	
}
