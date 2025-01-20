package mqo_chimera.testcases;

/**
 * Represents the solution to a MQO problem instance.
 * 
 * @author immanueltrummer
 *
 */
public class MqoSolution {
	/**
	 * The multiple query optimization problem instance that this solution refers to.
	 */
	public final ChimeraMqoProblem problem;
	/**
	 * Contains for each query the index of the selected plan.
	 */
	public final int[] planSelections;
	/**
	 * The execution cost for the given plan selections.
	 */
	public final double executionCost;
	
	public MqoSolution(ChimeraMqoProblem problem, int[] planSelections) {
		this.problem = problem;
		this.planSelections = planSelections;
		this.executionCost = planSelections != null ? problem.executionCost(planSelections) : Double.POSITIVE_INFINITY;
		//this.executionCost = problem.executionCost(planSelections);
	}
	public void toConsole() {
		if (planSelections == null) {
			System.out.println("Empty solution");
		} else {
			System.out.println("Execution cost: " + executionCost);
			System.out.println("--- Plan Selections ---");
			for (int query = 0; query < problem.nrQueries; ++query) {
				System.out.println("Query " + query + " plan selection: \t" + planSelections[query]);
			}
		}
	}

}
