package mqo_chimera.testcases;

import java.io.Serializable;
import java.util.*;

/**
 * Represents an MQO problem where the required connections between queries
 * comply with the Chimera graph structure implemented by D-Wave.
 * 
 * @author immanueltrummer
 *
 */
public class ChimeraMqoProblem implements Serializable {
	/**
	 * Used to verify the class version.
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * Indicates whether any of the query plans that are associated with qubits must be selected
	 * or if all qubits associated with the plans of a given query are allowed to be zero.
	 * The latter assignment symbolizes execution of a plan that does not interfer with any
	 * of the other plans.
	 */
	public final boolean allowIndependentProcessing;
	/**
	 * The number of queries for which we want to select plans.
	 */
	public final int nrQueries;
	/**
	 * The number of plans per query for which interaction between different queries (either
	 * the cost decreases for specific plan pairs if intermediate results can be shared or
	 * the cost increases if synchronization overhead is caused for instance). This does not
	 * count the possibility of processing a query independently from the others.
	 */
	public final int nrPlansPerQuery;
	/**
	 * Contains for each query and plan the associated cost (the associated cost for
	 * independent processing is automatically zero).
	 */
	public final double[][] planCost;
	/**
	 * Stores which specific plan pairs cause positive or negative interference.
	 */
	public final Map<PlanCoupling, Double> interactions;

	public final List<PlanCoupling> planCouplings;
	/**
	 * The optimal plan selections for each query according to the planted solution.
	 */
	public int[] plantedPlanSelections;
	/**
	 * Optimal qubit values of the planted solution.
	 */
	public boolean[] plantedQubitValues;
	/**
	 * This constructor initializes all variables describing the problem.
	 * 
	 * @param nrQueries						the number of queries for which to select plans
	 * @param nrPlansPerQuery				the number of dependent plans per query
	 * @param allowIndependentProcessing	whether it is allowed not so select any dependent plan
	 */
	public ChimeraMqoProblem(int nrQueries, int nrPlansPerQuery, boolean allowIndependentProcessing) {
		this.nrQueries = nrQueries;
		this.nrPlansPerQuery = nrPlansPerQuery;
		this.allowIndependentProcessing = allowIndependentProcessing;
		this.planCost = new double[nrQueries][nrPlansPerQuery];
		this.interactions = new HashMap<PlanCoupling, Double>();
		this.planCouplings = new ArrayList<>();
	}
	/**
	 * This constructor also initializes variables relating to a planted solution.
	 * 
	 * @param nrQueries						the number of queries for which to select plans
	 * @param nrPlansPerQuery				the number of dependent plans per query
	 * @param allowIndependentProcessing	whether it is allowed not so select any dependent plan
	 * @param plantedPlanSelections			the optimal plan index for each query (part of a planted solution)
	 * @param plantedQubitValues			the optimal value for each qubit (planted solution)
	 */
	public ChimeraMqoProblem(int nrQueries, int nrPlansPerQuery, boolean allowIndependentProcessing,
			int[] plantedPlanSelections, boolean[] plantedQubitValues) {
		this(nrQueries, nrPlansPerQuery, allowIndependentProcessing);
		this.plantedPlanSelections = plantedPlanSelections;
		this.plantedQubitValues = plantedQubitValues;
	}
	/**
	 * Copy constructor generating a deep copy of the given problem instance.
	 * 
	 * @param problem	a Chimera Mqo problem from which we want to obtain a deep copy
	 */
	public ChimeraMqoProblem(ChimeraMqoProblem problem) {
		this(problem.nrQueries, problem.nrPlansPerQuery, problem.allowIndependentProcessing, 
				Arrays.copyOf(problem.plantedPlanSelections, problem.plantedPlanSelections.length),
				Arrays.copyOf(problem.plantedQubitValues, problem.plantedQubitValues.length));
	}
	/**
	 * Returns maximal admissible plan index - if independent processing is enabled then
	 * the number of plans per query can be used as index, otherwise it is not admissible.
	 * 
	 * @return	the maximal index that can be used for a plan selection
	 */
	public int maxPlanIndex() {
		if (allowIndependentProcessing) {
			return nrPlansPerQuery;
		} else {
			return nrPlansPerQuery - 1;
		}
	}
	/**
	 * Adds information about an interference between two query plans, meaning that when selecting
	 * both plans then the cost either decreases or increases compared to the sum of the execution
	 * costs of the two plans. The function can be called multiple times for the same plan pair
	 * and the cost deltas accumulate as a sum.
	 * 
	 * @param query1		query of first interfering plan
	 * @param plan1			index of first interfering plan
	 * @param query2		query of second interfering plan
	 * @param plan2			index of second interfering plan
	 * @param costDelta		by how much the total cost changes in case that both plans are selected
	 */
	public void addInterference(int query1, int plan1, int query2, int plan2, double costDelta) {
		/*assert(query1 >= 0 && query1 < nrQueries);
		assert(query2 >= 0 && query2 < nrQueries);
		assert(plan1 >= 0 && ((allowIndependentProcessing && plan1 <= nrPlansPerQuery) || plan1 < nrPlansPerQuery));
		assert(plan2 >= 0 && ((allowIndependentProcessing && plan2 <= nrPlansPerQuery) || plan2 < nrPlansPerQuery));
		*/
		// generate corresponding coupling
		long startTimeCoupling = System.nanoTime();
		PlanCoupling coupling = new PlanCoupling(query1, plan1, query2, plan2);
		long endTimeCoupling = System.nanoTime();
		//System.out.println("Create coupling time: " + (endTimeCoupling-startTimeCoupling));
		/* TODO: Verify if this performance change is correct
		// make sure that this coupling is already stored
		if (!interactions.containsKey(coupling)) {
			interactions.put(coupling, 0.0);
		}
		// add cost delta
		//double oldCost = interactions.get(coupling);
		double newCost = oldCost + costDelta;
		// store
		interactions.put(coupling, newCost);
		*/
		long startTimeInteractions = System.nanoTime();
		interactions.put(coupling, costDelta);
		long endTimeInteractions = System.nanoTime();
		//System.out.println("Add interaction time: " + (endTimeInteractions-startTimeInteractions));
	}

	public void addInterferenceOld2(int query1, int plan1, int query2, int plan2, double costDelta) {
		PlanCoupling coupling = new PlanCoupling(query1, plan1, query2, plan2, costDelta);
		planCouplings.add(coupling);
	}

	public void addInterferenceOld(int query1, int plan1, int query2, int plan2, double costDelta) {
		assert(query1 >= 0 && query1 < nrQueries);
		assert(query2 >= 0 && query2 < nrQueries);
		assert(plan1 >= 0 && ((allowIndependentProcessing && plan1 <= nrPlansPerQuery) || plan1 < nrPlansPerQuery));
		assert(plan2 >= 0 && ((allowIndependentProcessing && plan2 <= nrPlansPerQuery) || plan2 < nrPlansPerQuery));

		// generate corresponding coupling
		PlanCoupling coupling = new PlanCoupling(query1, plan1, query2, plan2);
		// make sure that this coupling is already stored
		if (!interactions.containsKey(coupling)) {
			interactions.put(coupling, 0.0);
		}
		// add cost delta
		double oldCost = interactions.get(coupling);
		double newCost = oldCost + costDelta;
		// store
		interactions.put(coupling, newCost);

	}
	/**
	 * Returns information about an interference between two query plans.
	 * 
	 * @param query1		query of first interfering plan
	 * @param plan1			index of first interfering plan
	 * @param query2		query of second interfering plan
	 * @param plan2			index of second interfering plan
	 */
	public double getInterference(int query1, int plan1, int query2, int plan2) {
		// generate corresponding coupling
		PlanCoupling coupling = new PlanCoupling(query1, plan1, query2, plan2);
		// return cost delta zero if no interaction is known - otherwise return stored delta

		if (interactions.containsKey(coupling)) {
			return interactions.get(coupling);
		} else {
			return 0;
		}
	}

	public double getInterferenceOld(int query1, int plan1, int query2, int plan2) {
		// generate corresponding coupling
		PlanCoupling coupling = new PlanCoupling(query1, plan1, query2, plan2);
		// return cost delta zero if no interaction is known - otherwise return stored delta

		for (PlanCoupling planCoupling: planCouplings) {
			if (planCoupling.toString().equals(coupling.toString())) {
				return planCoupling.costSaving;
			}
		}
		return 0;
	}
	/**
	 * Returns execution cost of a single plan selection, returning value 0 for
	 * independent processing if this is allowed.
	 * 
	 * @param query	query index
	 * @param plan	plan index
	 * @return		execution cost of plan
	 */
	public double planExecutionCost(int query, int plan) {
		assert(plan >= 0);
		assert((plan <= nrPlansPerQuery && allowIndependentProcessing) || plan < nrPlansPerQuery);
		if (plan == nrPlansPerQuery) {
			return 0;
		} else {
			return planCost[query][plan];
		}
	}
	/**
	 * Returns the estimated total execution cost for the given plan selections,
	 * taking into account
	 * 
	 * @param planSelections	vector associating each query with the selected plan
	 * @return					total execution cost of all plans
	 */
	public double executionCost(int[] planSelections) {
		assert(planSelections.length == nrQueries);
		// Contains total execution cost in the end
		double totalCost = 0;
		// Add up execution cost of single selected plans
		for (int query=0; query<nrQueries; ++query) {
			int selectedPlan = planSelections[query];
			double executionCost = planExecutionCost(query,selectedPlan);
			totalCost += executionCost;
		}
		// Take plan interactions into account
		for (int query1=0; query1<nrQueries; ++query1) {
			int selectedPlan1 = planSelections[query1];
			for (int query2=query1+1; query2<nrQueries; ++query2) {
				int selectedPlan2 = planSelections[query2];
				double costDelta = getInterference(query1, selectedPlan1, query2, selectedPlan2);
				totalCost += costDelta;
			}
		}
		return totalCost;
	}
	/**
	 * Outputs this problem instance on the console.
	 */
	void toConsole() {
		System.out.println("nrQueries: " + nrQueries);
		System.out.println("nrPlansPerQuery: " + nrPlansPerQuery);
		for (int query=0; query<nrQueries; ++query) {
			System.out.println("Query " + query);
			System.out.println(Arrays.toString(planCost[query]));
		}
	}
}
