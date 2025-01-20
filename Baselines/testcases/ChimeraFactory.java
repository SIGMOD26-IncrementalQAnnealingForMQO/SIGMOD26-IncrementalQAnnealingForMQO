package mqo_chimera.testcases;

import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

import mqo_chimera.benchmark.TestcaseClass;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.mapping.LogicalVariable;
import mqo_chimera.mapping.MappingType;
import mqo_chimera.mapping.VariableGroup;
import mqo_chimera.util.AmesUtil;
import mqo_chimera.util.Coupling;
import mqo_chimera.util.RandomUtil;

/**
 * Produces QUBOs representing MQO problems such that the QUBOs map well onto D-Wave's Chimera
 * graph structure. The resulting mapping is specialized for the D-Wave machine currently
 * available at NASA Ames research center whose characteristics are read in from files on disc.
 * 
 * @author immanueltrummer
 *
 */
public class ChimeraFactory {
	/**
	 * Returns the qubits that form the left colon of a unit cell.
	 * 
	 * @param cornerQubit	lowest-index qubit in the same unit cell
	 * @return				all working qubits in the left colon
	 */
	static Stack<Integer> getLeftColon(int cornerQubit) {
		Stack<Integer> resultStack = new Stack<Integer>();
		for (int i=0; i<4; ++i) {
			int qubit = cornerQubit + i;
			// TODO
			/*if (AmesUtil.amesQubits.contains(qubit)) {
				resultStack.push(qubit);
			}*/
			resultStack.push(qubit);
		}
		return resultStack;
	}
	/**
	 * Returns the qubits that form the right colon of a unit cell.
	 * 
	 * @param cornerQubit	lowest-index qubit in the same unit cell
	 * @return				all working qubits in the right colon
	 */
	static Stack<Integer> getRightColon(int cornerQubit) {
		Stack<Integer> resultStack = new Stack<Integer>();
		for (int i=4; i<8; ++i) {
			int qubit = cornerQubit + i;
			// TODO
			/*if (AmesUtil.amesQubits.contains(qubit)) {
				resultStack.push(qubit);
			}*/
			resultStack.push(qubit);
		}
		return resultStack;
	}
	/**
	 * Returns a list of groups of QUBO variables represented by qubits within the same
	 * unit cell such that the variables in each group are mutually connected to each
	 * other via couplings at Ames.
	 * 
	 * @param cornerQubit		lowest-index qubit in unit cell
	 * @param varsPerGroup		how many logical variables we need per group
	 * @return					list of fully connected groups
	 */
	public static List<VariableGroup> getConnectedQubits(int cornerQubit, int varsPerGroup) {
		//assert(AmesUtil.amesQubits.contains(cornerQubit));	// corner qubit might be broken
		assert(cornerQubit + 7 <= AmesUtil.highestQubitIndex);
		assert(varsPerGroup >= 1);
		assert(varsPerGroup <= 5);
		// There is one unit cell where internal connections are broken -
		// this causes problems with 5 query plans
		if (cornerQubit == 344 && varsPerGroup == 5) {
			return new LinkedList<VariableGroup>();
		}
		Stack<Integer> leftColon = getLeftColon(cornerQubit);
		Stack<Integer> rightColon = getRightColon(cornerQubit);
		List<VariableGroup> groupList = new LinkedList<VariableGroup>();
		if (varsPerGroup == 1) {
			// Singleton sets of mutually connected variables allow to represent variables
			// by only one qubit.
			Stack<Integer> allQubits = new Stack<Integer>();
			allQubits.addAll(leftColon);
			allQubits.addAll(rightColon);
			for (int qubit : allQubits) {
				VariableGroup newGroup = new VariableGroup();
				newGroup.addVariable(new LogicalVariable(qubit));
				groupList.add(newGroup);
			}
		} else {
			// We can represent the first two variables by one qubit while
			// the other variables must be represented by two qubits each.
			int requiredQubitPairs = varsPerGroup - 1;
			while (leftColon.size() >= requiredQubitPairs && rightColon.size() >= requiredQubitPairs) {
				VariableGroup newGroup = new VariableGroup();
				// Add first two variables, represented by two qubits in opposite columns
				LogicalVariable firstVariable = new LogicalVariable();
				LogicalVariable secondVariable = new LogicalVariable();
				firstVariable.addQubit(leftColon.pop());
				secondVariable.addQubit(rightColon.pop());
				newGroup.addVariable(firstVariable);
				newGroup.addVariable(secondVariable);
				// Add remaining variables, each is represented by a pair of qubits
				int remainingVars = varsPerGroup - 2;
				for (int varCtr=0; varCtr<remainingVars; ++varCtr) {
					LogicalVariable newVariable = new LogicalVariable();
					newVariable.addQubit(leftColon.pop());
					newVariable.addQubit(rightColon.pop());
					newGroup.addVariable(newVariable);
				}
				// Add variable group
				groupList.add(newGroup);
			}
		}
		// Make sure that all variable groups have the same size
		for (VariableGroup group : groupList) {
			assert(group.getVariables().size() == varsPerGroup);
		}
		// Make sure that all variables within each group are connected
		for (VariableGroup group : groupList) {
			for (LogicalVariable var1 : group.getVariables()) {
				for (LogicalVariable var2 : group.getVariables()) {
					if (var1 != var2) {
						assert(AmesUtil.amesConnected(var1, var2)) :
							"Vars not connected -  " +
							"var 1 qubits: " + var1.getQubits().toString() +
							"; var 2 qubits: " + var2.getQubits().toString();
					}
				}	
			}
		}
		// Make sure that the qubits reserved for different variables do not overlap
		for (VariableGroup group : groupList) {
			for (LogicalVariable var1 : group.getVariables()) {
				for (LogicalVariable var2 : group.getVariables()) {
					if (var1 != var2) {
						AmesUtil.assertNoOverlap(var1.getQubits(), var2.getQubits());
					}
				}
			}
		}
		return groupList;
	}
	/**
	 * Assign each logical variable to a set of qubits on the D-Wave qubit matrix.
	 * 
	 * @param mapping			the mapping contains variable to qubit assignments
	 */
	static void assignVariables(ChimeraMqoMapping mapping) {
		// extract problem dimensions
		int nrQueries = mapping.problem.nrQueries;
		int nrPlansPerQuery = mapping.problem.nrPlansPerQuery;
		// allocate space for variables
		mapping.planVars = new LogicalVariable[nrQueries][nrPlansPerQuery];
		// obtain enough connected variable groups - we need one variable group per query
		List<VariableGroup> varGroups = new LinkedList<VariableGroup>();
		int cornerQubit = 0;
		while (varGroups.size() < nrQueries) {
			varGroups.addAll(getConnectedQubits(cornerQubit, nrPlansPerQuery));
			cornerQubit += 8;
		}
		// map one plan variable after the other to qubits
		for (int query=0; query<nrQueries; ++query) {
			VariableGroup group = varGroups.get(query);
			for (int plan=0; plan<nrPlansPerQuery; ++plan) {
				mapping.planVars[query][plan] = group.getVariables().get(plan);
			}
		}
		// make sure that the qubit assignments for different variables do not overlap
		assert(!mapping.hasOverlap());
	}
	/**
	 * Given a mapping from logical variables to qubits, create a mapping from qubits
	 * to the logical variables (more precisely: map from each qubit to the query
	 * whose plans it represents and to the plan index).
	 * 
	 * @param mapping	holds mapping from variables to qubits 
	 * 					will hold mapping from qubits to queries and plans
	 */
	static void markQubits(ChimeraMqoMapping mapping) {
		// extract problem dimensions
		int nrQueries = mapping.problem.nrQueries;
		int nrPlansPerQuery = mapping.problem.nrPlansPerQuery;
		// allocate and initialize
		mapping.associatedQuery = new int[1152];
		mapping.associatedPlan = new int[1152];
		Arrays.fill(mapping.associatedQuery, -1);
		Arrays.fill(mapping.associatedPlan, -1);
		// associate queries and plans
		for (int query=0; query<nrQueries; ++query) {
			for (int plan=0; plan<nrPlansPerQuery; ++plan) {
				LogicalVariable var = mapping.planVars[query][plan];
				for (int qubit : var.getQubits()) {
					mapping.associatedQuery[qubit] = query;
					mapping.associatedPlan[qubit] = plan;
				}
			}
		}
	}
	/**
	 * Add weights on and between qubits to make sure that the minimum energy level is reached
	 * once all qubits representing the same logical variable are assigned to the same value.
	 * 
	 * @param mapping			weights will be added to the mapping motivating assigning the same value to qubits representing same variable
	 * @param equalityScaling	constraint term will be scaled by that factor
	 */
	static void imposeEqualityConstraints(ChimeraMqoMapping mapping, double equalityScaling) {
		// extract problem dimensions
		int nrQueries = mapping.problem.nrQueries;
		int nrPlansPerQuery = mapping.problem.nrPlansPerQuery;
		// add equality constraints
		for (int query=0; query<nrQueries; ++query) {
			for (int plan=0; plan<nrPlansPerQuery; ++plan) {
				LogicalVariable var = mapping.planVars[query][plan];
				var.addEquality(mapping, equalityScaling);
			}
		}
	}
	/**
	 * Add weights making sure that the minimum energy level is reached once exactly one plan is selected.
	 * A Boolean flag indicates whether not selecting any of the potentially dependent plans constitutes a
	 * minimum-energy configuration as well.
	 * 
	 * @param mapping						weights will be added between qubits representing alternative plans for the same query
	 * @param selectionScaling				energy term representing selection constraint is scaled by that factor
	 * @param allowIndependentProcessing	is not selecting any of the modeled plans a valid selection too?
	 */
	static void imposeAtLeatOneSelection(ChimeraMqoMapping mapping, double selectionScaling) {
		// extract problem dimensions
		int nrQueries = mapping.problem.nrQueries;
		int nrPlansPerQuery = mapping.problem.nrPlansPerQuery;
		// add terms enforcing at least one selection if independent processing is disabled
			for (int query=0; query<nrQueries; ++query) {
				for (int plan=0; plan<nrPlansPerQuery; ++plan) {
					LogicalVariable planVar = mapping.planVars[query][plan];
					planVar.addWeight(mapping, -selectionScaling);
				}
			}
	}
	
	static void imposeAtMostOneSelection(ChimeraMqoMapping mapping, double selectionScaling) {
		// extract problem dimensions
		int nrQueries = mapping.problem.nrQueries;
		int nrPlansPerQuery = mapping.problem.nrPlansPerQuery;
		// add terms enforcing at most one selection
		for (int query=0; query<nrQueries; ++query) {
			for (int plan1=0; plan1<nrPlansPerQuery; ++plan1) {
				for (int plan2=plan1+1; plan2<nrPlansPerQuery; ++plan2) {
					LogicalVariable planVar1 = mapping.planVars[query][plan1];
					LogicalVariable planVar2 = mapping.planVars[query][plan2];
					planVar1.addWeight(mapping, planVar2, selectionScaling);
				}
			}
		}
	}
	/*
	static void imposeSelectionConstraints(ChimeraMqoMapping mapping, double selectionScaling, boolean allowIndependentProcessing) {
		// extract problem dimensions
		int nrQueries = mapping.problem.nrQueries;
		int nrPlansPerQuery = mapping.problem.nrPlansPerQuery;
		// add terms enforcing at least one selection if independent processing is disabled
		if (!allowIndependentProcessing) {
			for (int query=0; query<nrQueries; ++query) {
				for (int plan=0; plan<nrPlansPerQuery; ++plan) {
					LogicalVariable planVar = mapping.planVars[query][plan];
					planVar.addWeight(mapping, -selectionScaling);
				}
			}
		}
		// add terms enforcing at most one selection
		for (int query=0; query<nrQueries; ++query) {
			for (int plan1=0; plan1<nrPlansPerQuery; ++plan1) {
				for (int plan2=plan1+1; plan2<nrPlansPerQuery; ++plan2) {
					LogicalVariable planVar1 = mapping.planVars[query][plan1];
					LogicalVariable planVar2 = mapping.planVars[query][plan2];
					planVar1.addWeight(mapping, planVar2, 2 * selectionScaling);
				}
			}
		}
	}
	*/
	/**
	 * Select one random plan for each query using a uniform random distribution.
	 * 
	 * @param problem						we extract the number of queries and plans from here
	 * @param allowIndependentProcessing	whether plan index nrPlansPerQuery is admissible as selection
	 * @return								a vector containing for each query the optimal plan index
	 * 										(index nrPlansPerQuery represents independent processing from other queries,
	 * 										neither sharing intermediate results nor causing synchronization overhead)
	 */
	public static int[] pickRandomPlans(ChimeraMqoProblem problem, boolean allowIndependentProcessing) {
		// extract problem dimensions
		int nrQueries = problem.nrQueries;
		int nrPlansPerQuery = problem.nrPlansPerQuery;
		// select random plans
		int[] randomPlans = new int[nrQueries];
		for (int query=0; query<nrQueries; ++query) {
			int randomPlan = allowIndependentProcessing ? 
					RandomUtil.uniformInt(0, nrPlansPerQuery) :
						RandomUtil.uniformInt(0, nrPlansPerQuery-1);
			randomPlans[query] = randomPlan;
		}
		return randomPlans;
	}
	/**
	 * Given an optimal plan for each query, infer the optiaml values for the qubits.
	 * 
	 * @param optimalPlans		vector containing the optimal plan index for each query
	 * @param mapping			maps logical variables to qubits
	 * @return					Boolean vector representing the optimal value for each qubit
	 */
	static boolean[] inferOptimalQubitValues(int[] optimalPlans, ChimeraMqoMapping mapping) {
		// extract problem dimensions
		int nrQueries = mapping.problem.nrQueries;
		int nrPlansPerQuery = mapping.problem.nrPlansPerQuery;
		int nrQubits = mapping.nrQubits;
		// infer qubit values
		boolean[] optimalQubitValues = new boolean[nrQubits];
		for (int query=0; query<nrQueries; ++query) {
			int optimalPlan = optimalPlans[query];
			// The last plan index with value nrPlansPerQuery means independent processing;
			// this possibility is modeled by setting all qubits associated with the query
			// to 0. In the latter case no qubits are set to true, otherwise the qubits
			// associated with the selected query plan are set to true.
			if (optimalPlan < nrPlansPerQuery) {
				LogicalVariable optimalPlanVar = mapping.planVars[query][optimalPlan];
				for (int qubit : optimalPlanVar.getQubits()) {
					optimalQubitValues[qubit] = true;
				}				
			}
		}
		return optimalQubitValues;
	}
	/**
	 * Given a qubit associated with the plan of one specific query, randomly selects
	 * a qubit associated with another query that is directly connected to the first
	 * qubit in the Chimera graph. This function is used when constructing loops
	 * over qubits in the Chimera graph. Returns -1 if no such qubit exists.
	 * 
	 * @param qubit		a qubit associated with a specific query plan
	 * @param mapping	assigns queries and plans to qubits
	 * @return			the index of a qubit connected to the input qubit but representing a different query
	 */
	static int getConnectedOtherQueryQubit(int qubit, ChimeraMqoMapping mapping) {
		int startQuery = mapping.associatedQuery[qubit];
		// Obtain all Chimera graph couplings in random order
		List<Coupling> shuffledCouplings = new LinkedList<Coupling>();
		shuffledCouplings.addAll(AmesUtil.amesCouplings);
		Collections.shuffle(shuffledCouplings);
		Iterator<Coupling> shuffledCouplingsIter = shuffledCouplings.iterator();
		// Follow first suitable coupling
		while (shuffledCouplingsIter.hasNext()) {
			Coupling coupling = shuffledCouplingsIter.next();
			int qubit1 = coupling.qubit1;
			int qubit2 = coupling.qubit2;
			// Check whether both coupled qubits are associated with query plans
			if (mapping.associatedQuery[qubit1] != -1 && mapping.associatedQuery[qubit2] != -1) {
				// Make sure that one of the two qubits is the one we want to connect to
				if (qubit1 == qubit || qubit2 == qubit) {
					int otherQubit = qubit1 == qubit ? qubit2 : qubit1;
					int targetQuery = mapping.associatedQuery[otherQubit];
					// Make sure that the connected qubit is not associated with a different plan for the same query.
					// The reason for this check is that our problem model assigns a fixed weight between qubits
					// representing different plans for the same query that is not problem instance dependent.
					if (targetQuery != startQuery) {
						return otherQubit;
					}
				}				
			}
		}
		// If we arrive here then we tried all couplings without success
		return -1;
	}
	/**
	 * Select a qubit representing a query plan with uniform random distribution
	 * over queries, plans, and qubits.
	 * 
	 * @param mapping	maps query plans to qubit sets
	 * @return			a qubit index that is associated with a query plan
	 */
	static int getRandomPlanQubit(ChimeraMqoMapping mapping) {
		// extract problem dimensions
		int nrQueries = mapping.problem.nrQueries;
		int nrPlans = mapping.problem.nrPlansPerQuery;
		// randomly select query and plan
		int query = RandomUtil.uniformInt(0, nrQueries-1);
		int plan = RandomUtil.uniformInt(0, nrPlans-1);
		LogicalVariable var = mapping.planVars[query][plan];
		// randomly select a qubit among the ones representing selected query plan
		List<Integer> varQubits = new LinkedList<Integer>(var.getQubits());
		int nrQubits = varQubits.size();
		int qubitIndex = RandomUtil.uniformInt(0, nrQubits-1);
		return varQubits.get(qubitIndex);
	}
	/**
	 * Get a loop of connected qubits in the D-Wave Ames Chimera graph such that each qubit
	 * represents a query plan and consecutive qubits are associated with
	 * different queries.
	 * 
	 * @param mapping	maps qubits to query plans
	 * @return			a sequence of qubit indices representing a loop in the Chimera graph
	 */
	static List<Integer> getAlternatingQueryLoop(ChimeraMqoMapping mapping) {
		// this will become the result of the function
		List<Integer> loop = new LinkedList<Integer>();
		// select connected start qubit
		int startQubit;
		do {
			startQubit = getRandomPlanQubit(mapping);
		} while (getConnectedOtherQueryQubit(startQubit, mapping) == -1);
		// Perform random walk until a loop is discovered
		int nextQubit = startQubit;
		// While loop not closed
		while (!loop.contains(nextQubit)) {
			loop.add(nextQubit);
			nextQubit = getConnectedOtherQueryQubit(nextQubit, mapping);
		}
		loop.add(nextQubit);
		// Cut path before crossing
		int crossingIndex = loop.indexOf(nextQubit);
		int loopLength = loop.size();
		return loop.subList(crossingIndex, loopLength);
	}
	/**
	 * Given two qubits and the Boolean values they are assigned to in the optimal solution,
	 * add weights on the qubit matrix that motivate this assignment.
	 * 
	 * @param qubit1		first qubit index
	 * @param qubit2		second qubit index
	 * @param qubit1Val		optimal value for first qubit
	 * @param qubit2Val		optimal value for second qubit
	 * @param mapping		qubit matrix on which to add weights
	 * @param scaling		multiply added weights by this factor
	 */
	static void addMotivatingWeights(int qubit1, int qubit2, boolean qubit1Val, boolean qubit2Val, ChimeraMqoMapping mapping, double scaling) {
		if (qubit1Val == qubit2Val) {
			// this term becomes minimal (value 0) if both qubits have the same value
			mapping.addWeight(qubit1, qubit2, -2 * scaling);
			mapping.addWeight(qubit1, qubit1, scaling);
			mapping.addWeight(qubit2, qubit2, scaling);
		} else {
			// this term becomes minimal (value -scaling) if both qubits have different values
			mapping.addWeight(qubit1, qubit2, 2 * scaling);
			mapping.addWeight(qubit1, qubit1, -scaling);
			mapping.addWeight(qubit2, qubit2, -scaling);
		}
	}
	/**
	 * After adding motivating weights, this function adapts the plan cost values
	 * of the associated MQO problem for consistency.
	 * 
	 * @param qubit1		first qubit index
	 * @param qubit2		second qubit index
	 * @param qubit1Val		optimal value for first qubit
	 * @param qubit2Val		optimal value for second qubit
	 * @param mapping		qubit matrix on which to add weights
	 * @param scaling		multiply added weights by this factor
	 */
	static void addMotivatingCost(int qubit1, int qubit2, boolean qubit1Val, boolean qubit2Val, ChimeraMqoMapping mapping, double scaling) {
		// Map from qubits to queries and plans
		int query1 = mapping.associatedQuery[qubit1];
		int query2 = mapping.associatedQuery[qubit2];
		int plan1 = mapping.associatedPlan[qubit1];
		int plan2 = mapping.associatedPlan[qubit2];
		// Extract problem to modify
		ChimeraMqoProblem problem = mapping.problem;
		// Distinguish whether equality or inequalty was motivated
		if (qubit1Val == qubit2Val) {
			problem.planCost[query1][plan1] += scaling;
			problem.planCost[query2][plan2] += scaling;
			problem.addInterference(query1, plan1, query2, plan2, -2 * scaling);
		} else {
			problem.planCost[query1][plan1] -= scaling;
			problem.planCost[query2][plan2] -= scaling;
			problem.addInterference(query1, plan1, query2, plan2, 2 * scaling);
		}
	}
	/**
	 * Given two qubits and the Boolean values they are assigned to in the optimal solution,
	 * add weights on the qubit matrix that *DE*-motivate this assignment.
	 * 
	 * @param qubit1		first qubit index
	 * @param qubit2		second qubit index
	 * @param qubit1Val		optimal value for first qubit
	 * @param qubit2Val		optimal value for second qubit
	 * @param mapping		qubit matrix on which to add weights
	 * @param scaling		multiply added weights by this factor
	 */
	static void addDeMotivatingWeights(int qubit1, int qubit2, boolean qubit1Val, boolean qubit2Val, ChimeraMqoMapping mapping, double scaling) {
		if (qubit1Val != qubit2Val) {
			// this term becomes minimal (value 0) if both qubits have the same value
			mapping.addWeight(qubit1, qubit2, -2 * scaling);
			mapping.addWeight(qubit1, qubit1, scaling);
			mapping.addWeight(qubit2, qubit2, scaling);
		} else {
			// this term becomes minimal (value -scaling) if both qubits have different values
			mapping.addWeight(qubit1, qubit2, 2 * scaling);
			mapping.addWeight(qubit1, qubit1, -scaling);
			mapping.addWeight(qubit2, qubit2, -scaling);
		}
	}
	/**
	 * After adding *DE*-motivating weights, this function adapts the plan cost values
	 * of the associated MQO problem for consistency.
	 * 
	 * @param qubit1		first qubit index
	 * @param qubit2		second qubit index
	 * @param qubit1Val		optimal value for first qubit
	 * @param qubit2Val		optimal value for second qubit
	 * @param mapping		qubit matrix on which to add weights
	 * @param scaling		multiply added weights by this factor
	 */
	static void addDeMotivatingCost(int qubit1, int qubit2, boolean qubit1Val, boolean qubit2Val, ChimeraMqoMapping mapping, double scaling) {
		// Map from qubits to queries and plans
		int query1 = mapping.associatedQuery[qubit1];
		int query2 = mapping.associatedQuery[qubit2];
		int plan1 = mapping.associatedPlan[qubit1];
		int plan2 = mapping.associatedPlan[qubit2];
		// Extract problem to modify
		ChimeraMqoProblem problem = mapping.problem;
		// Distinguish whether equality or inequalty was motivated
		if (qubit1Val != qubit2Val) {
			problem.planCost[query1][plan1] += scaling;
			problem.planCost[query2][plan2] += scaling;
			problem.addInterference(query1, plan1, query2, plan2, -2 * scaling);
		} else {
			problem.planCost[query1][plan1] -= scaling;
			problem.planCost[query2][plan2] -= scaling;
			problem.addInterference(query1, plan1, query2, plan2, 2 * scaling);
		}
	}
	/**
	 * Checks whether a given loop is admissible, meaning that we can add
	 * weights along it. A loop is admissible if the weights on the qubits
	 * and couplings covered by the loop are sufficiently far from their
	 * boundaries. We require at least the specified margin on single
	 * qubits and the double for couplings.
	 * 
	 * @param loop		sequence of connected qubits
	 * @param mapping	weighted qubit matrix
	 * @param margin	current weights must be so far from the boundaries for an admissible loop
	 * @return			true if all qubits and couplings used by the loop have no extreme weights
	 */
	/*
	static boolean admissibleLoop(List<Integer> loop, ChimeraMqoMapping mapping, double margin) {
		assert(margin >= 0 && margin < 1);
		// Check all qubits in the loop
		int loopLength = loop.size();
		for (int i=0; i<loopLength-1; ++i) {
			int qubit1 = loop.get(i);
			int qubit2 = loop.get(i+1);
			if (mapping.getWeight(qubit1) > 1.0 - margin || mapping.getWeight(qubit1) < - (1.0 - margin) ) {
				return false;
			}
			if (mapping.getWeight(qubit2) > 1.0 - margin || mapping.getWeight(qubit2) < - (1.0 - margin)) {
				return false;
			}
			if (mapping.getConnectionWeight(qubit1, qubit2) > 1.0 - 2 * margin || 
					mapping.getConnectionWeight(qubit1, qubit2) < - (1.0 - 2 * margin)) {
				return false;
			}
		}
		return true;
	}
	*/
	/**
	 * Randomly add weights on the qubit matrix in order to create a difficult QUBO optimization problem 
	 * that represents an instance of a MQO problem. Difficult QUBO problems are generated using a
	 * similar method as the one described in "Probing for quantum speedup in spin glass problems 
	 * with planted solutions" by Hen et al.: we randomly generate qubit loops in the Chimera graph
	 * and add weights along them that motivate choosing the previously determined optimal solution
	 * ("planted solution").
	 * 
	 * @param mapping				weighted qubit matrix - will be changed by invocation of this function
	 * @param optimalQubitValues	describes the optimal solution to be planted
	 * @param nrLoops				how many loops to generate
	 * @param minLoopLength			minimum length of those loops
	 * @param weightLevels			choose weight to add on loops with uniform random distribution among that many levels
	 */
	static void addWeights(ChimeraMqoMapping mapping, boolean[] optimalQubitValues, int nrLoops, int minLoopLength, int weightLevels) {
		// Add a certain number of loops
		for (int loopCtr=0; loopCtr<nrLoops; ++loopCtr) {
			if (loopCtr % 10 == 0) {
				System.out.println("Generated " + loopCtr + " loops");				
			}
			boolean loopAdded = false;
			while (!loopAdded) {
				// Generate loop of sufficient length
				List<Integer> loop;
				do {
					loop = getAlternatingQueryLoop(mapping);
				} while (loop.size() < minLoopLength);
				// Make sure that we obtained a loop
				int loopLength = loop.size();
				assert(loop.get(0).equals(loop.get(loopLength-1))) : loop.toString();
				// Add weights along the loop motivating the planted solution
				double scaling = RandomUtil.uniformInt(1, weightLevels) * 0.25/(double)weightLevels;
				for (int i=0; i<loopLength-1; ++i) {
					int qubit1 = loop.get(i);
					int qubit2 = loop.get(i+1);
					boolean qubit1OptimalValue = optimalQubitValues[qubit1];
					boolean qubit2OptimalValue = optimalQubitValues[qubit2];
					addMotivatingWeights(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);
					addMotivatingCost(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);
				}
				// Flip one randomly selected connection to create frustrated system
				int randomLoopStep = RandomUtil.uniformInt(0, loopLength-2);
				{
					int qubit1 = loop.get(randomLoopStep);
					int qubit2 = loop.get(randomLoopStep+1);
					boolean qubit1OptimalValue = optimalQubitValues[qubit1];
					boolean qubit2OptimalValue = optimalQubitValues[qubit2];
					addDeMotivatingWeights(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);
					addDeMotivatingWeights(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);
					addDeMotivatingCost(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);
					addDeMotivatingCost(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);					
				}
				// Check if resulting mapping is within admissible weight range
				if (mapping.toIsing().respectsIsingRanges()) {
					loopAdded = true;
				} else {
					// Restore old values 
					for (int i=0; i<loopLength-1; ++i) {
						int qubit1 = loop.get(i);
						int qubit2 = loop.get(i+1);
						boolean qubit1OptimalValue = optimalQubitValues[qubit1];
						boolean qubit2OptimalValue = optimalQubitValues[qubit2];
						addDeMotivatingWeights(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);
						addDeMotivatingCost(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);
					}
					int qubit1 = loop.get(randomLoopStep);
					int qubit2 = loop.get(randomLoopStep+1);
					boolean qubit1OptimalValue = optimalQubitValues[qubit1];
					boolean qubit2OptimalValue = optimalQubitValues[qubit2];
					addMotivatingWeights(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);
					addMotivatingWeights(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);
					addMotivatingCost(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);
					addMotivatingCost(qubit1, qubit2, qubit1OptimalValue, qubit2OptimalValue, mapping, scaling);					
				}
			}
		}
	}
	/**
	 * Verifies whether all inter-cell connections (connections between qubits in the same unit cell) are ok.
	 * 
	 * @return	true if all inter-cell connections are ok
	 */
	public static boolean allInterCellConnectionsOk() {
		boolean ok = true;
		for (int cornerQubit=0; cornerQubit<1152; cornerQubit+=8) {
			Stack<Integer> leftColon = getLeftColon(cornerQubit);
			Stack<Integer> rightColon = getRightColon(cornerQubit);
			for (int leftQubit : leftColon) {
				for (int rightQubit : rightColon) {
					// TODO
					/*if (!AmesUtil.amesConnected(leftQubit, rightQubit)) {
						System.out.println("No connection between " + leftQubit + " and " + rightQubit);
						ok = false;
					}*/
				}
			}
		}
		return ok;
	}
	/**
	 * Randomly generates a mapping representing a binary Chimera-shaped MQO problem instance.
	 * This method uses a generation method that adds frustrated loops to the qubit matrix.
	 * 
	 * @param nrQueries						the number of queries that may share intermediate results
	 * @param nrPlansPerQuery				the number of alternative query plans per query
	 * @param nrLoops						so many frustrated loops are added to the problem
	 * @param minLoopLength					minimum loop length
	 * @param weightLevels					the weight to put on the loops is uniformly selected from that many levels
	 * @param allowIndependentProcessing	is not selecting any of the modeled plans a valid selection too?
	 * @return								a randomly generated multiple query optimization problem instance
	 */
	public static ChimeraMqoMapping produceLoopsTestcase(int nrQueries, int nrPlansPerQuery, 
			int nrLoops, int minLoopLength, int weightLevels, boolean allowIndependentProcessing) {
		// Currently only up to five plans per query supported
		assert(nrPlansPerQuery <= 5);
		assert(nrPlansPerQuery >= 1);
		// Check integrity of Ames machine
		allInterCellConnectionsOk();
		// Encapsulates weights of the created QUBO problem
		ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
		mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlansPerQuery, allowIndependentProcessing);
		// Assign variables to qubits on the matrix
		assignVariables(mapping);
		// Create index mapping qubit indices to query indices
		markQubits(mapping);
		// Make sure that qubits representing the same variable will obtain the same value assigned
		//imposeEqualityConstraints(mapping);
		// Make sure that not more than one plan is selected
		imposeAtMostOneSelection(mapping, 0.5);
		if (!allowIndependentProcessing) {
			imposeAtLeatOneSelection(mapping, 0.5);
		}
		// randomly choose optimal solution to plant: pick random plans
		int[] optimalPlans = pickRandomPlans(mapping.problem, allowIndependentProcessing);
		// translate optimal plans into optimal qubit values
		boolean[] optimalQubitValues = inferOptimalQubitValues(optimalPlans, mapping);
		// add weights making the previously determined solution optimal
		addWeights(mapping, optimalQubitValues, nrLoops, minLoopLength, weightLevels);
		// get energy value of planted solution
		double plantedEnergy = mapping.getEnergy(optimalQubitValues);
		// store information about planted solution
		mapping.problem.plantedPlanSelections = optimalPlans;
		mapping.problem.plantedQubitValues = optimalQubitValues;
		mapping.plantedEnergy = plantedEnergy;
		// Return mapping with associated problem - the cost of the associated problem are not updated!
		return mapping;
	}
	/**
	 * Produces a standard test case, consisting of a MQO problem and a corresponding QUBO mapping.
	 * 
	 * @param testClass	contains the number of queries and plans for the random test case
	 * @return			a new MQO test case
	 */
	public static ChimeraMqoMapping produceStandardTestcase(TestcaseClass testClass) {
		// Extract dimensions of the test case problem to generate
		int nrQueries = testClass.nrQueries;
		int nrPlans = testClass.nrPlans;
		boolean allowIndependentProcessing = testClass.allowIndependentProcessing;
		// Check integrity of Ames machine
		allInterCellConnectionsOk();
		// Encapsulates weights of the created QUBO problem
		ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
		mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, allowIndependentProcessing);
		// Assign variables to qubits on the matrix
		assignVariables(mapping);
		// Create index mapping qubit indices to query indices
		markQubits(mapping);
		
		double scaling = 0.125;
		// Iterate over couplings
		for (Coupling coupling : AmesUtil.amesCouplings) {
			int qubit1 = coupling.qubit1;
			int qubit2 = coupling.qubit2;
			int query1 = mapping.associatedQuery[qubit1];
			int query2 = mapping.associatedQuery[qubit2];
			int plan1 = mapping.associatedPlan[qubit1];
			int plan2 = mapping.associatedPlan[qubit2];
			// Check if coupling connects plans of two different queries
			if (query1 != query2 && query1 != -1 && query2 != -1) {
				
				// Each plan obtains additional cost 0.25 for each other plan
				// that it can share intermediate results with - the potential
				// cost savings by saving intermediate results are determined
				// randomly.
				//mapping.addWeight(qubit1, qubit1, scaling * 1);
				//mapping.addWeight(qubit2, qubit2, scaling * 1);
				//double savings = -0.5 + (-0.25 + RandomUtil.uniformInt(0, 2) * 0.25);
				//double savings = -0.5 + (-0.375 + RandomUtil.uniformInt(0, 2) * 0.375);
				//double savings = -2 + (-1 + RandomUtil.uniformInt(0, 2) * 1);
				double savings = -scaling * RandomUtil.uniformInt(1, 2);
				mapping.addWeight(qubit1, qubit2, savings);
				// Adapt MQO problem model accordingly
				//mapping.problem.planCost[query1][plan1] += scaling * 1;
				//mapping.problem.planCost[query2][plan2] += scaling * 1;
				mapping.problem.addInterference(query1, plan1, query2, plan2, scaling * savings);
			}
		} 
		int[] nrPlansToNrConnections = new int[] {-1, -1, 5, 4, 2, 2}; 
		int maxPlanConnections = nrPlansToNrConnections[nrPlans];
		double selectionScaling = 2 * scaling * (maxPlanConnections+1);
		// Make sure that at most one plan is selected
		// TODO: at least one plan?
		imposeAtMostOneSelection(mapping, selectionScaling);
		// smear
		for (int query=0; query<nrQueries; ++query) {
			for (int plan=0; plan<nrPlans; ++plan) {
				LogicalVariable planVar = mapping.planVars[query][plan];
				planVar.smear(mapping);
			}
		}
		// Make sure that qubits representing the same variable will obtain the same value assigned
		//double equalityScaling = (scaling) * (maxPlanConnections+2);
		double maxEqualityScaling = 0;
		for (int query=0; query<nrQueries; ++query) {
			for (int plan=0; plan<nrPlans; ++plan) {
				LogicalVariable var = mapping.planVars[query][plan];
				double equalityScaling = var.calculateEqualityScalingGeneric(mapping);
				maxEqualityScaling = Math.max(equalityScaling, maxEqualityScaling);
			}
		}
		System.out.println("Equality scaling: " + maxEqualityScaling);
		imposeEqualityConstraints(mapping, maxEqualityScaling);
		// Perform some consistency tests
		System.out.println("maxAbsDiagonalWeight: " + mapping.toIsing().maxAbsDiagonalWeight());
		System.out.println("maxAbsNonDiagonalWeight: " + mapping.toIsing().maxAbsNonDiagonalWeight());
		//assert(mapping.toIsing().respectsIsingRanges());
		return mapping;
	}
}
