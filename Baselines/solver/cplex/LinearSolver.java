package mqo_chimera.solver.cplex;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;

import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.solver.Solver;
import mqo_chimera.testcases.ChimeraMqoProblem;
import mqo_chimera.testcases.MqoSolution;
import mqo_chimera.testcases.QuadraticMqoSolution;
import mqo_chimera.testcases.PlanCoupling;
import mqo_chimera.util.Coupling;
import mqo_chimera.util.GenericUtil;
import ilog.concert.IloException;
import ilog.concert.IloIntVar;
import ilog.concert.IloLinearIntExpr;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloNumExpr;
import ilog.cplex.IloCplex;
import ilog.cplex.IloCplex.Status;

/**
 * Solves Qubo and Ising problems by reformulating them as ILPs.
 * 
 * @author immanueltrummer
 *
 */
public class LinearSolver extends Solver {
	/**
	 * Contains after each run the number of elapsed milliseconds during solving.
	 */
	public long lastRunSolverMillis;
	/**
	 * Contains the optimal execution cost for the last solved problem instance.
	 */
	public double lastRunOptimalCost;
	/**
	 * Provides access to CPLEX solver. This reformulation was recommended in the publicatoin
	 * "A note on QUBO instances defined on Chimera graphs" by Dash, 2013.
	 */
	static public IloCplex cplex;
	/**
	 * The constructor initializes the CPLEX object; each invocation of the solve method clears the model.
	 * 
	 * @throws IloException
	 */
	public LinearSolver() throws IloException {		
		cplex = new IloCplex();
	}
	/**
	 * Add weights to energy formula that are set on single qubits (not between qubits).
	 * 
	 * @param mapping			qubit matrix with weights
	 * @param qubitVars			variables representing the state of single qubits
	 * @param energyFormula		the energy formula that CPLEX will minimize
	 * @throws IloException
	 */
	void treatQubitWeights(ChimeraMqoMapping mapping, IloIntVar[] qubitVars, IloLinearNumExpr energyFormula) throws IloException {
		int nrQubits = mapping.nrQubits;
		for (int qubit=0; qubit<nrQubits; ++qubit) {
			double weight = mapping.getWeight(qubit);
			if (weight != 0) {
				IloIntVar qubitVar = qubitVars[qubit];
				energyFormula.addTerm(weight, qubitVar);
			}
		}
	}
	/**
	 * Add auxiliary variables representing couplings, corresponding constraints making sure that the
	 * auxiliary variables are set to one if and only if both connected qubits are set to one, and
	 * add weights corresponding to couplings to energy formula.
	 * 
	 * @param mapping			qubit matrix with weights
	 * @param qubitVars			variables representing the state of single qubits
	 * @param couplingVars		variables representing interactions between qubit pairs
	 * @param energyFormula		the energy formula that CPLEX will minimize
	 * @param couplings			stores for each coupling variable the associated qubit pair
	 * @throws IloException
	 */
	void treatCouplingWeights(ChimeraMqoMapping mapping, IloIntVar[] qubitVars, IloIntVar[] couplingVars, 
			IloLinearNumExpr energyFormula, Coupling[] couplings) throws IloException {
		int nrQubits = mapping.nrQubits;
		int nextCouplingIndex = 0;
		for (int qubit1=0; qubit1<nrQubits; ++qubit1) {
			for (int qubit2=qubit1 + 1; qubit2<nrQubits; ++qubit2) {
				double weight = mapping.getConnectionWeight(qubit1, qubit2);
				if (weight != 0) {
					// associate coupling with connected qubits
					couplings[nextCouplingIndex] = new Coupling(qubit1, qubit2);
					// impose constraint on coupling variable
					IloIntVar couplingVar = couplingVars[nextCouplingIndex];
					cplex.addGe(qubitVars[qubit1], couplingVar);
					cplex.addGe(qubitVars[qubit2], couplingVar);
					IloLinearIntExpr qubitSumMinusOne = cplex.linearIntExpr();
					qubitSumMinusOne.setConstant(-1);
					qubitSumMinusOne.addTerm(1, qubitVars[qubit1]);
					qubitSumMinusOne.addTerm(1, qubitVars[qubit2]);
					cplex.addGe(couplingVar, qubitSumMinusOne);
					// integrate coupling variable into energy formula
					energyFormula.addTerm(weight, couplingVar);
					// advance used coupling index
					++nextCouplingIndex;
				}
			}
		}
	}
	/**
	 * Checks that qubit and coupling variables are set consistently in the optimal solution.
	 * 
	 * @param qubitVars			variables representing the state of single qubits
	 * @param couplingVars		variables representing interactions between qubit pairs
	 * @param couplings			stores for each coupling variable the associated qubit pair
	 * @return					true if assignments are consistent
	 * @throws Exception
	 */
	boolean consistentCouplingVarAssignments(IloIntVar[] qubitVars, IloIntVar[] couplingVars, Coupling[] couplings) throws Exception {
		assert(couplingVars.length == couplings.length);
		int nrCouplings = couplings.length;
		for (int couplingIndex=0; couplingIndex<nrCouplings; ++couplingIndex) {
			IloIntVar couplingVar = couplingVars[couplingIndex];
			boolean couplingValue = cplex.getValue(couplingVar) > 0.5 ? true : false;
			Coupling coupling = couplings[couplingIndex];
			int qubit1 = coupling.qubit1;
			int qubit2 = coupling.qubit2;
			IloIntVar qubit1Var = qubitVars[qubit1];
			IloIntVar qubit2Var = qubitVars[qubit2];
			boolean qubit1Value = cplex.getValue(qubit1Var) > 0.5 ? true : false;
			boolean qubit2Value = cplex.getValue(qubit2Var) > 0.5 ? true : false;
			if (couplingValue) {
				if (!(qubit1Value && qubit2Value)) {
					return false;
				}
			} else {
				if (!(!qubit1Value || !qubit2Value)) {
					return false;
				}
			}
		}
		return true;
	}
	/**
	 * Extracts for each query the selected plan from CPLEX.
	 * 
	 * @param planVars		matarix of binary variables representing whether 
	 * 						a specific plan is selected for a specific query
	 * @param nrQueries		the number of queries
	 * @param nrPlans		the number of alternative plans per query
	 * @return				an integer array capturing for each query the index of the optimal plan
	 * @throws Exception  
	 */
	int[] extractPlanSelections(IloIntVar[][] planVars, int nrQueries, int nrPlans) throws Exception {
		// If Cplex did not yet find a solution then return null pointer
		if (cplex.getStatus() == IloCplex.Status.Unknown) {
			return null;
		}
		// Extract plan selections
		int[] planSelections = new int[nrQueries];
		for (int query=0; query<nrQueries; ++query) {
			int selected = -1;
			for (int plan=0; plan<nrPlans; ++plan) {
				IloIntVar planVar = planVars[query][plan];
				double varValue = cplex.getValue(planVar);
				if (varValue > 0.5) {
					// Make sure that at most one plan is selected
					assert(selected == -1);
					selected = plan;
				}
			}
			assert(selected != -1);
			planSelections[query] = selected;
		}
		return planSelections;
	}
	/**
	 * Solves a MQO problem instance directly without transforming it into
	 * a Qubo or Ising problem, stores statistics about how the quality
	 * of the solution improved over time, and returns the optimal solution.
	 */
	@Override
	public MqoSolution solve(ChimeraMqoProblem problem) throws Exception {
		System.out.println("Initializing CPLEX for solving an MQO instance");
		// clear CPLEX model
		cplex.clearModel();
		// extract variables
		int nrQueries = problem.nrQueries;
		int nrPlans = problem.nrPlansPerQuery;
		int nrCouplingsUsed = problem.interactions.size();
		// Create required CPELX variables
		IloIntVar[][] planVars = new IloIntVar[nrQueries][nrPlans];
		for (int query=0; query<nrQueries; ++query) {
			planVars[query] = cplex.boolVarArray(nrPlans);
		}
		IloIntVar[] couplingVars = cplex.boolVarArray(nrCouplingsUsed);
		// Add CPLEX variables to model to be able to retrieve their values in the end
		for (int query=0; query<nrQueries; ++query) {
			cplex.add(planVars[query]);
		}
		cplex.add(couplingVars);
		// Add constraints enforcing one plan selection
		for (int query=0; query<nrQueries; ++query) {
			IloLinearIntExpr nrPlanSelections = cplex.linearIntExpr();
			for (int plan=0; plan<nrPlans; ++plan) {
				IloIntVar planVar = planVars[query][plan];
				nrPlanSelections.addTerm(1, planVar);
			}
			cplex.addEq(nrPlanSelections, 1);
		}
		// This formula shall be minimized
		IloLinearNumExpr executionCost = cplex.linearNumExpr();
		// Integrate execution costs of single plans
		for (int query=0; query<nrQueries; ++query) {
			for (int plan=0; plan<nrPlans; ++plan) {
				IloIntVar planVar = planVars[query][plan];
				double cost = problem.planCost[query][plan];
				executionCost.addTerm(cost, planVar);
			}
		}
		// Integrate cost savings between plans
		int nextCouplingIndex = 0;
		for (Entry<PlanCoupling, Double> entry : problem.interactions.entrySet()) {
			PlanCoupling planCoupling = entry.getKey();
			Double costDelta = entry.getValue();
			IloIntVar planVar1 = planVars[planCoupling.query1][planCoupling.plan1];
			IloIntVar planVar2 = planVars[planCoupling.query2][planCoupling.plan2];
			// impose constraint on coupling variable
			IloIntVar couplingVar = couplingVars[nextCouplingIndex];
			cplex.addGe(planVar1, couplingVar);
			cplex.addGe(planVar2, couplingVar);
			IloLinearIntExpr qubitSumMinusOne = cplex.linearIntExpr();
			qubitSumMinusOne.setConstant(-1);
			qubitSumMinusOne.addTerm(1, planVar1);
			qubitSumMinusOne.addTerm(1, planVar2);
			cplex.addGe(couplingVar, qubitSumMinusOne);
			// integrate coupling variable into energy formula
			executionCost.addTerm(costDelta, couplingVar);
			// advance coupling counter
			++nextCouplingIndex;
		}
		// Prepare QUBO solving
		cplex.addMinimize(executionCost);
		// initialize benchmarking variables
		long startMillis = System.currentTimeMillis();
		Arrays.fill(lastRunCheckpointCost, Double.POSITIVE_INFINITY);
		int[] planSelections = null;
		// invoke CPLEX multiple times and store cost value after each time interval
		for (int intervalCtr=0; intervalCtr<BenchmarkConfiguration.nrBenchmarkTimes; ++intervalCtr) {
			System.out.println("ILP interval: " + intervalCtr);
			// TODO: Verify if this change is correct
			//long lowerBoundMillis = intervalCtr==0 ? 0 : BenchmarkConfiguration.benchmarkTimes[intervalCtr-1];
			long lowerBoundMillis = 0;
			long upperBoundMillis = BenchmarkConfiguration.benchmarkTimes[intervalCtr];
			long durationMillis = upperBoundMillis - lowerBoundMillis;
			// Set time limit (use default setting for clock mode: wall clock time)
			cplex.setParam(IloCplex.LongParam.TimeLimit, durationMillis/1000.0);
			cplex.solve();
			// Extract best solution found and update cost statistics
			planSelections = extractPlanSelections(planVars, nrQueries, nrPlans);
			double cost = planSelections != null ? problem.executionCost(planSelections) : Double.POSITIVE_INFINITY;
			updateStats(cost, upperBoundMillis);
		}
		// solve without timeout
		//cplex.setParam(IloCplex.LongParam.TimeLimit, 1E75);
		//cplex.solve();
		// stop timer
		//lastRunSolverMillis = System.currentTimeMillis() - startMillis;
		// verify that optimal solution was found
		//Status status = cplex.getStatus();
		//assert(status == IloCplex.Status.Optimal);
		// Extract plan selections
		//int[] planSelections = extractPlanSelections(planVars, nrQueries, nrPlans);
		lastRunOptimalCost = planSelections != null ? problem.executionCost(planSelections) : Double.POSITIVE_INFINITY;
		return new MqoSolution(problem, planSelections);
	}
	/**
	 * Solves a MQO problem instance in Qubo representation.
	 * 
	 * @param mapping		contains MQO problem and corresponding QUBO representation
	 * @return				MQO problem solution
	 * @throws Exception
	 */
	public QuadraticMqoSolution solveChimeraQubo(ChimeraMqoMapping mapping) throws Exception {
		System.out.println("Initializing CPLEX");
		// Extract variables
		int nrQubits = mapping.nrQubits;
		// clear CPLEX model
		cplex.clearModel();
		// analyze QUBO
		int nrCouplingsUsed = mapping.nrCouplingsUsed();
		// Create required CPELX variables
		IloIntVar[] qubitVars = cplex.boolVarArray(nrQubits);
		IloIntVar[] couplingVars = cplex.boolVarArray(nrCouplingsUsed);
		Coupling[] usedCouplings = new Coupling[nrCouplingsUsed];
		// Add CPLEX variables to model to be able to retrieve their values in the end
		cplex.add(qubitVars);
		cplex.add(couplingVars);
		// This formula shall be minimized
		IloLinearNumExpr energyFormula = cplex.linearNumExpr();
		// Treat weights on single qubits
		treatQubitWeights(mapping, qubitVars, energyFormula);
		// Treat weights between pairs of qubits
		treatCouplingWeights(mapping, qubitVars, couplingVars, energyFormula, usedCouplings);
		// Prepare QUBO solving
		cplex.addMinimize(energyFormula);
		// Get underlying MQO problem instance and dimensions
		ChimeraMqoProblem problem = mapping.problem;
		int nrQueries = problem.nrQueries;
		// initialize benchmarking variables
		long startMillis = System.currentTimeMillis();
		Arrays.fill(lastRunCheckpointCost, Double.POSITIVE_INFINITY);
		// invoke CPLEX multiple times and store cost value after each time interval
		for (int intervalCtr=0; intervalCtr<BenchmarkConfiguration.nrBenchmarkTimes; ++intervalCtr) {
			long lowerBoundMillis = intervalCtr==0 ? 0 : BenchmarkConfiguration.benchmarkTimes[intervalCtr-1];
			long upperBoundMillis = BenchmarkConfiguration.benchmarkTimes[intervalCtr];
			long durationMillis = upperBoundMillis - lowerBoundMillis;
			// Set time limit (use default setting for clock mode: wall clock time)
			cplex.setParam(IloCplex.LongParam.TimeLimit, durationMillis/1000.0);
			cplex.solve();
			// Extract best solution found and update cost statistics
			double[] qubitsDoubleValues = cplex.getValues(qubitVars);
			boolean[] qubitValues = qubitsDoubleValues == null ? null : 
				GenericUtil.extractQubitValues(qubitsDoubleValues);
			int[] planSelections = new int[nrQueries];
			for (int qubit=0; qubit<nrQubits; ++qubit) {
				int query = mapping.associatedQuery[qubit];
				int plan = mapping.associatedPlan[qubit];
				if (query!=-1 && plan!=-1) {
					if (qubitValues[qubit]) {
						planSelections[query] = plan;
					}
				}
			}
			double cost = planSelections != null ? problem.executionCost(planSelections) : 
				Double.POSITIVE_INFINITY;
			updateStats(cost, upperBoundMillis);
		}
		// solve without timeout
		System.out.println("Interrupting optimization after timeout - we don't search optimal solution");
		cplex.setParam(IloCplex.LongParam.TimeLimit, 1E75);
		cplex.solve();
		// stop timer
		lastRunSolverMillis = System.currentTimeMillis() - startMillis;
		// verify that optimal solution was found
		Status status = cplex.getStatus();
		assert(status == IloCplex.Status.Optimal);
		// Check that coupling variables have valid assignment
		assert(consistentCouplingVarAssignments(qubitVars, couplingVars, usedCouplings));
		// Get qubit values as doubles
		double[] qubitValuesDouble = cplex.getValues(qubitVars);
		// Get qubit values as Booleans
		boolean[] qubitValues = GenericUtil.extractQubitValues(qubitValuesDouble);
		// Generate solution
		QuadraticMqoSolution solution = new QuadraticMqoSolution(mapping, qubitValues);
		/*
		// Get objective value
		double objectiveValue = cplex.getObjValue();
		// Verify consistency
		assert(TestUtil.sameValue(objectiveValue, mapping.getEnergy(qubitValues)));
		*/
		// Transform QUBO solution into MQO solution
		return solution;
	}
	/**
	 * Solve an Ising problem.
	 * 
	 * @param isingMapping	a mapping representing an Ising problem
	 * @return				a solution to the Ising problem
	 * @throws Exception
	 */
	public QuadraticMqoSolution solveChimeraIsing(ChimeraMqoMapping isingMapping) throws Exception {
		System.out.println("Initializing CPLEX");
		// Extract variables
		int nrSpins = isingMapping.nrQubits;
		// clear CPLEX model
		cplex.clearModel();
		int nrCouplingsUsed = isingMapping.nrCouplingsUsed();
		// Create required CPELX variables
		IloIntVar[] binaryVars = cplex.boolVarArray(nrSpins);
		IloIntVar[] couplingVars = cplex.boolVarArray(nrCouplingsUsed);
		Coupling[] usedCouplings = new Coupling[nrCouplingsUsed];
		// Add CPLEX variables to model to be able to retrieve their values in the end
		cplex.add(binaryVars);
		cplex.add(couplingVars);
		// The sum of those terms shall be minimized
		List<IloNumExpr> energyTerms = new LinkedList<IloNumExpr>();
		// Treat weights on single spins (h vector)
		for (int spin=0; spin<nrSpins; ++spin) {
			double weight = isingMapping.getWeight(spin);
			if (weight != 0) {
				IloIntVar binaryVar = binaryVars[spin];
				IloLinearNumExpr weightedSpinExpr = cplex.linearNumExpr();
				weightedSpinExpr.setConstant(-weight);
				weightedSpinExpr.addTerm(2*weight, binaryVar);
				energyTerms.add(weightedSpinExpr);
			}
		}
		// Relate coupling variables to the spins and add corresponding weights
		int couplingIndex = 0;
		for (int spin1=0; spin1<nrSpins; ++spin1) {
			for (int spin2=spin1+1; spin2<nrSpins; ++spin2) {
				double weight = isingMapping.getConnectionWeight(spin1, spin2);
				if (weight != 0) {
					IloIntVar binaryVar1 = binaryVars[spin1];
					IloIntVar binaryVar2 = binaryVars[spin2];
					IloIntVar couplingVar = couplingVars[couplingIndex];
					// Generate required linear expressions
					IloLinearIntExpr onePlusTwo = cplex.linearIntExpr();
					IloLinearIntExpr oneMinusTwo = cplex.linearIntExpr();
					IloLinearIntExpr twoMinusOne = cplex.linearIntExpr();
					IloLinearIntExpr minusOneMinusTwoConst2 = cplex.linearIntExpr();
					//IloLinearIntExpr couplingMinusOne = cplex.linearIntExpr();
					// Assign linear expressions
					onePlusTwo.addTerm(1, binaryVar1);
					onePlusTwo.addTerm(1, binaryVar2);
					oneMinusTwo.addTerm(1, binaryVar1);
					oneMinusTwo.addTerm(-1, binaryVar2);
					twoMinusOne.addTerm(1, binaryVar2);
					twoMinusOne.addTerm(-1, binaryVar1);
					minusOneMinusTwoConst2.setConstant(2);
					minusOneMinusTwoConst2.addTerm(-1, binaryVar1);
					minusOneMinusTwoConst2.addTerm(-1, binaryVar2);
					// Impose constraints: coupling var corresponds to XOR of binary vars
					cplex.addLe(couplingVar, onePlusTwo);
					cplex.addGe(couplingVar, oneMinusTwo);
					cplex.addGe(couplingVar, twoMinusOne);
					cplex.addLe(couplingVar, minusOneMinusTwoConst2);
					// Generate spin for coupling - coupling var = 1 means
					// a negative coupling spin, coupling var = 0 means a
					// positive value.
					IloLinearNumExpr weightedCouplingSpin = cplex.linearNumExpr();
					weightedCouplingSpin.setConstant(weight);
					weightedCouplingSpin.addTerm(-2*weight, couplingVar);
					// Add weighted coupling spin to energy terms
					energyTerms.add(weightedCouplingSpin);
					// Register used coupling
					usedCouplings[couplingIndex] = new Coupling(spin1, spin2);
					// Advance coupling index
					++couplingIndex;
				}
			}
		}
		assert(couplingIndex == nrCouplingsUsed);
		// Transform list of energy terms into array
		IloNumExpr[] energyTermsArray = energyTerms.toArray(new IloNumExpr[energyTerms.size()]);
		// Add up energy terms
		IloNumExpr energyLevel = cplex.sum(energyTermsArray);
		// Prepare Ising solving
		cplex.addMinimize(energyLevel);
		// start timer
		long startMillis = System.currentTimeMillis();
		// solve
		cplex.solve();
		// stop timer
		lastRunSolverMillis = System.currentTimeMillis() - startMillis;
		// verify that optimal solution was found
		Status status = cplex.getStatus();
		assert(status == IloCplex.Status.Optimal);
		// Assert consistency
		for (couplingIndex=0; couplingIndex<nrCouplingsUsed; ++couplingIndex) {
			Coupling coupling = usedCouplings[couplingIndex];
			int spin1 = coupling.qubit1;
			int spin2 = coupling.qubit2;
			IloIntVar spin1Var = binaryVars[spin1];
			IloIntVar spin2Var = binaryVars[spin2];
			IloIntVar couplingVar = couplingVars[couplingIndex];
			boolean spin1Value = cplex.getValue(spin1Var) > 0.5 ? true : false;
			boolean spin2Value = cplex.getValue(spin2Var) > 0.5 ? true : false;
			boolean couplingValue = cplex.getValue(couplingVar) > 0.5 ? true : false;
			if (spin1Value != spin2Value) {
				assert(couplingValue);
			} else {
				assert(!couplingValue);
			}
		}
		// Extract qubit values as doubles
		double[] qubitValuesDouble = cplex.getValues(binaryVars);
		// Obtain qubit values as Booleans
		boolean[] qubitValues = GenericUtil.extractQubitValues(qubitValuesDouble);
		/*
		// Get objective value
		double objectiveValue = cplex.getObjValue();
		// Verify consistency
		assert(TestUtil.sameValue(objectiveValue, isingMapping.getEnergy(qubitValues))) :
			"Cplex objective value: " + objectiveValue + 
			"; Energy: " + isingMapping.getEnergy(qubitValues);
		*/
		// Transform QUBO solution into MQO solution
		return new QuadraticMqoSolution(isingMapping, qubitValues);
	}
	@Override
	public String solverID() {
		return "LIN";
	}
}

