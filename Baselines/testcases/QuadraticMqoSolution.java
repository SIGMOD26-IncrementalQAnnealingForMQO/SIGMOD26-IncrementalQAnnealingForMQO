package mqo_chimera.testcases;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;

import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.mapping.LogicalVariable;

/**
 * Represents the solution to a MQO problem that was reformulated as QUBO/ISING problem.
 * 
 * @author immanueltrummer
 *
 */
public class QuadraticMqoSolution extends MqoSolution implements Serializable {
	/**
	 * Used to verify class version
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * Contains the original MQO problem and how it maps to the QUBO problem.
	 */
	public final ChimeraMqoMapping mapping;
	/**
	 * The values assigned to the different qubits as Boolean values.
	 */
	public final boolean[] qubitValues;
	/**
	 * Objective value of the obtained solution.
	 */
	public final double objectiveValue;
	/**
	 * Contains the values for the logical variables representing alternative query plans
	 * assigned by the solution.
	 */
	public final boolean[][] planVarValues;
	/**
	 * Whether for each logical variable all qubits representing that variable are assigned
	 * to the same values.
	 */
	public final boolean consistenQubitAssignments;
	/**
	 * Whether exactly one plan was selected for each query.
	 */
	public final boolean consistentPlanAssignments;
	
	public QuadraticMqoSolution(ChimeraMqoMapping mapping, boolean[] qubitValues) {
		super(mapping.problem, extractPlanSelections(mapping, qubitValues));
		this.mapping = mapping;
		this.qubitValues = qubitValues;
		this.objectiveValue = mapping.getEnergy(qubitValues);
		this.planVarValues = extractPlanVarValues(mapping, qubitValues);
		this.consistenQubitAssignments = consistentQubitAssignments(mapping, qubitValues);
		this.consistentPlanAssignments = consistentPlanSelections(mapping, planVarValues);
	}
	/**
	 * Extracts the values for logical variables representing alternative query plans.
	 * 
	 * @param mapping		a mapping from MQO problem variables onto the qubit matrix
	 * @param qubitValues	assigned Boolean value for each qubit
	 * @return				Boolean array indicating for each query and plan whether it is selected
	 */
	public static boolean[][] extractPlanVarValues(ChimeraMqoMapping mapping, boolean[] qubitValues) {
		// Extract required variables
		ChimeraMqoProblem problem = mapping.problem;
		int nrQueries = problem.nrQueries;
		int nrPlans = problem.nrPlansPerQuery;
		// Iterate over plan variables
		boolean[][] planVarValues = new boolean[nrQueries][nrPlans];
		for (int query=0; query<nrQueries; ++query) {
			for (int plan=0; plan<nrPlans; ++plan) {
				LogicalVariable planVar = mapping.planVars[query][plan];
				int qubit = planVar.getQubits().iterator().next();
				boolean value = qubitValues[qubit];
				planVarValues[query][plan] = value;
			}
		}
		return planVarValues;
	}
	/**
	 * Extracts the index of the selected plan for each query from the binary plan
	 * selection variables.
	 * 
	 * @param mapping		a mapping from MQO problem variables onto the qubit matrix
	 * @param qubitValues	assigned Boolean value for each qubit
	 * @return				an integer array indicating for each query the index of the selected plan
	 */
	public static int[] extractPlanSelections(ChimeraMqoMapping mapping, boolean[] qubitValues) {
		// Extract required variables
		ChimeraMqoProblem problem = mapping.problem;
		int nrQueries = problem.nrQueries;
		int nrPlans = problem.nrPlansPerQuery;
		// Store the selected plan for each query
		int[] planSelections = new int[nrQueries];
		for (int query=0; query<nrQueries; ++query) {
			planSelections[query] = 0;
			for (int plan=0; plan<nrPlans; ++plan) {
				LogicalVariable planVar = mapping.planVars[query][plan];
				int qubit = planVar.getQubits().iterator().next();
				if (qubitValues[qubit]) {
					planSelections[query] = plan;
					break;
				}
			}
		}
		return planSelections;
	}
	/**
	 * Checks if all qubits representing the same logical variable are assigned
	 * to the same value by the optimal solution.
	 * 
	 * @param mapping		a mapping from MQO problem variables onto the qubit matrix
	 * @param qubitValues	assigned Boolean value for each qubit
	 * @return				true if all assignments are consistent
	 */
	public static boolean consistentQubitAssignments(ChimeraMqoMapping mapping, boolean[] qubitValues) {
		// Extract required variables
		ChimeraMqoProblem problem = mapping.problem;
		int nrQueries = problem.nrQueries;
		int nrPlans = problem.nrPlansPerQuery;
		// Iterate over plan variables to check consistency
		for (int query=0; query<nrQueries; ++query) {
			for (int plan=0; plan<nrPlans; ++plan) {
				LogicalVariable planVar = mapping.planVars[query][plan];
				// Collect all values assigned to qubits representing current variable
				Set<Boolean> valueSet = new TreeSet<Boolean>();
				for (int qubit : planVar.getQubits()) {
					boolean value = qubitValues[qubit];
					valueSet.add(value);
				}
				assert(!valueSet.isEmpty());
				// If more than one value has been assigned then the assignment is inconsistent
				if (valueSet.size() > 1) {
					return false;
				}
			}
		}
		// No inconsistencies have been detected if we arrive here
		return true;
	}
	/**
	 * Check whether at most one plan is selected if independent processing is allowed or
	 * checks if exactly one plan is selected if independent processing is disallowed.
	 * 
	 * @param mapping		a mapping from MQO problem variables onto the qubit matrix
	 * @param planVarValues	indicates for each query and plan whether it was selected
	 * @return				true the plan selections represent a valid solution
	 */
	public static boolean consistentPlanSelections(ChimeraMqoMapping mapping, boolean[][] planVarValues) {
		// Extract required variables
		ChimeraMqoProblem problem = mapping.problem;
		int nrQueries = problem.nrQueries;
		int nrPlans = problem.nrPlansPerQuery;
		boolean allowIndependentProcessing = problem.allowIndependentProcessing;
		// Iterate over plan variables to check consistency
		for (int query=0; query<nrQueries; ++query) {
			int nrSelections = 0;
			for (int plan=0; plan<nrPlans; ++plan) {
				if (planVarValues[query][plan]) {
					++nrSelections;
				}
			}
			// Case distinction: is it allowed to select plans that are not associated with qubits?
			if (allowIndependentProcessing) {
				// If we can select plans that are not associated with qubits
				// then 0 selections and 1 selections are both ok.
				if (nrSelections > 1) {
					return false;
				}				
			} else {
				// If we must select a plan that is associated with qubits then
				// we must have one selection.
				if (nrSelections != 1) {
					return false;
				}
			}
		}
		return true;
	}
	/**
	 * Prints out information about this solution on the console.
	 */
	@Override
	public void toConsole() {
		int nrQueries = mapping.problem.nrQueries;
		System.out.println("--- Mapping ---");
		mapping.toConsole();
		System.out.println("--- Qubit values ---");
		System.out.println(Arrays.toString(qubitValues));
		System.out.println("--- Plan Selections ---");
		for (int query=0; query<nrQueries; ++query) {
			System.out.println("Query " + query + " plan selections: \t");
			System.out.println(Arrays.toString(planVarValues[query]));
		}
	}
	/**
	 * Reads a serialized MQO solution from a file on disc.
	 * 
	 * @param filename		the name of the file containing a serialized MQO solution
	 * @return				a MQO solution object
	 * @throws Exception
	 */
	public static QuadraticMqoSolution readSerializedSolution(String filename) throws Exception {
		InputStream file = new FileInputStream(filename);
		InputStream buffer = new BufferedInputStream(file);
		ObjectInput input = new ObjectInputStream (buffer);
		QuadraticMqoSolution solution = (QuadraticMqoSolution)input.readObject();
		input.close();
		return solution;
	}
	/**
	 * Given a solution to a transformed problem and the transformation, transform the solution
	 * back such that it refers to the original problem afterwards.
	 * 
	 * @param transformedQubitValues	transformed values of qubits
	 * @param transformation			gauge transformation to which the solution refers - each qubit
	 * 									whose field is set to true must be negated.
	 * @return							solution to non-transformed problem
	 */
	public static boolean[] qubitValuesTransformedBack(
			boolean[] transformedQubitValues, boolean[] transformation) {
		// Make sure that description of gauge transformation has the right dimensions
		int nrSpins = transformedQubitValues.length;
		assert(nrSpins == transformation.length);
		// Will contain result of this function
		boolean[] qubitValues = new boolean[nrSpins];
		// Iterate over solution spins and inverse them depending on the transformation
		for (int spin=0; spin<nrSpins; ++spin) {
			qubitValues[spin] = transformation[spin] ? 
					transformedQubitValues[spin] : !transformedQubitValues[spin];
		}
		return qubitValues;
	}
	/**
	 * Reads multiple solutions to a gauge-transformed version of the original problem from a
	 * text file.
	 * 
	 * @param solutionPath			path to the text file containing the transformed solutions
	 * @param mapping				maps variables to qubits and stores a set of gauge transformations
	 * @param transformationIndex	the index of the gauge transformation that was used
	 * @return						a vector of quadratic solutions to the original problem
	 */
	public static QuadraticMqoSolution[] readTransformedTextSolutions(String solutionPath, 
			ChimeraMqoMapping mapping, int transformationIndex) throws Exception {
		// Dimension variables
		int nrSamples = BenchmarkConfiguration.nrSamplesPerTransformation;
		int nrQubits = 1152;
		// This will contain the result of this function call
		QuadraticMqoSolution[] solutions = new QuadraticMqoSolution[nrSamples];
		// This contains the qubit values read from file refering to a transformed problem formulation
		boolean[][] allTransformedQubitValues = new boolean[nrSamples][nrQubits];
		// Open solution file
		FileReader fileReader = new FileReader(solutionPath);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		// Iterate over the number of samples per gauge transformation
		for (int sampleCtr=0; sampleCtr<nrSamples; ++sampleCtr) {
			// Skip introductory lines
			bufferedReader.readLine();	// Line stating solution energy
			bufferedReader.readLine();	// Line stating solution length
			bufferedReader.readLine();	// Line stating number of occurrences
			bufferedReader.readLine();	// header
			// Read qubit values one after the other
			for (int qubitIndex=0; qubitIndex<nrQubits; ++qubitIndex) {
				// Read line representing assignment for one qubit
				String line = bufferedReader.readLine();
				// Values are separated by comma
				String[] splits = line.split(",");
				// Read values
				Integer qubitIndexRead = Integer.parseInt(splits[0]);
				Integer qubitValueRead = Integer.parseInt(splits[1]);
				// Make sure that read qubit value corresponds to expected one
				assert(qubitIndexRead.equals(qubitIndex));
				// Transform integer value into Boolean value
				allTransformedQubitValues[sampleCtr][qubitIndex] = qubitValueRead.equals(1) ? true : false;
			}
		}
		// Close solution file
		bufferedReader.close();
		// Transform qubit values back
		for (int sampleCtr=0; sampleCtr<nrSamples; ++sampleCtr) {
			boolean[] transformation = mapping.gaugeTransformations[transformationIndex];
			boolean[] currentTransformedQubiValues = allTransformedQubitValues[sampleCtr];
			boolean[] qubitValues = qubitValuesTransformedBack(currentTransformedQubiValues, transformation);
			solutions[sampleCtr] = new QuadraticMqoSolution(mapping, qubitValues);
		}
		// Return solutions
		return solutions;
	}
	/**
	 * Serializes solution to a file.
	 * 
	 * @param filename		the name of the file to create
	 * @throws Exception
	 */
	public void serializeToFile(String filename) throws Exception {
		FileOutputStream fout = new FileOutputStream(filename);
		ObjectOutputStream oos = new ObjectOutputStream(fout);   
		oos.writeObject(this);
		oos.close();
	}
}
