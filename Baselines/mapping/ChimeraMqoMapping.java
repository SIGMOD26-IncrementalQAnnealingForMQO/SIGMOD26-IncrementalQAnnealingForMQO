package mqo_chimera.mapping;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.Set;
import java.util.TreeSet;

import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.testcases.ChimeraMqoProblem;
import mqo_chimera.util.AmesUtil;
import mqo_chimera.util.Coupling;
import mqo_chimera.util.RandomUtil;


/**
 * Represents a mapping of a MQO problem instance into a QUBO (Quadratic Unconstrained Binary Optimization) 
 * or Ising problem.
 * 
 * @author immanueltrummer
 *
 */
public class ChimeraMqoMapping implements Serializable {
	/**
	 * Used to verify the class version.
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * Whether the mapping should be interpreted as QUBO or as ISING problem.
	 */
	public final MappingType mappingType;
	/**
	 * The multiple query optimization problem that this mapping maps into a QUBO or ISING representation.
	 */
	public ChimeraMqoProblem problem;
	/**
	 * The number of (broken or intact) qubits.
	 */
	public final int nrQubits = 1152;
	/**
	 * Maps queries to groups of variables representing alternative query plans.
	 */
	public LogicalVariable[][] planVars;
	/**
	 * Stores for each qubit representing a query plan the query index.
	 */
	public int[] associatedQuery;
	/**
	 * Stores for each qubit representing a query plan the plan index.
	 */
	public int[] associatedPlan;
	/**
	 * Optimal energy value derived from the planted solution.
	 */
	public double plantedEnergy;
	/**
	 * Stores for each qubit and each pair of qubits the associated coupling weight.
	 */
	final double[][] weights = new double[nrQubits][nrQubits];
	/**
	 * Describes gauge transformations that are used to cope with bias on the annealer.
	 */
	public boolean[][] gaugeTransformations = new boolean[BenchmarkConfiguration.nrTransformations][nrQubits];
	
	public ChimeraMqoMapping(MappingType mappingType) {
		this.mappingType = mappingType;
	}
	/**
	 * Returns the weight associated with the given qubit.
	 * 
	 * @param qubit	index of the qubit whose weight is returned
	 * @return		the weight assigned to the given qubit
	 */
	public double getWeight(int qubit) {
		return weights[qubit][qubit];
	}
	/**
	 * Returns the weight between two qubits, automatically uses smaller index as first index
	 * 
	 * @param qubit1	index of first qubit
	 * @param qubit2	index of second qubit
	 * @return			the coupling weight between the given qubits
	 */
	public double getConnectionWeight(int qubit1, int qubit2) {
		int min_index = Math.min(qubit1, qubit2);
		int max_index = Math.max(qubit1, qubit2);
		return weights[min_index][max_index];
	}
	/**
	 * Adds weight to the corresponding qubit or coupling
	 * 
	 * @param qubit1		index of first qubit
	 * @param qubit2		index of second qubit
	 * @param addedWeight	the weight to add to the one currently set
	 */
	public void addWeight(int qubit1, int qubit2, double addedWeight) {
		assert(qubit1 == qubit2 || AmesUtil.amesConnected(qubit1, qubit2));
		// we only use the lower triangle of the weight matrix
		int min_index = Math.min(qubit1, qubit2);
		int max_index = Math.max(qubit1, qubit2);
		weights[min_index][max_index] += addedWeight;
	}
	/**
	 * Write current weights into a file (from where they can be read by a program
	 * that communicates with the D-Wave hardware at NASA Ames research).
	 * 
	 * @param filename		the name of the output file
	 * @param description	header line contains a description
	 * @throws Exception
	 */
	public void weightsToFile(String filename, String description) throws Exception {
		PrintWriter writer = new PrintWriter(filename);
		writer.println(description);
		for (int i = 0; i < nrQubits; ++i) {
			for (int j = i; j < nrQubits; ++j) {
				if (weights[i][j] != 0) {
					assert(i == j && AmesUtil.amesQubits.contains(i) || AmesUtil.amesConnected(i, j));
					writer.println(i + "," + j + "," + weights[i][j]);
				}
			}
		}
		writer.close();
	}
	/**
	 * Writes a transformed weight matrix to a file on disc, using the gauge transformation
	 * whose index is stored as parameter.
	 * 
	 * @param filename				the name of the output file
	 * @param description			header line contains a description
	 * @param gaugeTransformIndex	index of gauge transformation to use
	 * @throws Exception
	 */
	public void transformedWeightsToFile(String filename, String description, int gaugeTransformIndex) throws Exception {
		assert(gaugeTransformIndex < BenchmarkConfiguration.nrTransformations);
		PrintWriter writer = new PrintWriter(filename);
		writer.println(description);
		// Treat single spins (vector h)
		for (int i = 0; i < nrQubits; ++i) {
			double weight = weights[i][i];
			if (weight != 0) {
				assert(AmesUtil.amesQubits.contains(i));
				int spinFactorI = gaugeTransformations[gaugeTransformIndex][i] ? 1 : -1;
				writer.println(i + "," + i + "," + spinFactorI * weights[i][i]);				
			}
		}
		// Treat spin couplings (matrix J)
		for (int i = 0; i < nrQubits; ++i) {
			for (int j = i+1; j < nrQubits; ++j) {
				if (weights[i][j] != 0) {
					assert(i == j || AmesUtil.amesConnected(i, j));
					int spinFactorI = gaugeTransformations[gaugeTransformIndex][i] ? 1 : -1;
					int spinFactorJ = gaugeTransformations[gaugeTransformIndex][j] ? 1 : -1; 
					writer.println(i + "," + j + "," + spinFactorI * weights[i][j] * spinFactorJ);
				}
			}
		}
		writer.close();
	}
	/**
	 * Reads the weights of a Qubo or Ising problem from disc and combines it with the
	 * variable mapping and problem definition from a given MQO mapping, returns the
	 * resulting mapping.
	 * 
	 * @param filename			a CSV file containing the weights
	 * @param nonWeightsSource	a mapping from which to extract all information except for the weights
	 * @return					a new mapping that combines the weights from the file with the variable mappings
	 * 							and problem statement from the given mapping
	 */
	public static ChimeraMqoMapping weightsFromFile(String filename, ChimeraMqoMapping nonWeightsSource) throws Exception {
		// Generate mapping to read
		ChimeraMqoMapping discMapping = new ChimeraMqoMapping(nonWeightsSource.mappingType);
		discMapping.problem = nonWeightsSource.problem;
		discMapping.associatedQuery = nonWeightsSource.associatedQuery;
		discMapping.associatedPlan = nonWeightsSource.associatedPlan;
		discMapping.planVars = nonWeightsSource.planVars;
		// Open file containing test case weights
		FileReader fileReader = new FileReader(filename);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		// Skip header
		bufferedReader.readLine();
		// Read weights from file
		String line;
		while ((line = bufferedReader.readLine()) != null) {
			String[] splits = line.split(",");
			int spin1 = Integer.parseInt(splits[0]);
			int spin2 = Integer.parseInt(splits[1]);
			double weight = Double.parseDouble(splits[2]);
			discMapping.addWeight(spin1, spin2, weight);
		}
		// Close weight file
		bufferedReader.close();
		// Return generated mapping
		return discMapping;
	}
	/**
	 * Write current mapping (the weights as well as the semantic for each qubit) to a file.
	 * 
	 * @param filename		the name of the file to create
	 * @throws Exception
	 */
	public void toFile(String filename) throws Exception {
		FileOutputStream fout = new FileOutputStream(filename);
		ObjectOutputStream oos = new ObjectOutputStream(fout);   
		oos.writeObject(this);
		oos.close();
	}
	/**
	 * Print out qubit assignments to console.
	 */
	public void toConsole() {
		for (int query=0; query<problem.nrQueries; ++query) {
			for (int plan=0; plan<problem.nrPlansPerQuery; ++plan) {
				System.out.println("Plan variable for query " + query + ", plan " + plan);
				LogicalVariable var = planVars[query][plan];
				var.toConsole();
			}
		}
	}
	/**
	 * Randomly generates all gauge transformations.
	 */
	public void generateGaugeTransformations() {
		for (int transformationCtr=0; transformationCtr<BenchmarkConfiguration.nrTransformations; ++transformationCtr) {
			for (int qubit=0; qubit<nrQubits; ++qubit) {
				gaugeTransformations[transformationCtr][qubit] = RandomUtil.random.nextBoolean();
			}
		}
	}
	/**
	 * Counts the number of couplings with non-zero weights
	 * 
	 * @return	the number of connections between qubits with non-zero weight
	 */
	public int nrCouplingsUsed() {
		int nrCouplingsUsed = 0;
		for (int i = 0; i < nrQubits; ++i) {
			for (int j = i+1; j < nrQubits; ++j) {
				if (weights[i][j] != 0) {
					++nrCouplingsUsed;
				}
			}
		}
		return nrCouplingsUsed;
	}
	/**
	 * Checks whether the qubit sets assigned to different logical variables overlap -
	 * this should not happen!
	 * 
	 * @return	true if an overlap is detected
	 */
	public boolean hasOverlap() {
		// if the number of qubit assignments is larger than the number of used qubits
		// then we must have an overlap
		int nrQubitAssignments = 0;
		Set<Integer> usedQubits = new TreeSet<Integer>();
		// iterate over all variables
		for (int query=0; query<problem.nrQueries; ++query) {
			for (int plan=0; plan<problem.nrPlansPerQuery; ++plan) {
				LogicalVariable planVar = planVars[query][plan];
				Set<Integer> curVarQubits = planVar.getQubits();
				nrQubitAssignments += curVarQubits.size();
				usedQubits.addAll(curVarQubits);
			}
		}
		// if there is no overlap then the number of used qubits is the same as
		// the distinct number of used qubits
		return nrQubitAssignments != usedQubits.size();
	}
	/**
	 * Calculate the energy level achieved for specific qubit values when
	 * using the current mapping.
	 * 
	 * @param qubitValues	Boolean vector indicating for each qubit its value
	 * @return				the energy level achieved by the qubit value combination
	 */
	/*
	public double getQuboEnergy(boolean[] qubitValues) {
		// make sure that value vector has the right length
		int nrQubits = qubitValues.length;
		assert(nrQubits == this.nrQubits);
		// treat weights on single qubits
		double energy = 0;
		for (int qubit=0; qubit<nrQubits; ++qubit) {
			if (qubitValues[qubit]) {
				energy += getWeight(qubit);
			}
		}
		// treat connections between qubits
		for (int qubit1=0; qubit1<nrQubits; ++qubit1) {
			for (int qubit2=qubit1+1; qubit2<nrQubits; ++qubit2) {
				if (qubitValues[qubit1] && qubitValues[qubit2]) {
					energy += getConnectionWeight(qubit1, qubit2);					
				}
			}
		}
		// return accumulated energy
		return energy;
	}
	*/
	/**
	 * Calculates the energy level of this mapping assuming that
	 * it represents a QUBO problem. 
	 * 
	 * @param qubitValues	Boolean vector indicating for each qubit its value
	 * @return				the energy level achieved by the qubit value combination
	 */
	private double getQuboEnergy(boolean[] qubitValues) {
		// make sure that value vector has the right length
		int nrQubits = qubitValues.length;
		assert(nrQubits == this.nrQubits);
		// treat weights on single qubits
		double energy = 0;
		for (int qubit=0; qubit<nrQubits; ++qubit) {
			if (qubitValues[qubit]) {
				energy += getWeight(qubit);
			}
		}
		// treat connections between qubits
		for (Coupling coupling : AmesUtil.amesCouplings) {
			int qubit1 = coupling.qubit1;
			int qubit2 = coupling.qubit2;
			if (qubitValues[qubit1] && qubitValues[qubit2]) {
				energy += getConnectionWeight(qubit1, qubit2);
			}
		}
		// return accumulated energy
		return energy;
	}
	/**
	 * Calculates energy for a specific spin configuration, interpreting
	 * the Boolean value true as positive spin and the Boolean value false
	 * as negative spin. 
	 * 
	 * @param qubitValues	assignment for the spins, true signifying a positive and false a negative spin
	 * @return				the energy level achieved by the corresponding Ising problem
	 */
	private double getIsingEnergy(boolean[] spinValues) {
		// make sure that value vector has the right length
		int nrQubits = spinValues.length;
		assert(nrQubits == this.nrQubits);
		// entries in the diagonal represent components of the h vector
		double energy = 0;
		for (int spin=0; spin<nrQubits; ++spin) {
			int spinFactor = spinValues[spin] ? +1 : -1;
			energy += spinFactor * getWeight(spin);
		}
		// the other entries represent interactions between spins
		for (int spin1=0; spin1<nrQubits; ++spin1) {
			for (int spin2=spin1+1; spin2<nrQubits; ++spin2) {
				int spinFactor1 = spinValues[spin1] ? +1 : -1;
				int spinFactor2 = spinValues[spin2] ? +1 : -1;
				int totalSpin = spinFactor1 * spinFactor2;
				energy += totalSpin * getConnectionWeight(spin1, spin2);
			}
		}
		// return accumulated energy
		return energy;
	}
	/**
	 * Calculates the energy for a specific qubit or spin value assignment, decides how
	 * to calculate the energy level based on the mapping type (Qubo or Ising).
	 * 
	 * @param spinOrQubitValues	Boolean vector containing true if the qubit is assigned to one
	 * 							respective the corresponding spin is set to +1
	 * @return					the energy level for the given value assignment
	 */
	public double getEnergy(boolean[] spinOrQubitValues) {
		return mappingType == MappingType.QUBO ? getQuboEnergy(spinOrQubitValues) : 
			getIsingEnergy(spinOrQubitValues);
	}
	/**
	 * Reads a serialized MQO mapping from a file on disc.
	 * 
	 * @param filename		the name of the file containing a serialized MQO problem
	 * @return				a MQO mapping object
	 * @throws Exception
	 */
	public static ChimeraMqoMapping readMapping(String filename) throws Exception {
		InputStream file = new FileInputStream(filename);
		InputStream buffer = new BufferedInputStream(file);
		ObjectInput input = new ObjectInputStream (buffer);
		ChimeraMqoMapping mapping = (ChimeraMqoMapping)input.readObject();
		input.close();
		return mapping;
	}
	/**
	 * Transforms the QUBO matrix into an Ising problem matrix such that it can be solved by C-Wave.
	 * We use the transformation as described in the documentation "Programming with QUBOs" by D-Wave.
	 * 
	 * @return	a new mapping in which the weight matrix and the objective value have been adapted
	 */
	public ChimeraMqoMapping toIsing() {
		// create new mapping and set first fields
		ChimeraMqoMapping isingMapping = new ChimeraMqoMapping(MappingType.ISING);
		isingMapping.problem = problem;
		isingMapping.planVars = planVars;
		isingMapping.associatedQuery = associatedQuery;
		isingMapping.gaugeTransformations = gaugeTransformations;
		// prepare data structures holding weights of Ising problem
		int nrSpins = nrQubits;
		double[] h = new double[nrSpins];
		double[][] J = new double[nrSpins][nrSpins];
		// treat matrix J - note that matrix J is an upper triangle
		for (int spin1=0; spin1<nrSpins; ++spin1) {
			for (int spin2=spin1+1; spin2<nrSpins; ++spin2) {
				J[spin1][spin2] = 0.25 * weights[spin1][spin2];
			}
		}
		// treat vector h
		for (int spin1=0; spin1<nrSpins; ++spin1) {
			h[spin1] = 0.5 * weights[spin1][spin1];
			for (int spin2=0; spin2<nrSpins; ++spin2) {
				h[spin1] += J[spin1][spin2];
				h[spin1] += J[spin2][spin1];
			}
		}
		// treat constant term by which the Qubo energy level differs from Ising level
		double R = 0;
		for (int spin1=0; spin1<nrSpins; ++spin1) {
			for (int spin2=spin1+1; spin2<nrSpins; ++spin2) {
				R += J[spin1][spin2];
			}
			R -= h[spin1];
		}
		// set corresponding fields in new mapping
		for (int spin1=0; spin1<nrSpins; ++spin1) {
			// Weights on single spins
			double hWeight = h[spin1];
			if (hWeight != 0) {
				isingMapping.addWeight(spin1, spin1, hWeight);
			}
			// Weights on couplings
			for (int spin2=spin1+1; spin2<nrSpins; ++spin2) {
				double Jweight = J[spin1][spin2];
				if (Jweight != 0) {
					isingMapping.addWeight(spin1, spin2, Jweight);
				}
			}
		}
		isingMapping.plantedEnergy = plantedEnergy + R;
		return isingMapping;
	}
	/**
	 * Returns the maximal absolute weight value on the diagonal of the weight matrix.
	 * For Ising problems, the diagonal entries correspond to vector h. Used to verifiy
	 * that weights are within the admissible range for D-Wave.
	 * 
	 * @return	the maximal absolute value over all entries in the diagonal of the weights matrix
	 */
	public double maxAbsDiagonalWeight() {
		double maxAbsWeight = 0;
		for (int qubit=0; qubit<nrQubits; ++qubit) {
			double diagonalWeight = weights[qubit][qubit];
			double absDiagonalWeight = Math.abs(diagonalWeight);
			maxAbsWeight = Math.max(maxAbsWeight, absDiagonalWeight);
		}
		return maxAbsWeight;
	}
	/**
	 * Returns the maximal absolute weight value over the entries in the weight matrix except
	 * the diagonal. For Ising problems, those entries correspond to matrix J. Used to verifiy
	 * that weights are within the admissible range for D-Wave.
	 * 
	 * @return	the maximal absolute value over all non-diagonal entries in the weight matrix
	 */
	public double maxAbsNonDiagonalWeight() {
		double maxAbsWeight = 0;
		for (int qubit1=0; qubit1<nrQubits; ++qubit1) {
			for (int qubit2=0; qubit2<nrQubits; ++qubit2) {
				if (qubit1 != qubit2) {
					double weight = weights[qubit1][qubit2];
					double absWeight = Math.abs(weight);
					maxAbsWeight = Math.max(maxAbsWeight, absWeight);
				}
			}
		}
		return maxAbsWeight;
	}
	/**
	 * Returns true if this mapping, interpreted as Ising mapping, respects
	 * the energy range constraints imposed by the D-Wave computer at NASA
	 * Ames research.
	 * 
	 * @return	true if all weights are within the admissible range
	 */
	public boolean respectsIsingRanges() {
		// The admissible weight range is [-2,2] for h and [-1,1] for J
		return maxAbsDiagonalWeight() <= 2 && maxAbsNonDiagonalWeight() <= 1;
	}
}