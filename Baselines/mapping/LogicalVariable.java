package mqo_chimera.mapping;


import java.io.Serializable;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import mqo_chimera.util.AmesUtil;

/**
 * Represents a logical variable of a QUBO or ISING problem that might be represented either by one
 * or by multiple qubits.
 * 
 * @author immanueltrummer
 *
 */
public class LogicalVariable implements Serializable {
	/**
	 * Used to verify class version.
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * The qubits that represent this variable.
	 */
	private Set<Integer> qubits;
	
	public LogicalVariable() {
		qubits = new TreeSet<Integer>();
	}
	/**
	 * The variable is represented by the qubit passed as parameter
	 * (more qubits can be added).
	 * 
	 * @param qubit	index of a qubit that represents this variable
	 */
	public LogicalVariable(int qubit) {
		qubits = new TreeSet<Integer>();
		addQubit(qubit);
	}
	/**
	 * Add qubit after verifying that the qubit index is working at NASA Ames.
	 * 
	 * @param qubit	index of a qubit that represents this variable
	 */
	public void addQubit(int qubit) {
		assert(AmesUtil.amesQubits.contains(qubit));
		assert(!qubits.contains(qubit));
		qubits.add(qubit);
	}
	/**
	 * Get the set of qubits that represent this variable.
	 * 
	 * @return	the set of qubit indices
	 */
	public Set<Integer> getQubits() {
		return qubits;
	}
	/**
	 * Add the given weight to one of the qubits representing this variable
	 * on the given mapper matrix.
	 * 
	 * @param mapping		maps qubits and couplings to weights and is changed in this call
	 * @param addedWeight	the weight to add
	 */
	public void addWeight(ChimeraMqoMapping mapping, double addedWeight) {
		int qubit = qubits.iterator().next();
		mapping.addWeight(qubit, qubit, addedWeight);
	}
	/**
	 * Get the accumulated weight for the qubits representing this variable on the given mapping.
	 * 
	 * @param mapping	the mapping associating qubits with weights
	 * @return			the sum of weights over all of this variables' qubits
	 */
	public double getWeight(ChimeraMqoMapping mapping) {
		double accumulatedWeight = 0;
		for (int qubit : qubits) {
			accumulatedWeight += mapping.getWeight(qubit);
		}
		return accumulatedWeight;
	}
	/**
	 * Add the given weight on a coupling connecting the qubits of the two variables.
	 * 
	 * @param mapping		maps qubits and couplings to weights and is changed in this call
	 * @param otherVar		the variable to which to connect
	 * @param addedWeight	the weight to add
	 */
	public void addWeight(ChimeraMqoMapping mapping, LogicalVariable otherVar, double addedWeight) {
		for (int qubit : qubits) {
			for (int otherQubit : otherVar.qubits) {
				if (AmesUtil.amesConnected(qubit, otherQubit)) {
					mapping.addWeight(qubit, otherQubit, addedWeight);
					return;
				}
			}
		}
		// We expect this function to be only invoked for variables that can be connected
		assert(false);
	}
	/**
	 * Get accumulated weight between the qubit sets representing this variable and another
	 * variable.
	 * 
	 * @param mapping	the mapping associating pairs of qubits with weights
	 * @param otherVar	another Qubo variable
	 * @return			the sum of weights on couplings connecting a qubit
	 * 					from this variable to one of the other one
	 */
	public double getWeight(ChimeraMqoMapping mapping, LogicalVariable otherVar) {
		// We expect the qubits of the two variables not to overlap
		AmesUtil.assertNoOverlap(qubits, otherVar.qubits);
		double accumulatedWeight = 0;
		for (int qubit : qubits) {
			for (int otherQubit : otherVar.qubits) {
				accumulatedWeight += mapping.getConnectionWeight(qubit, otherQubit);
			}
		}
		return accumulatedWeight;
	}
	/**
	 * Add an equality constraint for the qubits represented by this variable.
	 * 
	 * @param mapping	the qubit matrix on which the corresponding weights are added
	 * @param scaling	if the equality constraint is violated then this weight is added at least
	 */
	public void addEquality(ChimeraMqoMapping mapping, double scaling) {
		assert(!qubits.isEmpty());
		// We expect equality constraints to be scaled by a positive weight
		assert(scaling >= 0);
		// Nothing to do if this variable is represented by only one qubit
		if (qubits.size() == 1) {
			return;
		}
		// Chain qubits representing this variable
		List<Integer> chain = AmesUtil.qubitChain(qubits);
		assert(qubits.size() == chain.size());
		// Add equality constraints between consecutive qubits
		int nrQubits = qubits.size();
		for (int chainIndex=0; chainIndex<nrQubits-1; ++chainIndex) {
			int qubit1 = chain.get(chainIndex);
			int qubit2 = chain.get(chainIndex+1);
			mapping.addWeight(qubit1, qubit1, scaling);
			mapping.addWeight(qubit2, qubit2, scaling);
			mapping.addWeight(qubit1, qubit2, -2.0 * scaling);
		}
	}
	public void toConsole() {
		System.out.println("Used qubits: " + qubits.toString());
	}
	public void smear(ChimeraMqoMapping mapping) {
		double totalWeight = getWeight(mapping);
		// remove weight
		for (int qubit : qubits) {
			double curQubitWeight = mapping.getWeight(qubit);
			mapping.addWeight(qubit, qubit, -curQubitWeight);
		}
		// re-assign weight
		double weightPerQubit = totalWeight / qubits.size();
		for (int qubit : qubits) {
			mapping.addWeight(qubit, qubit, weightPerQubit);
		}
	}
	
	// Pessimistic estimate of local energy for one specific qubit when assigning one
	// specific value (counting the weight of the qubit itself and of all its neighbors
	// in the Chimera graph).
	public static double pessimisticLocalEnergy(int qubit, int value, ChimeraMqoMapping mapping) {
		assert(value>=0 && value<=1);
		double energy = 0;
		// consider weight of qubit itself
		energy += value * mapping.getWeight(qubit);
		// consider weights of connections to other qubits
		Set<Integer> neighbors = AmesUtil.connectedQubits(qubit);
		for (int neighbor : neighbors) {
			double connectionWeight = mapping.getConnectionWeight(qubit, neighbor);
			// Assume both possible neighbor values and take maximum as pessimistic for estimate.
			double energyDelta1		= value * 0 * connectionWeight;
			double energyDelta2		= value * 1 * connectionWeight;
			double pessimisticDelta	= Math.max(energyDelta1, energyDelta2);
			energy += pessimisticDelta;
		}
		return energy;
	}
	
	// Optimistic estimate of local energy for one specific qubit when assigning one
	// specific value (counting the weight of the qubit itself and of all its neighbors
	// in the Chimera graph).
	public static double optimisticLocalEnergy(int qubit, int value, ChimeraMqoMapping mapping) {
		assert(value>=0 && value<=1);
		double energy = 0;
		// consider weight of qubit itself
		energy += value * mapping.getWeight(qubit);
		// consider weights of connections to other qubits
		Set<Integer> neighbors = AmesUtil.connectedQubits(qubit);
		for (int neighbor : neighbors) {
			double connectionWeight = mapping.getConnectionWeight(qubit, neighbor);
			// Assume both possible neighbor values and take maximum as pessimistic for estimate.
			double energyDelta1		= value * 0 * connectionWeight;
			double energyDelta2		= value * 1 * connectionWeight;
			double optimisticDelta	= Math.min(energyDelta1, energyDelta2);
			energy += optimisticDelta;
		}
		return energy;
	}
	
	// Calculate minimal scaling that guarantees that the minimum energy configuration
	// assigns all qubits representing this variable to the same value.
	public double calculateEqualityScalingGeneric(ChimeraMqoMapping mapping) {
		// calculate pessimistic estimate of minimal energy level for consistent assignment
		double allZeroEnergy	= 0;
		double allOneEnergy 	= 0;
		for (int qubit : qubits) {
			allZeroEnergy 	+= pessimisticLocalEnergy(qubit, 0, mapping);
			allOneEnergy	+= pessimisticLocalEnergy(qubit, 1, mapping);
		}
		// the local energy of each consistent assignment is calculated pessimistically
		// but at least we can choose whether the value is zero or one for all qubits.
		double consistentPessimistic = Math.min(allZeroEnergy, allOneEnergy);
		// calculate optimistically minimal energy of inconsistent assignment
		double inconsistentOptimistic = 0;
		for (int qubit : qubits) {
			double zeroEnergy				= optimisticLocalEnergy(qubit, 0, mapping);
			double oneEnergy				= optimisticLocalEnergy(qubit, 1, mapping);
			double optimisticLocalEnergy 	= Math.min(zeroEnergy, oneEnergy);
			inconsistentOptimistic			+= optimisticLocalEnergy;
		}
		// how much energy can be maximally gained by using an inconsistent instead of a consistent assignment?
		double gap = Math.max(0, consistentPessimistic - inconsistentOptimistic);
		// An inconsistent assignment breaks at least one equality constraint - equality weight must be
		// higher than the gap to make a consistent assignment more attractive than any inconsistent
		// assignment.
		return gap * 1.25;
	}
}