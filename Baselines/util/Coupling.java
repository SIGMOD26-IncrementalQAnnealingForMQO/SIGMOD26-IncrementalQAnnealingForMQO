package mqo_chimera.util;

/**
 * Represents an available coupling between two qubits which allows to define a weight
 * that gets activated once both qubits are set to state one. 
 * 
 * @author immanueltrummer
 *
 */
public class Coupling {
	/**
	 * Index of first qubit
	 */
	public final int qubit1;
	/**
	 * Index of second qubit
	 */
	public final int qubit2;
	
	public Coupling(int qubit1, int qubit2) {
		assert(qubit1 != qubit2);
		this.qubit1 = qubit1;
		this.qubit2 = qubit2;
	}
	/**
	 * Two couplings are equal if they connect the same two qubits.
	 */
	@Override
	public boolean equals(Object otherObject) {
		Coupling otherCoupling = (Coupling)otherObject;
		return (qubit1 == otherCoupling.qubit1 && qubit2 == otherCoupling.qubit2) ||
				(qubit1 == otherCoupling.qubit2 && qubit2 == otherCoupling.qubit1);
	}
	/**
	 * Two couplings are equal independently from the order of the qubits -
	 * therefore the hash code function must be commutative.
	 */
	@Override
	public int hashCode() {
		return qubit1 * qubit2;
	}
}
