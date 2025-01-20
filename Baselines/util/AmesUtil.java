package mqo_chimera.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import mqo_chimera.mapping.LogicalVariable;

/**
 * Contains static utility functions concerning the qubit matrix of the D-Wave machine at NASA Ames research.
 * The machine properties must be read from disc before the utility functions can be used.
 * 
 * @author immanueltrummer
 *
 */
public class AmesUtil {
	/**
	 * Indices of the available qubits at NASA Ames.
	 */
	public static Set<Integer> amesQubits;
	/**
	 * Available couplings between qubits at NASA Ames.
	 */
	public static Set<Coupling> amesCouplings;
	/**
	 * Highest qubit index available at NASA Ames.
	 */
	public static int highestQubitIndex = -1;
	/**
	 * Reads characteristics of the Ames machine from files on disc.
	 * 
	 * @throws Exception
	 */
	public static void initAmes() throws Exception {
		// Read qubit indices from text file
		amesQubits = new HashSet<Integer>();
		{
			FileReader fileReader = new FileReader("AmesQubits.txt");
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			String line;
			while ((line = bufferedReader.readLine()) != null) {
				amesQubits.add(Integer.parseInt(line));
			}
			bufferedReader.close();
		}
		// Read couplings from text file
		amesCouplings = new HashSet<Coupling>();
		{
			FileReader fileReader = new FileReader("AmesCouplings.txt");
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			String line;
			while ((line = bufferedReader.readLine()) != null) {
				String[] splits = line.split(",");
				int qubit1 = Integer.parseInt(splits[0]);
				int qubit2 = Integer.parseInt(splits[1]);
				Coupling coupling = new Coupling(qubit1, qubit2);
				amesCouplings.add(coupling);
			}
			bufferedReader.close();
		}
		// Determine index of highest qubit
		highestQubitIndex = Collections.max(amesQubits);
	}
	/**
	 * Checks whether the D-Wave graph of the NASA Ames D-Wave machine contains
	 * a coupling between the specified qubits.
	 * 
	 * @param qubit1	index of first qubit
	 * @param qubit2	index of second qubit
	 * @return			true if the two qubits are connected
	 */
	public static boolean amesConnected(int qubit1, int qubit2) {
		assert(qubit1 != qubit2);
		for (Coupling coupling : amesCouplings) {
			if ((coupling.qubit1 == qubit1 && coupling.qubit2 == qubit2) ||
					(coupling.qubit1 == qubit2 && coupling.qubit2 == qubit1)) {
				return true;
			}
		}
		return false;
	}
	/**
	 * Checks whether the D-Wave graph of the NASA Ames D-Wave machine contains
	 * a coupling between the qubits of the specified variables.
	 * 
	 * @param var1		first Qubo variable
	 * @param var2		second Qubo variable
	 * @return			true if the two variables are connected
	 */
	public static boolean amesConnected(LogicalVariable var1, LogicalVariable var2) {
		for (int qubit1 : var1.getQubits()) {
			for (int qubit2 : var2.getQubits()) {
				if (amesConnected(qubit1, qubit2)) {
					return true;
				}
			}
		}
		return false;
	}
	/**
	 * Given one qubit and another set of qubits, this function returns one
	 * out of the other qubits that is connected to the first qubit.
	 * 
	 * @param qubit			one qubit
	 * @param otherQubits	a set of qubits where at least one is connected to the first qubit
	 * @return				a connected qubit among the other qubits
	 */
	public static int connectedQubit(int qubit, Set<Integer> otherQubits) {
		// Make sure that the single qubit and the set of qubits are disjunct
		assert(!otherQubits.contains(qubit));
		// Iterate over other qubits and return first connected qubit
		for (int otherQubit : otherQubits) {
			if (amesConnected(qubit, otherQubit)) {
				return otherQubit;
			}
		}
		assert(false);
		return -1;
	}
	
	public static Set<Integer> connectedQubits(int qubit) {
		Set<Integer> connectedQubits = new TreeSet<Integer>();
		for (Coupling coupling : amesCouplings) {
			if (coupling.qubit1 == qubit || coupling.qubit2 == qubit) {
				int otherQubit = coupling.qubit1 == qubit ? coupling.qubit2 : coupling.qubit1;
				connectedQubits.add(otherQubit);
			}
		}
		return connectedQubits;
	}
	/**
	 * Count the number of neighbors (i.e., directly connected qubits)
	 * of one qubit in a given qubit set.
	 * 
	 * @param qubit		a qubit index
	 * @param qubitSet	a set of qubit indices
	 * @return			the number of qubits in the set that are connected to the single qubit
	 */
	public static int nrNeighbors(int qubit, Set<Integer> qubitSet) {
		int nrNeighbors = 0;
		for (int neighborCandidate : qubitSet) {
			if (neighborCandidate != qubit) {
				if (amesConnected(qubit, neighborCandidate)) {
					++nrNeighbors;
				}
			}
		}
		return nrNeighbors;
	}
	/**
	 * Order a set of connected qubits into a chain; starting from one arbitrary end
	 * of the chain and such that consecutive qubits are connected. Attention: expects
	 * that there is at least one qubit in the set that is only connected to one other
	 * qubit and that is made the chain end.
	 * 
	 * @param qubitSet	a set of connected qubits
	 * @return			the input qubits as ordered sequence such that consecutive qubits are connected
	 */
	public static List<Integer> qubitChain(Set<Integer> qubitSet) {
		assert(qubitSet.size()>0);
		List<Integer> result = new LinkedList<Integer>();
		if (qubitSet.size()==1) {
			// Treat special case of one single qubit.
			result.add(qubitSet.iterator().next());
		} else {
			// Search one end of the chain which is a qubit that is only connected
			// to one (instead of two) other qubits.
			int chainEnd = -1;
			for (int chainEndCandidate : qubitSet) {
				if (nrNeighbors(chainEndCandidate, qubitSet) == 1) {
					chainEnd = chainEndCandidate;
					break;
				}
			}
			assert(chainEnd != -1);
			// Starting from end of chain, keep retrieving neighbor of current qubit,
			// insert into list, and continue with neighbor as current qubit.
			Set<Integer> remainingQubits = new HashSet<Integer>();
			remainingQubits.addAll(qubitSet);
			int curQubit = chainEnd;
			while (remainingQubits.size()>1) {
				result.add(curQubit);
				remainingQubits.remove(curQubit);
				curQubit = connectedQubit(curQubit, remainingQubits);
			}
			if (remainingQubits.size()>=1) {
				result.add(remainingQubits.iterator().next());
			}
		}
		return result;
	}
	/**
	 * Asserts that there is no overlap between the two sets of qubits.
	 * 
	 * @param qubitSet1	set of qubit indices
	 * @param qubitSet2	set of qubit indices
	 */
	public static void assertNoOverlap(Set<Integer> qubitSet1, Set<Integer> qubitSet2) {
		for (int qubit1 : qubitSet1) {
			assert !qubitSet2.contains(qubit1) : "Qubit " + qubit1 + " in set " + qubitSet2;
		}
	}
}
