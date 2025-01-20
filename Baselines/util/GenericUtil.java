package mqo_chimera.util;

public class GenericUtil {
	/**
	 * Transforms the double values assigned to the qubits into Boolean values.
	 * 
	 * @return	a Boolean vector containing value assignments for each qubit
	 */
	public static boolean[] extractQubitValues(double[] doubleValues) {
		int nrValues = doubleValues.length;
		boolean[] qubitValues = new boolean[nrValues];
		for (int i=0; i<nrValues; ++i) {
			qubitValues[i] = doubleValues[i] > 0.5 ? true : false;
		}
		return qubitValues;
	}
}
