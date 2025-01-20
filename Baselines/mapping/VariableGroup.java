package mqo_chimera.mapping;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;

import mqo_chimera.util.AmesUtil;

/**
 * Aggregates a group of QUBO variables.
 * 
 * @author immanueltrummer
 *
 */
public class VariableGroup implements Serializable {
	/**
	 * Used to verify class version.
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * A list of Qubo variables forming this group.
	 */
	private List<LogicalVariable> variables = new LinkedList<LogicalVariable>();
	/**
	 * Adds a new variable to this group after veryfying that its qubits do not
	 * overlap with the ones of any variable alraedy in the set.
	 * 
	 * @param var	new variable to add
	 */
	public void addVariable(LogicalVariable var) {
		// Make sure that the qubits of different variables do not overlap
		for (LogicalVariable oldVar : variables) {
			AmesUtil.assertNoOverlap(var.getQubits(), oldVar.getQubits());
		}
		// Add new variable
		variables.add(var);
	}
	/**
	 * Returns all variables in this group.
	 * 
	 * @return	list of Qubo variables
	 */
	public List<LogicalVariable> getVariables() {
		return variables;
	}
}
