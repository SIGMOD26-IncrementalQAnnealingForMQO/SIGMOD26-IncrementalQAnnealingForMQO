package mqo_chimera.mapping;

import static org.junit.Assert.*;

import org.junit.Test;

public class VariableGroupTest {

	@Test
	public void test() {
		 // Adding and getting variables
		VariableGroup group = new VariableGroup();
		LogicalVariable var = new LogicalVariable(1);
		group.addVariable(var);
		assertEquals(1, group.getVariables().size());
		assertTrue((group.getVariables().iterator()).next().getQubits().contains(1));
		assertEquals(1, (group.getVariables().iterator()).next().getQubits().size());
	}

}
