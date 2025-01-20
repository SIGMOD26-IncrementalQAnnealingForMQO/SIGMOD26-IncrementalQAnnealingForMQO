package mqo_chimera.mapping;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.TreeSet;

import mqo_chimera.util.AmesUtil;
import mqo_chimera.util.TestUtil;

import org.junit.Test;

public class LogicalVariableTest {

	@Test
	public void test() throws Exception {
		// Using Qubo variables requires prior initialization of Ames utils
		AmesUtil.initAmes();
		// Constructor initialization
		{
			LogicalVariable var = new LogicalVariable(123);
			assertEquals(new TreeSet<Integer>(Arrays.asList(new Integer[] {123})), var.getQubits());
		}
		// Adding qubits
		{
			LogicalVariable var = new LogicalVariable(123);
			var.addQubit(0);
			var.addQubit(4);
			assertEquals(new TreeSet<Integer>(Arrays.asList(new Integer[] {123, 0, 4})), var.getQubits());
		}
		{
			LogicalVariable var = new LogicalVariable();
			var.addQubit(1151);
			var.addQubit(4);
			assertEquals(new TreeSet<Integer>(Arrays.asList(new Integer[] {1151, 4})), var.getQubits());
		}
		// adding weights on single variable and getting them back
		{
			// if variable is represented by only one qubit
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				LogicalVariable var = new LogicalVariable(0);
				assertEquals(0, var.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
				var.addWeight(mapping, 2);
				assertEquals(2, var.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
				var.addWeight(mapping, 1);
				assertEquals(3, var.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
			}
			// if variable is represented by multiple qubits
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				LogicalVariable var = new LogicalVariable(4);
				var.addQubit(1);
				var.addQubit(5);
				assertEquals(0, var.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
				var.addWeight(mapping, -1.5);
				assertEquals(-1.5, var.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
				var.addWeight(mapping, 0.5);
				assertEquals(-1, var.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
			}
		}
		// adding and getting weights between variables
		{
			// if variable is represented by only one qubit
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				LogicalVariable var1 = new LogicalVariable(1075);
				LogicalVariable var2 = new LogicalVariable(1078);
				var1.addWeight(mapping, var2, 3);
				assertEquals(3, var1.getWeight(mapping, var2), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(3, var2.getWeight(mapping, var1), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, var1.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, var2.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
				var2.addWeight(mapping, var1, 0.5);
				assertEquals(3.5, var1.getWeight(mapping, var2), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(3.5, var2.getWeight(mapping, var1), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, var1.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, var2.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
			}
			// if variable is represented by multiple qubits
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				LogicalVariable var1 = new LogicalVariable(1);
				var1.addQubit(1075);
				LogicalVariable var2 = new LogicalVariable(1078);
				var2.addQubit(123);
				var1.addWeight(mapping, var2, -3);
				assertEquals(-3, var1.getWeight(mapping, var2), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(-3, var2.getWeight(mapping, var1), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, var1.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, var2.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
				var2.addWeight(mapping, var1, 0.5);
				assertEquals(-2.5, var1.getWeight(mapping, var2), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(-2.5, var2.getWeight(mapping, var1), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, var1.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, var2.getWeight(mapping), TestUtil.DOUBLE_TOLERANCE);
			}
		}
		// adding equality constraints
		{
			// Variable represented by one qubit
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				LogicalVariable var1 = new LogicalVariable(1);
				var1.addEquality(mapping, 2);
				assertEquals(0, mapping.getWeight(0), TestUtil.DOUBLE_TOLERANCE);
			}
			// Variable represented by two qubits
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				LogicalVariable var1 = new LogicalVariable(1);
				var1.addQubit(4);
				var1.addEquality(mapping, 1);
				assertEquals(1, mapping.getWeight(1), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(1, mapping.getWeight(4), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(-2, mapping.getConnectionWeight(1, 4), TestUtil.DOUBLE_TOLERANCE);
			}
			// Variable represented by three qubits
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				LogicalVariable var1 = new LogicalVariable(1);
				var1.addQubit(4);
				var1.addQubit(0);
				var1.addEquality(mapping, 2);
				assertEquals(2, mapping.getWeight(1), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(4, mapping.getWeight(4), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(2, mapping.getWeight(0), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(-4, mapping.getConnectionWeight(0, 4), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(-4, mapping.getConnectionWeight(4, 0), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(-4, mapping.getConnectionWeight(1, 4), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(-4, mapping.getConnectionWeight(4, 1), TestUtil.DOUBLE_TOLERANCE);
			}
		}
	}

}
