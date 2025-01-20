package mqo_chimera.util;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import mqo_chimera.mapping.LogicalVariable;

import org.junit.Test;

public class AmesUtilTest {

	@Test
	public void test() throws Exception {
		// Initialization
		{
			AmesUtil.initAmes();
			// Check qubit indices
			assertTrue(AmesUtil.amesQubits.contains(25));
			assertTrue(AmesUtil.amesQubits.contains(30));
			assertFalse(AmesUtil.amesQubits.contains(26));
			assertFalse(AmesUtil.amesQubits.contains(1096));
			// Check couplings
			{
				Coupling coupling = new Coupling(0, 4);
				assertTrue(AmesUtil.amesCouplings.contains(coupling));
			}
			{
				Coupling coupling = new Coupling(4, 0);
				assertTrue(AmesUtil.amesCouplings.contains(coupling));
			}
			{
				Coupling coupling = new Coupling(522, 524);
				assertTrue(AmesUtil.amesCouplings.contains(coupling));
			}
			{
				Coupling coupling = new Coupling(711, 719);
				assertTrue(AmesUtil.amesCouplings.contains(coupling));
			}
			{
				Coupling coupling = new Coupling(0, 9);
				assertFalse(AmesUtil.amesCouplings.contains(coupling));
			}
			{
				Coupling coupling = new Coupling(73,80);
				assertFalse(AmesUtil.amesCouplings.contains(coupling));
			}
			// Check nr. qubits
			assertEquals(1151, AmesUtil.highestQubitIndex);
		}
		// Connections between qubits
		{
			assertTrue(AmesUtil.amesConnected(0, 5));
			assertTrue(AmesUtil.amesConnected(1147, 1151));
			assertTrue(AmesUtil.amesConnected(0, 4));
			assertFalse(AmesUtil.amesConnected(11, 16));
			assertFalse(AmesUtil.amesConnected(1002, 1009));
			assertFalse(AmesUtil.amesConnected(1, 24));
		}
		// Connections between variables
		{
			// Variables represented by one qubit
			{
				LogicalVariable var1 = new LogicalVariable(0);
				LogicalVariable var2 = new LogicalVariable(4);
				assertTrue(AmesUtil.amesConnected(var1, var2));
			}
			{
				LogicalVariable var1 = new LogicalVariable(241);
				LogicalVariable var2 = new LogicalVariable(245);
				assertTrue(AmesUtil.amesConnected(var1, var2));
			}
			{
				LogicalVariable var1 = new LogicalVariable(0);
				LogicalVariable var2 = new LogicalVariable(10);
				assertFalse(AmesUtil.amesConnected(var1, var2));
			}
			{
				LogicalVariable var1 = new LogicalVariable(101);
				LogicalVariable var2 = new LogicalVariable(120);
				assertFalse(AmesUtil.amesConnected(var1, var2));
			}
			// Variables represented by multiple qubits
			{
				LogicalVariable var1 = new LogicalVariable(0);
				LogicalVariable var2 = new LogicalVariable(4);
				var1.addQubit(1);
				assertTrue(AmesUtil.amesConnected(var1, var2));
			}
			{
				LogicalVariable var1 = new LogicalVariable(241);
				LogicalVariable var2 = new LogicalVariable(320);
				var2.addQubit(242);
				assertFalse(AmesUtil.amesConnected(var1, var2));
			}
			{
				LogicalVariable var1 = new LogicalVariable(0);
				LogicalVariable var2 = new LogicalVariable(109);
				var1.addQubit(1);
				var2.addQubit(1000);
				assertFalse(AmesUtil.amesConnected(var1, var2));
			}
			{
				LogicalVariable var1 = new LogicalVariable(0);
				LogicalVariable var2 = new LogicalVariable(109);
				var1.addQubit(1);
				var2.addQubit(1000);
				var1.addQubit(1005);
				assertTrue(AmesUtil.amesConnected(var1, var2));
			}
		}
		// Finding one connected qubit
		{
			{
				assertEquals(4, AmesUtil.connectedQubit(0, new TreeSet<Integer>(
						Arrays.asList(new Integer[] {4, 34, 1100}))));
			}
			{
				int connectedQubit = AmesUtil.connectedQubit(0, new TreeSet<Integer>(
						Arrays.asList(new Integer[] {4, 5, 1100})));
				assertTrue(connectedQubit == 4 || connectedQubit == 5);
			}
		}
		// Counting the number of connected qubits
		{
			assertEquals(2, AmesUtil.nrNeighbors(0, new TreeSet<Integer>(
						Arrays.asList(new Integer[] {4, 5, 1100}))));
			assertEquals(0, AmesUtil.nrNeighbors(0, new TreeSet<Integer>(
					Arrays.asList(new Integer[] {421, 125, 1100}))));
			assertEquals(3, AmesUtil.nrNeighbors(996, new TreeSet<Integer>(
					Arrays.asList(new Integer[] {988, 992, 993}))));
		}
		// Creating qubit chains
		{
			Set<Integer> qubits = new TreeSet<Integer>(
				Arrays.asList(new Integer[] {0, 1, 4})
			);
			List<Integer> chain = AmesUtil.qubitChain(qubits);
			// Verify that chain contains the right indices
			assertTrue(chain.contains(0));
			assertTrue(chain.contains(1));
			assertTrue(chain.contains(4));
			assertEquals(3, chain.size());
			// Verify that chain contains the qubits in an admissible order
			assertTrue(AmesUtil.amesConnected(chain.get(0), chain.get(1)));
			assertTrue(AmesUtil.amesConnected(chain.get(2), chain.get(1)));
		}
		{
			Set<Integer> qubits = new TreeSet<Integer>(
				Arrays.asList(new Integer[] {1146, 1149, 1147, 1150, 1143, 1135, 1151})
			);
			List<Integer> chain = AmesUtil.qubitChain(qubits);
			// Verify that chain contains the right indices
			for (int qubit : qubits) {
				assertTrue(chain.contains(qubit));
			}
			assertEquals(qubits.size(), chain.size());
			// Verify that chain contains the qubits in an admissible order
			for (int i=0; i<4; ++i) {
				assertTrue(AmesUtil.amesConnected(chain.get(i), chain.get(i+1)));
			}
		}
	}
}
