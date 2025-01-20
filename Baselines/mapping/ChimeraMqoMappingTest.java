package mqo_chimera.mapping;

import static org.junit.Assert.*;

import java.util.Arrays;

import mqo_chimera.testcases.ChimeraFactory;
import mqo_chimera.testcases.ChimeraMqoProblem;
import mqo_chimera.util.AmesUtil;
import mqo_chimera.util.RandomUtil;
import mqo_chimera.util.TestUtil;

import org.junit.Test;

public class ChimeraMqoMappingTest {

	@Test
	public void test() throws Exception {
		AmesUtil.initAmes();
		{
			// Adding weights on single qubits
			ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
			mapping.addWeight(9, 9, 2);
			mapping.addWeight(9, 9, 1);
			assertEquals(3, mapping.getWeight(9), TestUtil.DOUBLE_TOLERANCE);
			// Adding weights between qubits
			mapping.addWeight(192, 196, 1);
			mapping.addWeight(196, 192, 2);
			assertEquals(3, mapping.getConnectionWeight(192, 196), TestUtil.DOUBLE_TOLERANCE);
			// Counting the number of used couplings
			assertEquals(1, mapping.nrCouplingsUsed());
			mapping.addWeight(192, 96, -5);
			assertEquals(-5, mapping.getConnectionWeight(96, 192), TestUtil.DOUBLE_TOLERANCE);
			assertEquals(2, mapping.nrCouplingsUsed());			
		}
		// Test of overlap
		{
			ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
			mapping.problem = new ChimeraMqoProblem(2, 1, true, null, null);
			mapping.planVars = new LogicalVariable[2][1];
			mapping.planVars[0][0] = new LogicalVariable(1);
			mapping.planVars[1][0] = new LogicalVariable(1);
			assertTrue(mapping.hasOverlap());
		}
		{
			ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
			mapping.problem = new ChimeraMqoProblem(2, 1, true, null, null);
			mapping.planVars = new LogicalVariable[2][1];
			mapping.planVars[0][0] = new LogicalVariable(196);
			mapping.planVars[1][0] = new LogicalVariable(4);
			assertFalse(mapping.hasOverlap());
		}
		// Test calculating energy for Qubo
		{
			// weights on single qubits
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				mapping.addWeight(4, 4, 1.5);
				boolean[] qubitValues = new boolean[1152];
				Arrays.fill(qubitValues, false);
				assertEquals(0, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				qubitValues[4] = true;
				assertEquals(1.5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				mapping.addWeight(1143, 1143, -5);
				assertEquals(1.5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				qubitValues[1143] = true;
				assertEquals(-3.5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
			}
			// weights between qubits
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				mapping.addWeight(0, 4, 2.5);
				boolean[] qubitValues = new boolean[1152];
				Arrays.fill(qubitValues, false);
				assertEquals(0, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				qubitValues[0] = true;
				assertEquals(0, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				qubitValues[4] = true;
				assertEquals(2.5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				mapping.addWeight(5, 0, -3);
				assertEquals(2.5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				qubitValues[5] = true;
				assertEquals(-0.5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
			}
		}
		// Test calculating energy for Ising problem
		{
			// weights on single spins
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.ISING);
				boolean[] qubitValues = new boolean[1152];
				Arrays.fill(qubitValues, false);
				assertEquals(0, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				mapping.addWeight(4, 4, 1.5);
				assertEquals(-1.5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				qubitValues[4] = true;
				assertEquals(1.5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				mapping.addWeight(1143, 1143, -5);
				assertEquals(1.5 + 5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				qubitValues[1143] = true;
				assertEquals(1.5 - 5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
			}
			// weights between spins
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.ISING);
				boolean[] qubitValues = new boolean[1152];
				Arrays.fill(qubitValues, false);
				assertEquals(0, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				mapping.addWeight(0, 4, 2.5);
				assertEquals(2.5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				qubitValues[0] = true;
				assertEquals(-2.5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				qubitValues[4] = true;
				assertEquals(2.5, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				mapping.addWeight(5, 0, 3);
				assertEquals(2.5-3, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
				qubitValues[5] = true;
				assertEquals(2.5+3, mapping.getEnergy(qubitValues), TestUtil.DOUBLE_TOLERANCE);
			}
		}
		// Test transformation into Ising problem
		{
			{
				ChimeraMqoMapping quboMapping = new ChimeraMqoMapping(MappingType.QUBO);
				quboMapping.problem = new ChimeraMqoProblem(2, 1, true);
				quboMapping.addWeight(0, 4, 1);
				ChimeraMqoMapping isingMapping = quboMapping.toIsing();
				assertEquals(0.25, isingMapping.getWeight(0), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0.25, isingMapping.getWeight(4), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0.25, isingMapping.getConnectionWeight(4, 0), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(-0.25, isingMapping.plantedEnergy, TestUtil.DOUBLE_TOLERANCE);
			}			
			for (int i=0; i<10; ++i) {
				final int nrQueries = RandomUtil.uniformInt(150, 200);
				final int nrPlansPerQuery = RandomUtil.uniformInt(1, 3);
				final int nrLoops = 50;
				final int minLoopLength = 5;
				final int weightLevels = 2;
				final boolean allowIndependentProcessing = true;
				// Generate QUBO representation of MQO problem with planted solution
				ChimeraMqoMapping quboMapping = ChimeraFactory.produceLoopsTestcase(nrQueries, nrPlansPerQuery, 
						nrLoops, minLoopLength, weightLevels, allowIndependentProcessing);
				// Transform into Ising problem
				ChimeraMqoMapping isingMapping = quboMapping.toIsing();
				// Make sure that formerly optimal solution is still optimal
				boolean[] optimalQubitValues = quboMapping.problem.plantedQubitValues;
				double optimalIsingEnergy = isingMapping.plantedEnergy;
				assertEquals(optimalIsingEnergy, isingMapping.getEnergy(optimalQubitValues), 
						TestUtil.DOUBLE_TOLERANCE);
			}
		}
		// Test calculating maximal absolute weight in diagonal
		{
			ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
			mapping.problem = new ChimeraMqoProblem(2, 1, true);
			assertEquals(0, mapping.maxAbsDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(0, 0, 1);
			assertEquals(1, mapping.maxAbsDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(0, 0, -2);
			assertEquals(1, mapping.maxAbsDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(0, 0, -2);
			assertEquals(3, mapping.maxAbsDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(0, 4, 10);
			assertEquals(3, mapping.maxAbsDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(4, 4, 10);
			assertEquals(10, mapping.maxAbsDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
		}
		// Test calculating maximal absolute weight outside diagonal
		{
			ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
			mapping.problem = new ChimeraMqoProblem(2, 1, true);
			assertEquals(0, mapping.maxAbsNonDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(0, 0, 1);
			assertEquals(0, mapping.maxAbsNonDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(0, 0, -2);
			assertEquals(0, mapping.maxAbsNonDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(0, 0, -2);
			assertEquals(0, mapping.maxAbsNonDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(0, 4, 1);
			assertEquals(1, mapping.maxAbsNonDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(4, 4, 10);
			assertEquals(1, mapping.maxAbsNonDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(4, 0, 1.5);
			assertEquals(2.5, mapping.maxAbsNonDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(1143, 1151, -3);
			assertEquals(3, mapping.maxAbsNonDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
			mapping.addWeight(1143, 1151, -3);
			assertEquals(6, mapping.maxAbsNonDiagonalWeight(), TestUtil.DOUBLE_TOLERANCE);
		}
		// Testing weight range test
		{
			ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.ISING);
			mapping.problem = new ChimeraMqoProblem(2, 1, true);
			assertEquals(true, mapping.respectsIsingRanges());
			mapping.addWeight(0, 4, 1);
			assertEquals(true, mapping.respectsIsingRanges());
			mapping.addWeight(0, 5, 1);
			assertEquals(true, mapping.respectsIsingRanges());
			mapping.addWeight(0, 6, 1);
			assertEquals(true, mapping.respectsIsingRanges());
			mapping.addWeight(0, 0, 2.5);
			assertEquals(false, mapping.respectsIsingRanges());
			mapping.addWeight(0, 0, -0.5);
			assertEquals(true, mapping.respectsIsingRanges());
			mapping.addWeight(0, 4, 0.5);
			assertEquals(false, mapping.respectsIsingRanges());
			mapping.addWeight(0, 4, -0.5);
			assertEquals(true, mapping.respectsIsingRanges());
			mapping.addWeight(1143, 1151, -1.5);
			assertEquals(false, mapping.respectsIsingRanges());
		}
	}

}
