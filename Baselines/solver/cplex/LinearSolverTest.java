package mqo_chimera.solver.cplex;

import static org.junit.Assert.*;
import mqo_chimera.benchmark.TestcaseClass;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.testcases.ChimeraFactory;
import mqo_chimera.testcases.ChimeraMqoProblem;
import mqo_chimera.testcases.MqoSolution;
import mqo_chimera.testcases.QuadraticMqoSolution;
import mqo_chimera.util.AmesUtil;
import mqo_chimera.util.RandomUtil;
import mqo_chimera.util.TestUtil;

import org.junit.Test;

public class LinearSolverTest {

	@Test
	public void test() throws Exception {
		// Initializations
		AmesUtil.initAmes();
		LinearSolver solver = new LinearSolver();
		// No interactions - cheaper plans must be selected
		/*
		{
			ChimeraMqoMapping mapping = new ChimeraMqoMapping();
			mapping.problem = new ChimeraMqoProblem(2, 1, true, null, null);
			mapping.planVars = new LogicalVariable[2][1];
			mapping.planVars[0][0] = new LogicalVariable(0);
			mapping.planVars[1][0] = new LogicalVariable(1);
			mapping.addWeight(0, 0, 2.5);
			mapping.addWeight(1, 1, -0.5);
			ChimeraMqoSolution solution = solver.solve(mapping);
			assertEquals(-0.5, solution.objectiveValue, TestUtil.DOUBLE_TOLERANCE);
			assertEquals(0, solution.doubleValues[0], TestUtil.DOUBLE_TOLERANCE);
			assertEquals(1, solution.doubleValues[1], TestUtil.DOUBLE_TOLERANCE);
			assertEquals(false, solution.qubitValues[0]);
			assertEquals(true, solution.qubitValues[1]);
			assertEquals(false, solution.planVarValues[0][0]);
			assertEquals(true, solution.planVarValues[1][0]);
		}
		// With interactions - select both plans to save by interaction
		{
			ChimeraMqoMapping mapping = new ChimeraMqoMapping();
			mapping.problem = new ChimeraMqoProblem(2, 1, true, null, null);
			mapping.planVars = new LogicalVariable[2][1];
			mapping.planVars[0][0] = new LogicalVariable(0);
			mapping.planVars[1][0] = new LogicalVariable(4);
			mapping.addWeight(0, 0, 0.5);
			mapping.addWeight(4, 4, 0.5);
			mapping.addWeight(0, 4, -2);
			ChimeraMqoSolution solution = solver.solve(mapping);
			assertEquals(-1, solution.objectiveValue, TestUtil.DOUBLE_TOLERANCE);
			assertEquals(1, solution.doubleValues[0], TestUtil.DOUBLE_TOLERANCE);
			assertEquals(1, solution.doubleValues[4], TestUtil.DOUBLE_TOLERANCE);
			assertEquals(true, solution.qubitValues[0]);
			assertEquals(true, solution.qubitValues[4]);
			assertEquals(true, solution.planVarValues[0][0]);
			assertEquals(true, solution.planVarValues[1][0]);
		}
		*/
		// Check consistency between QUBO and MQO solver
		{
			for (int i=0; i<10; ++i) {
				int nrQueries = RandomUtil.uniformInt(2, 50);
				int nrPlans = RandomUtil.uniformInt(2, 5);
				TestcaseClass testClass = new TestcaseClass(nrQueries, nrPlans, false);
				ChimeraMqoMapping quboMapping = ChimeraFactory.produceStandardTestcase(testClass);
				ChimeraMqoProblem problem = quboMapping.problem;
				QuadraticMqoSolution quboSolution = solver.solveChimeraQubo(quboMapping);
				MqoSolution mqoSolution = solver.solve(quboMapping.problem);
				int[] quboSelections = quboSolution.planSelections;
				int[] mqoSelections = mqoSolution.planSelections;
				double quboCost = problem.executionCost(quboSelections);
				double mqoCost = problem.executionCost(mqoSelections);
				assertEquals(quboCost, mqoCost, TestUtil.DOUBLE_TOLERANCE);
			}
		}
		// Check consistency between Qubo and Ising solver
		{
			for (int i=0; i<10; ++i) {
				int nrQueries = RandomUtil.uniformInt(2, 50);
				int nrPlans = RandomUtil.uniformInt(2, 5);
				TestcaseClass testClass = new TestcaseClass(nrQueries, nrPlans, false);
				ChimeraMqoMapping quboMapping = ChimeraFactory.produceStandardTestcase(testClass);
				ChimeraMqoMapping isingMapping = quboMapping.toIsing();
				QuadraticMqoSolution quboSolution = solver.solveChimeraQubo(quboMapping);
				QuadraticMqoSolution isingSolution = solver.solveChimeraIsing(isingMapping);
				double quboInQuboEnergy = quboSolution.objectiveValue;
				double isingInIsingEnergy = isingSolution.objectiveValue;
				System.out.println("Objective value of QUBO solution in QUBO problem: " + quboInQuboEnergy);
				System.out.println("Objective value of Ising solution in Ising problem: " + isingInIsingEnergy);
				boolean[] quboValues = quboSolution.qubitValues;
				boolean[] isingValues = isingSolution.qubitValues;
				double quboInIsingEnergy = isingMapping.getEnergy(quboValues);
				double isingInQuboEnergy = quboMapping.getEnergy(isingValues);
				System.out.println("Objective value of QUBO solution in Ising problem: " + quboInIsingEnergy);
				System.out.println("Objective value of Ising solution in QUBO problem: " + isingInQuboEnergy);
				assertEquals(quboInQuboEnergy, isingInQuboEnergy, TestUtil.DOUBLE_TOLERANCE);
				assertEquals(isingInIsingEnergy, quboInIsingEnergy, TestUtil.DOUBLE_TOLERANCE);
			}
		}

	}

}
