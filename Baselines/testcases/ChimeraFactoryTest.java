package mqo_chimera.testcases;

import static org.junit.Assert.*;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.TreeSet;

import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.mapping.LogicalVariable;
import mqo_chimera.mapping.MappingType;
import mqo_chimera.mapping.VariableGroup;
import mqo_chimera.util.AmesUtil;
import mqo_chimera.util.TestUtil;

import org.junit.Test;

public class ChimeraFactoryTest {

	@Test
	public void test() throws Exception {
		AmesUtil.initAmes();
		// Getting qubits in left column
		{
			Stack<Integer> expectedQubits = new Stack<Integer>();
			for (int qubit : new int[] {0, 1, 2, 3}) {
				expectedQubits.add(qubit);
			}
			assertEquals(expectedQubits, ChimeraFactory.getLeftColon(0));
		}
		{
			Stack<Integer> expectedQubits = new Stack<Integer>();
			for (int qubit : new int[] {8, 9, 10, 11}) {
				expectedQubits.add(qubit);
			}
			assertEquals(expectedQubits, ChimeraFactory.getLeftColon(8));
		}
		// Getting qubits in right column
		{
			Stack<Integer> expectedQubits = new Stack<Integer>();
			for (int qubit : new int[] {4, 5, 6, 7}) {
				expectedQubits.add(qubit);
			}
			assertEquals(expectedQubits, ChimeraFactory.getRightColon(0));
		}
		{
			Stack<Integer> expectedQubits = new Stack<Integer>();
			for (int qubit : new int[] {12, 13, 14, 15}) {
				expectedQubits.add(qubit);
			}
			assertEquals(expectedQubits, ChimeraFactory.getRightColon(8));
		}
		// Getting groups of qubits
		{
			// Verify number of groups - on intact unit cell
			{
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(0, 1);
				assertEquals(8, groups.size());
			}
			{
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(0, 2);
				assertEquals(4, groups.size());
			}
			{
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(0, 3);
				assertEquals(2, groups.size());
			}
			{
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(0, 4);
				assertEquals(1, groups.size());
			}
			{
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(0, 5);
				assertEquals(1, groups.size());
			}
			// Verify number of groups - on unit cell with incomplete connections
			{
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(24, 1);
				assertEquals(7, groups.size());
			}
			{
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(24, 2);
				assertEquals(3, groups.size());
			}
			{
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(24, 3);
				assertEquals(1, groups.size());
			}
			{
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(24, 4);
				assertEquals(1, groups.size());
			}
			{
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(24, 5);
				assertEquals(0, groups.size());
			}
			// Verify connections
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(24, 2);
				for (VariableGroup group : groups) {
					for (LogicalVariable var1 : group.getVariables()) {
						for (LogicalVariable var2 : group.getVariables()) {
							if (var1 != var2) {
								var1.addWeight(mapping, var2, 1);
								var2.addWeight(mapping, var1, 1);
							}
						}
					}
				}
			}
			{
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				List<VariableGroup> groups = ChimeraFactory.getConnectedQubits(0, 3);
				for (VariableGroup group : groups) {
					for (LogicalVariable var1 : group.getVariables()) {
						for (LogicalVariable var2 : group.getVariables()) {
							if (var1 != var2) {
								var1.addWeight(mapping, var2, 1);
								var2.addWeight(mapping, var1, 1);
							}
						}
					}
				}
			}
		}
		// assigning variables
		{
			ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
			mapping.problem = new ChimeraMqoProblem(4, 1, true, null, null);
			ChimeraFactory.assignVariables(mapping);
			for (int query=0; query<4; ++query) {
				for (int plan=0; plan<1; ++plan) {
					LogicalVariable planVar = mapping.planVars[query][plan];
					assertEquals(1, planVar.getQubits().size());
					for (int qubit : planVar.getQubits()) {
						// Qubits must all be within first unit cell
						assertTrue(qubit < 8);
					}
				}
			}
			assertFalse(mapping.hasOverlap());
		}
		{
			ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
			mapping.problem = new ChimeraMqoProblem(3, 2, true, null, null);
			ChimeraFactory.assignVariables(mapping);
			for (int query=0; query<3; ++query) {
				for (int plan=0; plan<2; ++plan) {
					LogicalVariable planVar = mapping.planVars[query][plan];
					assertTrue(planVar.getQubits().size() >= 1 && planVar.getQubits().size() <= 2);
					for (int qubit : planVar.getQubits()) {
						// Qubits must all be within first unit cell
						assertTrue(qubit < 16);
					}
				}
			}
			assertFalse(mapping.hasOverlap());
		}
		// marking qubits
		{
			ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
			mapping.problem = new ChimeraMqoProblem(3, 4, true, null, null);
			ChimeraFactory.assignVariables(mapping);
			ChimeraFactory.markQubits(mapping);
			List<Integer> associatedQueries = new LinkedList<Integer>();
			List<Integer> associatedPlans = new LinkedList<Integer>();
			for (int qubit=0; qubit<1152; ++qubit) {
				associatedQueries.add(mapping.associatedQuery[qubit]);
				associatedPlans.add(mapping.associatedPlan[qubit]);
			}
			// Check associated queries
			assertTrue(associatedQueries.contains(-1));
			assertTrue(associatedQueries.contains(0));
			assertTrue(associatedQueries.contains(1));
			assertTrue(associatedQueries.contains(2));
			assertFalse(associatedQueries.contains(3));
			// Check associated plans
			assertTrue(associatedPlans.contains(-1));
			assertTrue(associatedPlans.contains(0));
			assertTrue(associatedPlans.contains(1));
			assertTrue(associatedPlans.contains(2));
			assertTrue(associatedPlans.contains(3));
			assertFalse(associatedPlans.contains(4));
		}
		// imposing equality constraints
		/*
		{
			for (int nrPlans=1; nrPlans<=5; ++nrPlans) {
				ChimeraMqoMapping mapping = new ChimeraMqoMapping();
				mapping.problem = new ChimeraMqoProblem(5, nrPlans, true, null, null);
				ChimeraFactory.assignVariables(mapping);
				ChimeraFactory.imposeEqualityConstraints(mapping, 0.25);
				for (int query=0; query<5; ++query) {
					for (int plan=0; plan<nrPlans; ++plan) {
						LogicalVariable var = mapping.planVars[query][plan];
						Set<Integer> qubits = var.getQubits();
						for (int qubit : qubits) {
							if (qubits.size() > 1) {
								assertEquals(0.25, mapping.getWeight(qubit), TestUtil.DOUBLE_TOLERANCE);
							} else {
								assertEquals(0, mapping.getWeight(qubit), TestUtil.DOUBLE_TOLERANCE);
							}
						}
						for (int qubit1 : qubits) {
							for (int qubit2 : qubits) {
								if (qubit1 != qubit2) {
									assertEquals(-0.5, mapping.getConnectionWeight(qubit1, qubit2), TestUtil.DOUBLE_TOLERANCE);
									assertEquals(-0.5, mapping.getConnectionWeight(qubit2, qubit1), TestUtil.DOUBLE_TOLERANCE);
								}
							}
						}
					}
				}				
			}
		}
		*/
		// impose selection constraints
		/*
		{
			for (int nrPlans=1; nrPlans<=5; ++nrPlans) {
				ChimeraMqoMapping mapping = new ChimeraMqoMapping();
				mapping.problem = new ChimeraMqoProblem(6, nrPlans, true, null, null);
				ChimeraFactory.assignVariables(mapping);
				ChimeraFactory.imposeSelectionConstraints(mapping, 0.25, true);
				for (int query=0; query<5; ++query) {
					for (int plan1=0; plan1<nrPlans; ++plan1) {
						for (int plan2=plan1+1; plan2<nrPlans; ++plan2) {
							LogicalVariable var1 = mapping.planVars[query][plan1];
							LogicalVariable var2 = mapping.planVars[query][plan2];
							assertEquals(0.25, var1.getWeight(mapping, var2), TestUtil.DOUBLE_TOLERANCE);
							assertEquals(0.25, var2.getWeight(mapping, var1), TestUtil.DOUBLE_TOLERANCE);
						}
					}
				}
			}
		}
		*/
		// picking optimal plans
		{
			// make sure that the whole plan index range might get selected
			for (boolean allowIndependentProcessing : new boolean[] {false, true}) {
				int nrQueries = 10000;
				int nrPlans = 5;
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
				int[] optimalPlans = ChimeraFactory.pickRandomPlans(mapping.problem, allowIndependentProcessing);
				assertEquals(nrQueries, optimalPlans.length);
				Set<Integer> optimalPlansSet = new TreeSet<Integer>();
				for (int i=0; i<optimalPlans.length; ++i) {
					optimalPlansSet.add(optimalPlans[i]);
				}
				for (int plan=0; plan<nrPlans; ++plan) {
					assertTrue(optimalPlansSet.contains(plan));
				}
				if (allowIndependentProcessing) {
					assertTrue(optimalPlansSet.contains(nrPlans));
				} else {
					assertFalse(optimalPlansSet.contains(nrPlans));
				}
				assertFalse(optimalPlansSet.contains(-1));
				assertFalse(optimalPlansSet.contains(nrPlans + 1));
				assertFalse(optimalPlansSet.contains(nrPlans + 2));				
			}
		}
		// inferring optimal qubit values
		{
			int nrQueries = 4;
			int nrPlans = 3;
			ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
			mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
			mapping.planVars = new LogicalVariable[nrQueries][nrPlans];
			ChimeraFactory.assignVariables(mapping);
			int[] optimalPlans = new int[] {2, 3, 0, 1};
			boolean[] optimalQubits = ChimeraFactory.inferOptimalQubitValues(optimalPlans, mapping);
			// Case: one of the plans allowing interactions is selected
			{
				Iterator<Integer> qubitIter = mapping.planVars[0][2].getQubits().iterator();
				while (qubitIter.hasNext()) {
					int qubit = qubitIter.next();
					assertTrue(optimalQubits[qubit]);
				}
			}
			{
				for (int plan : new int[] {0, 1}) {
					Iterator<Integer> qubitIter = mapping.planVars[0][plan].getQubits().iterator();
					while (qubitIter.hasNext()) {
						int qubit = qubitIter.next();
						assertFalse(optimalQubits[qubit]);
					}					
				}
			}
			// Special case: independent processing is selected
			{
				for (int plan : new int[] {0, 1, 2}) {
					Iterator<Integer> qubitIter = mapping.planVars[1][plan].getQubits().iterator();
					while (qubitIter.hasNext()) {
						int qubit = qubitIter.next();
						assertFalse(optimalQubits[qubit]);
					}					
				}
			}
		}
		// getting connected qubits associated with a different query
		{
			// Case: no suitable qubits available since only one query
			{
				int nrQueries = 1;
				for (int nrPlans=1; nrPlans<=5; ++nrPlans) {
					ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
					mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
					mapping.planVars = new LogicalVariable[nrQueries][nrPlans];
					ChimeraFactory.assignVariables(mapping);
					ChimeraFactory.markQubits(mapping);
					for (int plan=0; plan<nrPlans; ++plan) {
						LogicalVariable planVar = mapping.planVars[0][plan];
						for (int qubit : planVar.getQubits()) {
							assertEquals(-1, ChimeraFactory.getConnectedOtherQueryQubit(qubit, mapping));
						}
					}					
				}
			}
			// Case suitable qubits are certainly available
			// Make sure that obtained qubit is connected and belongs to a different plan
			for (int nrQueries=2; nrQueries<10; ++nrQueries) {
				for (int nrPlans=2; nrPlans<=5; ++nrPlans) {
					ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
					mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
					mapping.planVars = new LogicalVariable[nrQueries][nrPlans];
					ChimeraFactory.assignVariables(mapping);
					ChimeraFactory.markQubits(mapping);
					int qubit = 7;
					int connectedQubit = ChimeraFactory.getConnectedOtherQueryQubit(qubit, mapping);
					assertTrue(AmesUtil.amesConnected(qubit, connectedQubit));
					assertTrue(AmesUtil.amesConnected(connectedQubit, qubit));
					int qubitQuery = mapping.associatedQuery[qubit];
					int connectedQubitQuery = mapping.associatedQuery[connectedQubit];
					assertTrue(qubitQuery != connectedQubitQuery);
					assertTrue(qubitQuery != -1 && connectedQubitQuery != -1);
				}
			}
		}
		// getting a random plan qubit
		{
			// Make sure that each query and plan has the chance of being selected
			for (int nrQueries=2; nrQueries<10; nrQueries+=2) {
				for (int nrPlans=1; nrPlans<=5; ++nrPlans) {
					// generate and initialize mapping
					ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
					mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
					mapping.planVars = new LogicalVariable[nrQueries][nrPlans];
					ChimeraFactory.assignVariables(mapping);
					ChimeraFactory.markQubits(mapping);
					// generate set of qubits that should be eligible for random selection
					Set<Integer> qubitsToCover = new TreeSet<Integer>();
					for (int query=0; query<nrQueries; ++query) {
						for (int plan=0; plan<nrPlans; ++plan) {
							LogicalVariable planVar = mapping.planVars[query][plan];
							for (int qubit : planVar.getQubits()) {
								qubitsToCover.add(qubit);
								assertEquals(query, mapping.associatedQuery[qubit]);
							}
						}
					}
					for (int i=0; i<10000; ++i) {
						int qubit = ChimeraFactory.getRandomPlanQubit(mapping);
						qubitsToCover.remove(qubit);
					}
					assertTrue(qubitsToCover.isEmpty());
				}
			}
			
		}
		// generating loops
		{
			for (int nrQueries=2; nrQueries<10; nrQueries+=2) {
				for (int nrPlans=2; nrPlans<=5; ++nrPlans) {
					// generate and initialize mapping
					ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
					mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
					mapping.planVars = new LogicalVariable[nrQueries][nrPlans];
					ChimeraFactory.assignVariables(mapping);
					ChimeraFactory.markQubits(mapping);
					List<Integer> loop = ChimeraFactory.getAlternatingQueryLoop(mapping);
					// make sure that loop is closed
					assertEquals(loop.get(0), loop.get(loop.size()-1));
					// make sure that loop nodes are connected
					for (int i=0; i<loop.size()-1; ++i) {
						assertTrue(AmesUtil.amesConnected(loop.get(i), loop.get(i+1)));
					}
					// make sure that queries are alternating on the loop
					for (int i=0; i<loop.size()-1; ++i) {
						int qubit1 = loop.get(i);
						int qubit2 = loop.get(i+1);
						assertNotEquals(mapping.associatedQuery[qubit1], mapping.associatedQuery[qubit2]);
					}
				}
			}
		}
		// adding motivating weights
		/*
		{
			LinearSolver solver = new LinearSolver();
			for (int nrQueries=4; nrQueries<6; ++nrQueries) {
				for (int nrPlans=2; nrPlans<=5; ++nrPlans) {
					ChimeraMqoMapping mapping = new ChimeraMqoMapping();
					mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
					mapping.planVars = new LogicalVariable[nrQueries][nrPlans];
					ChimeraFactory.assignVariables(mapping);
					ChimeraFactory.markQubits(mapping);
					ChimeraFactory.imposeEqualityConstraints(mapping, 1);
					int qubit1;
					do {
						qubit1 = ChimeraFactory.getRandomPlanQubit(mapping);
					} while (ChimeraFactory.getConnectedOtherQueryQubit(qubit1, mapping) == -1);
					int qubit2 = ChimeraFactory.getConnectedOtherQueryQubit(qubit1, mapping);
					assertNotEquals(-1, qubit2);
					assertTrue(AmesUtil.amesConnected(qubit1, qubit2));
					boolean qubit1Val = RandomUtil.random.nextBoolean();
					boolean qubit2Val = RandomUtil.random.nextBoolean();
					boolean motivatedEqual = qubit1Val == qubit2Val;
					ChimeraFactory.addMotivatingWeights(qubit1, qubit2, qubit1Val, qubit2Val, mapping, 1.5);
					// generate planted solution
					boolean[] plantedQubitValues = new boolean[1152];
					int query1 = mapping.associatedQuery[qubit1];
					int plan1 = mapping.associatedPlan[qubit1];
					int query2 = mapping.associatedQuery[qubit2];
					int plan2 = mapping.associatedPlan[qubit2];
					LogicalVariable planVar1 = mapping.planVars[query1][plan1];
					LogicalVariable planVar2 = mapping.planVars[query2][plan2];
					for (int qubit : planVar1.getQubits()) {
						plantedQubitValues[qubit] = qubit1Val;
					}
					for (int qubit : planVar2.getQubits()) {
						plantedQubitValues[qubit] = qubit2Val;
					}
					mapping.problem.plantedQubitValues = plantedQubitValues;
					// calculate planted energy
					mapping.plantedEnergy = mapping.getQuboEnergy(plantedQubitValues);
					// solve
					ChimeraMqoSolution solution = solver.solveChimeraQubo(mapping);
					boolean equal = solution.qubitValues[qubit1] == solution.qubitValues[qubit2];
					assertEquals(motivatedEqual, equal);					
				}
			}
		}
		*/
		// adding de-motivating weights
		/*
		{
			LinearSolver solver = new LinearSolver();
			for (int nrQueries=4; nrQueries<6; ++nrQueries) {
				for (int nrPlans=2; nrPlans<=5; ++nrPlans) {
					ChimeraMqoMapping mapping = new ChimeraMqoMapping();
					mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
					mapping.planVars = new LogicalVariable[nrQueries][nrPlans];
					ChimeraFactory.assignVariables(mapping);
					ChimeraFactory.markQubits(mapping);
					ChimeraFactory.imposeEqualityConstraints(mapping, 1);
					int qubit1;
					do {
						qubit1 = ChimeraFactory.getRandomPlanQubit(mapping);
					} while (ChimeraFactory.getConnectedOtherQueryQubit(qubit1, mapping) == -1);
					int qubit2 = ChimeraFactory.getConnectedOtherQueryQubit(qubit1, mapping);
					assertNotEquals(-1, qubit2);
					assertTrue(AmesUtil.amesConnected(qubit1, qubit2));
					boolean qubit1Val = RandomUtil.random.nextBoolean();
					boolean qubit2Val = RandomUtil.random.nextBoolean();
					boolean motivatedEqual = qubit1Val == qubit2Val;
					ChimeraFactory.addDeMotivatingWeights(qubit1, qubit2, qubit1Val, qubit2Val, mapping, 1.5);
					// generate planted solution
					boolean[] plantedQubitValues = new boolean[1152];
					int query1 = mapping.associatedQuery[qubit1];
					int plan1 = mapping.associatedPlan[qubit1];
					int query2 = mapping.associatedQuery[qubit2];
					int plan2 = mapping.associatedPlan[qubit2];
					LogicalVariable planVar1 = mapping.planVars[query1][plan1];
					LogicalVariable planVar2 = mapping.planVars[query2][plan2];
					for (int qubit : planVar1.getQubits()) {
						plantedQubitValues[qubit] = qubit1Val;
					}
					for (int qubit : planVar2.getQubits()) {
						plantedQubitValues[qubit] = !qubit2Val;
					}
					mapping.problem.plantedQubitValues = plantedQubitValues;
					// calculate planted energy
					mapping.plantedEnergy = mapping.getQuboEnergy(plantedQubitValues);
					ChimeraMqoSolution solution = solver.solveChimeraQubo(mapping);
					boolean equal = solution.qubitValues[qubit1] == solution.qubitValues[qubit2];
					assertNotEquals(motivatedEqual, equal);					
				}
			}
		}
		*/
		// add motivating and demotivating cost values
		{
			int nrQueries = 50;
			int nrPlans = 3;
			for (int i=0; i<10; ++i) {
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
				ChimeraFactory.assignVariables(mapping);
				ChimeraFactory.markQubits(mapping);
				int qubit1 = ChimeraFactory.getRandomPlanQubit(mapping);
				int qubit2 = ChimeraFactory.getConnectedOtherQueryQubit(qubit1, mapping);
				ChimeraFactory.addMotivatingCost(qubit1, qubit2, true, true, mapping, 1);
				int query1 = mapping.associatedQuery[qubit1];
				int query2 = mapping.associatedQuery[qubit2];
				int plan1 = mapping.associatedPlan[qubit1];
				int plan2 = mapping.associatedPlan[qubit2];
				assertEquals(1, mapping.problem.planCost[query1][plan1], TestUtil.DOUBLE_TOLERANCE);
				assertEquals(1, mapping.problem.planCost[query2][plan2], TestUtil.DOUBLE_TOLERANCE);
				assertEquals(-2, mapping.problem.getInterference(query1, plan1, query2, plan2), TestUtil.DOUBLE_TOLERANCE);				
			}
		}
		{
			int nrQueries = 50;
			int nrPlans = 2;
			for (int i=0; i<10; ++i) {
				ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
				mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
				ChimeraFactory.assignVariables(mapping);
				ChimeraFactory.markQubits(mapping);
				int qubit1 = ChimeraFactory.getRandomPlanQubit(mapping);
				int qubit2 = ChimeraFactory.getConnectedOtherQueryQubit(qubit1, mapping);
				ChimeraFactory.addDeMotivatingCost(qubit1, qubit2, true, false, mapping, 1);
				int query1 = mapping.associatedQuery[qubit1];
				int query2 = mapping.associatedQuery[qubit2];
				int plan1 = mapping.associatedPlan[qubit1];
				int plan2 = mapping.associatedPlan[qubit2];
				assertEquals(1, mapping.problem.planCost[query1][plan1], TestUtil.DOUBLE_TOLERANCE);
				assertEquals(1, mapping.problem.planCost[query2][plan2], TestUtil.DOUBLE_TOLERANCE);
				assertEquals(-2, mapping.problem.getInterference(query1, plan1, query2, plan2), TestUtil.DOUBLE_TOLERANCE);				
			}
		}
		// check whether a loop is admissible
		/*
		{
			for (int nrQueries=4; nrQueries<6; ++nrQueries) {
				for (int nrPlans=2; nrPlans<=5; ++nrPlans) {
					ChimeraMqoMapping mapping = new ChimeraMqoMapping();
					mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
					mapping.planVars = new LogicalVariable[nrQueries][nrPlans];
					ChimeraFactory.assignVariables(mapping);
					ChimeraFactory.markQubits(mapping);
					List<Integer> loop = ChimeraFactory.getAlternatingQueryLoop(mapping);
					// Currently all weights are zero so loop should be admissible for margin at most 0.5
					assertTrue(ChimeraFactory.admissibleLoop(loop, mapping, 0));
					assertTrue(ChimeraFactory.admissibleLoop(loop, mapping, 0.25));
					assertTrue(ChimeraFactory.admissibleLoop(loop, mapping, 0.5));
					assertFalse(ChimeraFactory.admissibleLoop(loop, mapping, 0.6));
					assertFalse(ChimeraFactory.admissibleLoop(loop, mapping, 0.99));
					// Increasing weight on one single qubit on loop up
					{
						int loopLength = loop.size();
						int randomIndex = RandomUtil.uniformInt(0, loopLength-1);
						int loopQubit = loop.get(randomIndex);
						mapping.addWeight(loopQubit, loopQubit, 0.75);						
					}
					// Admissible for margin at most 0.25
					assertTrue(ChimeraFactory.admissibleLoop(loop, mapping, 0));
					assertTrue(ChimeraFactory.admissibleLoop(loop, mapping, 0.25));
					assertFalse(ChimeraFactory.admissibleLoop(loop, mapping, 0.5));
					assertFalse(ChimeraFactory.admissibleLoop(loop, mapping, 0.6));
					assertFalse(ChimeraFactory.admissibleLoop(loop, mapping, 0.99));
					// Decrease weight on coupling
					{
						int loopLength = loop.size();
						int randomIndex = RandomUtil.uniformInt(0, loopLength-2);
						int qubit1 = loop.get(randomIndex);
						int qubit2 = loop.get(randomIndex+1);
						mapping.addWeight(qubit1, qubit2, -0.6);						
					}
					// Admissible for margin at most 0.2
					assertTrue(ChimeraFactory.admissibleLoop(loop, mapping, 0));
					assertFalse(ChimeraFactory.admissibleLoop(loop, mapping, 0.25));
					assertFalse(ChimeraFactory.admissibleLoop(loop, mapping, 0.5));
					assertFalse(ChimeraFactory.admissibleLoop(loop, mapping, 0.6));
					assertFalse(ChimeraFactory.admissibleLoop(loop, mapping, 0.99));
				}
			}
		}
		*/
		// adding weights
		{
			for (int nrQueries=100; nrQueries<200; ++nrQueries) {
				for (int nrPlans=2; nrPlans<=2; ++nrPlans) {
					ChimeraMqoMapping mapping = new ChimeraMqoMapping(MappingType.QUBO);
					mapping.problem = new ChimeraMqoProblem(nrQueries, nrPlans, true, null, null);
					mapping.planVars = new LogicalVariable[nrQueries][nrPlans];
					ChimeraFactory.assignVariables(mapping);
					ChimeraFactory.markQubits(mapping);
					int[] optimalPlans = ChimeraFactory.pickRandomPlans(mapping.problem, true);
					boolean[] optimalQubitValues = ChimeraFactory.inferOptimalQubitValues(optimalPlans, mapping);
					ChimeraFactory.addWeights(mapping, optimalQubitValues, 10, 4, 1);
					// Make sure that weights on single qubits do not become too high/low
					for (int qubit=0; qubit<mapping.nrQubits; ++qubit) {
						assertTrue(mapping.getWeight(qubit) <= 1 && mapping.getWeight(qubit) >= -1);
					}
					// Make sure that weights on couplings do not become too high/low
					for (int qubit1=0; qubit1<mapping.nrQubits; ++qubit1) {
						for (int qubit2=0; qubit2<mapping.nrQubits; ++qubit2) {
							assertTrue(mapping.getConnectionWeight(qubit1, qubit2) <= 1);
							assertTrue(mapping.getConnectionWeight(qubit2, qubit1) <= 1);
							assertTrue(mapping.getConnectionWeight(qubit1, qubit2) >= -1);
							assertTrue(mapping.getConnectionWeight(qubit2, qubit1) >= -1);
						}
					}
					// Make sure that some weights were added
					boolean hadNonzeroWeights = false;
					for (int qubit=0; qubit<mapping.nrQubits; ++qubit) {
						//System.out.println("Qubit " + qubit + " :" + mapping.getWeight(qubit));
						if (mapping.getWeight(qubit) != 0) {
							hadNonzeroWeights = true;
						}
					}
					assertTrue(hadNonzeroWeights);
					// Make sure that some weights were added on couplings
					boolean hadNonzeroCouplings = false;
					for (int qubit1=0; qubit1<mapping.nrQubits; ++qubit1) {
						for (int qubit2=0; qubit2<mapping.nrQubits; ++qubit2) {
							if (mapping.getConnectionWeight(qubit1, qubit2) != 0) {
								hadNonzeroCouplings = true;
							}
						}
					}
					assertTrue(hadNonzeroCouplings);
				}
			}
		}
		// producing a binary

	}

}
