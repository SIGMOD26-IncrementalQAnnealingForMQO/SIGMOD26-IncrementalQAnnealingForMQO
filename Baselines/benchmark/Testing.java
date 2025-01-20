package mqo_chimera.benchmark;

import java.util.Arrays;

import mqo_chimera.solver.climbing.HillClimber;
import mqo_chimera.solver.cplex.LinearSolver;
import mqo_chimera.solver.genetic.GeneticSolver;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.testcases.ChimeraFactory;
import mqo_chimera.testcases.MqoSolution;
import mqo_chimera.util.AmesUtil;

public class Testing {

	public static void main(String[] args) throws Exception {
		AmesUtil.initAmes();
		ChimeraMqoMapping mapping = ChimeraFactory.produceStandardTestcase(new TestcaseClass(537, 2, false));
		LinearSolver linearSolver = new LinearSolver();
		GeneticSolver geneticSolver = new GeneticSolver(100);
		HillClimber climber = new HillClimber();
		climber.solve(mapping.problem);
		MqoSolution linearSolution = linearSolver.solve(mapping.problem);
		geneticSolver.solve(mapping.problem);
		System.out.println("Linear cost decrease: " + Arrays.toString(linearSolver.lastRunCheckpointCost));
		System.out.println("Genetic cost decrease: " + Arrays.toString(geneticSolver.lastRunCheckpointCost));
		System.out.println("Climber cost decrease: " + Arrays.toString(climber.lastRunCheckpointCost));
		System.out.println("Linear final millis: " + linearSolver.lastRunSolverMillis);
		System.out.println("Linear solution: " + mapping.problem.executionCost(linearSolution.planSelections));
	}

}
