package mqo_chimera;

import com.fasterxml.jackson.databind.ObjectMapper;
import ilog.concert.IloException;
import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.benchmark.TestcaseClass;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.solver.Solver;
import mqo_chimera.solver.climbing.HillClimber;
import mqo_chimera.solver.cplex.LinearSolver;
import mqo_chimera.solver.genetic.GeneticSolver;
import mqo_chimera.testcases.ChimeraFactory;
import mqo_chimera.testcases.ChimeraMqoProblem;
import mqo_chimera.testcases.MqoSolution;
import org.jgap.Gene;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.Array;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import java.util.zip.GZIPInputStream;


public class App 
{

    public static void main( String[] args )
    {
        
    }
}
