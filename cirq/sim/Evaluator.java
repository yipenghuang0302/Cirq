package edu.ucla.belief.ace;
import java.lang.Double;
import java.io.*;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import org.apache.commons.math3.complex.*;

/**
 * An example of using the Ace evaluator API.  The files simple.net.lmap and
 * simple.net.ac must have been compiled using the SOP kind and must be in the
 * directory from which this program is executed.
 * <code>
 * Usage java edu.ucla.belief.Evaluator
 * </code>
 *
 * @author Mark Chavira
 */

public class Evaluator {

  /**
   * The main program.
   *
   * @param args command line parameters - ignored.
   * @throws Exception if execution fails.
   */

  static OnlineEngineSop g;
  static int qubitCount;
  static int[] qubitFinalToVar;
  static Evidence evidence;
  static long evidenceDuration;
  static long evaluationDuration;
  static BufferedWriter csv;

  public static void main(String[] args) throws Exception {

    BufferedReader lmReader = new BufferedReader(new InputStreamReader(System.in));

    // Create the online inference engine.  An OnlineEngine reads from disk
    // a literal map and an arithmetic circuit compiled for a network.
    g = new OnlineEngineSop(
        args[0],
        args[1],
        true);
    qubitCount = Integer.parseInt(args[2]);

    // Obtain some objects representing variables in the network.  We are not
    // creating network variables here, just retrieving them by name from the
    // OnlineEngine.

    try {
      while( g.readLiteralMap(lmReader, OnlineEngine.CompileKind.ALWAYS_SUM) != null ){

        qubitFinalToVar = new int[qubitCount];
        for (int qubit=0; qubit<qubitCount; qubit++) {
          for (int trial=g.moment; ; trial--) {
            String qubitName = String.format("n%04dq%04d", trial, qubit);
            try {
              int varForQubit = g.varForName(qubitName);
              qubitFinalToVar[qubit] = varForQubit;
              break;
            } catch (Exception e) {
            }
          }
        }

        // Construct evidence.
        evidence = new Evidence(g);
        evidenceDuration = 0L;
        evaluationDuration = 0L;
        csv = new BufferedWriter(new FileWriter(g.basename+".buf"));
        // csv.write("outputQubitString,amplitude,evidenceDuration,evaluationDuration");
        // csv.newLine();

        if ( g.repetitions!=0 ) {
          long markov = ThreadLocalRandom.current().nextLong( 1L<<qubitCount );
          for ( int iter=0; iter<64; iter++ ) { // warmup
            // System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
            // System.out.println("markov = " + markov);
            Complex markovAmplitude = findAmplitude(markov, false);
            // System.out.println("probability = " + markovAmplitude.abs() * markovAmplitude.abs());
            markov = findDerivatives(markov);
            // System.out.println("findDerivatives = " + markov);
          }
          for ( int iter=0; iter<g.repetitions; iter++ ) {
            // System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
            // System.out.println("markov = " + markov);
            Complex markovAmplitude = findAmplitude(markov, true);
            // System.out.println("probability = " + markovAmplitude.abs() * markovAmplitude.abs());
            markov = findDerivatives(markov);
            // System.out.println("findDerivatives = " + markov);
          }
        } else if (!g.bitstrings.isEmpty()) {
          for (int outputQubitString: g.bitstrings) {
            Complex amplitude = findAmplitude(outputQubitString, true);
          }
        } else {
          double probabilitySum = 0.0;
          for (long outputQubitString=0; outputQubitString<1L<<qubitCount; outputQubitString++) {
            Complex amplitude = findAmplitude(outputQubitString, true);
            probabilitySum += amplitude.abs() * amplitude.abs();
          }
          assert Math.abs(probabilitySum-1.0) < 1.0/65536.0;
        }

        csv.close();
        File oldfile = new File (g.basename+".buf");
        File newfile = new File (g.basename+".csv");
        oldfile.renameTo(newfile);
      }
    } catch(IOException e) {
        e.printStackTrace();
    } finally {
        lmReader.close();
    }
  }

  private static Complex findAmplitude (
    long outputQubitString,
    boolean print
  ) throws Exception {

    long evidenceStart = System.nanoTime();
    for (int qubit=0; qubit<qubitCount; qubit++) {
      int varForQubit = qubitFinalToVar[qubit];
      // to adhere to Cirq's endian convention:
      evidence.varCommit(varForQubit, ((int)(outputQubitString>>(qubitCount-qubit-1)))&1);
    }
    evidenceDuration += System.nanoTime() - evidenceStart;  //divide by 1000000 to get milliseconds.

    // Perform online inference in the context of the evidence set by
    // invoking OnlineEngine.evaluate().  Doing so will compute probability of
    // evidence.  Inference runs in time that is linear in the size of the
    // arithmetic circuit.

    long evaluationStart = System.nanoTime();
    g.evaluate(evidence);
    evaluationDuration += System.nanoTime() - evaluationStart;  //divide by 1000000 to get milliseconds.

    // Now retrieve the result of inference.  The following method invocation
    // performs no inference, simply looking up the requested value that was
    // computed by OnlineEngine.evaluate().

    Complex amplitude = g.evaluationResults();

    // Finally, write the results to out file.
    if (print) {
      if ( amplitude.getImaginary()<0 ) {
        csv.write(outputQubitString+","+amplitude.getReal()+""+amplitude.getImaginary()+"j"+","+evidenceDuration+","+evaluationDuration);
      } else if ( amplitude.getImaginary()==0 ) {
        csv.write(outputQubitString+","+amplitude.getReal()+","+evidenceDuration+","+evaluationDuration);
      } else {
        csv.write(outputQubitString+","+amplitude.getReal()+"+"+amplitude.getImaginary()+"j"+","+evidenceDuration+","+evaluationDuration);
      }
      csv.newLine();
    }

    return amplitude;
  }

  private static long findDerivatives (
    long outputQubitString
  ) throws Exception {

    // This time, also differentiate.  Answers to many additional queries then
    // become available.  Inference time will still be linear in the size of the
    // arithmetic circuit, but the constant factor will be larger.

    g.differentiate();

    // Once again retrieve results without performing inference.  We get
    // probability of evidence, derivatives, marginals, and posterior marginals
    // for both variables and potentials.  OnlineEngine.variables() returns an
    // unmodifiable set of all network variables and OnlineEngine.potentials()
    // returns an unmodifiable set of all network potentials.

    // Complex[][] varPartials = new Complex[g.numVariables()][];
    // Complex[][] varMarginals = new Complex[g.numVariables()][];
    // Complex[][] varPosteriors = new Complex[g.numVariables()][];
    // for (int v = 0; v < g.numVariables(); ++v) {
    //   varPartials[v] = g.varPartials(v);
    //   varMarginals[v] = g.varMarginals(v);
    //   varPosteriors[v] = g.varPosteriors(v);
    // }

    int randomQubit = ThreadLocalRandom.current().nextInt(qubitCount);
    // System.out.println("randomQubit = " + randomQubit);
    int varForQubit = qubitFinalToVar[randomQubit];

    double partial_0 = g.varPartials(varForQubit)[0].abs() * g.varPartials(varForQubit)[0].abs();
    double partial_1 = g.varPartials(varForQubit)[1].abs() * g.varPartials(varForQubit)[1].abs();
    double probability = partial_1/(partial_0+partial_1);
    // System.out.println("probability = " + probability);
    if (Double.isNaN(probability)) {
      outputQubitString = ThreadLocalRandom.current().nextLong( 1L<<qubitCount );
      // throw new Exception("Gibbs sampling transition probability is NaN.");
    } else {
      if ( ThreadLocalRandom.current().nextDouble() <= probability ) {
          // outputQubitString |=  (1L << randomQubit);
          outputQubitString |=  (1L << (qubitCount-randomQubit-1));
      } else {
          // outputQubitString &= ~(1L << randomQubit);
          outputQubitString &= ~(1L << (qubitCount-randomQubit-1));
      }
    }

    // for (int qubit=0; qubit<qubitCount; qubit++) {
    //   varForQubit = qubitFinalToVar[qubit];
    //   System.out.println(
    //       "(PD wrt " + g.nameForVar(varForQubit) + ")(evidence) =" +
    //       " |0>:" + g.varPartials(varForQubit)[0].abs() * g.varPartials(varForQubit)[0].abs() +
    //       " |1>:" + g.varPartials(varForQubit)[1].abs() * g.varPartials(varForQubit)[1].abs()
    //       );
    // }

    // for (int v = 0; v < g.numVariables(); ++v) {
    //   System.out.println(
    //       "(PD wrt " + g.nameForVar(v) + ")(evidence) = " +
    //       Arrays.toString(varPartials[v]));
    // }
    // for (int v = 0; v < g.numVariables(); ++v) {
    //   System.out.println(
    //       "Pr(" + g.nameForVar(v) + ", evidence) = " +
    //       Arrays.toString(varMarginals[v]));
    // }
    // for (int v = 0; v < g.numVariables(); ++v) {
    //   System.out.println(
    //       "Pr(" + g.nameForVar(v) + " | evidence) = " +
    //       Arrays.toString(varPosteriors[v]));
    // }

    return outputQubitString;
  }

}
