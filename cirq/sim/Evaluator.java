package edu.ucla.belief.ace;
import java.io.*;
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

  static int qubitCount;

  public static void main(String[] args) throws Exception {

    BufferedReader lmReader = new BufferedReader(new InputStreamReader(System.in));

    // Create the online inference engine.  An OnlineEngine reads from disk
    // a literal map and an arithmetic circuit compiled for a network.
    OnlineEngineSop g = new OnlineEngineSop(
        args[0],
        args[1],
        false);
    qubitCount = Integer.parseInt(args[2]);

    // Obtain some objects representing variables in the network.  We are not
    // creating network variables here, just retrieving them by name from the
    // OnlineEngine.
    // for (int qubit=0; qubit<qubitCount; qubit++) {
    //   String qubitName = String.format("q%04dn0000", qubit);
    //   int varForQubit = g.varForName(qubitName);
    //   System.out.print(qubitName);
    //   System.out.println(varForQubit);
    // }

    try {
      while( g.readLiteralMap(lmReader, OnlineEngine.CompileKind.ALWAYS_SUM) != null ){

        int[] qubitFinalToVar = new int[qubitCount];
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

        BufferedWriter csv = new BufferedWriter(new FileWriter(g.basename+".buf"));

        // Construct evidence.
        Evidence evidence = new Evidence(g);

        // long outputQubitString = ThreadLocalRandom.current().nextLong( 1L<<qubitCount );

        // System.out.println("outputQubitString,amplitude,probabilitySum,evidenceDuration,evaluationDuration");
        // csv.write("outputQubitString,amplitude,probabilitySum,evidenceDuration,evaluationDuration");
        // csv.newLine();
        long evidenceDuration = 0L;
        long evaluationDuration = 0L;
        double probabilitySum = 0.0;
        for (long outputQubitString=0; outputQubitString<1L<<qubitCount; outputQubitString++) {

          long evidenceStart = System.nanoTime();
          // int randomQubit = ThreadLocalRandom.current().nextInt(qubitCount);
          // outputQubitString ^= 1L << randomQubit;
          // int varForQubit = qubitFinalToVar[randomQubit];
          // evidence.varCommit(varForQubit, ((int)(outputQubitString>>randomQubit))&1);
          for (int qubit=0; qubit<qubitCount; qubit++) {
            int varForQubit = qubitFinalToVar[qubit];
            // String qubitName = g.nameForVar(varForQubit);
            // to adhere to Cirq's endian convention:
            evidence.varCommit(varForQubit, ((int)(outputQubitString>>(qubitCount-qubit-1)))&1);
            // System.out.println(qubitName);
            // System.out.println(varForQubit);
          }
          evidenceDuration += (System.nanoTime() - evidenceStart);  //divide by 1000000 to get milliseconds.

          // Perform online inference in the context of the evidence set by
          // invoking OnlineEngine.evaluate().  Doing so will compute probability of
          // evidence.  Inference runs in time that is linear in the size of the
          // arithmetic circuit.

          long evaluationStart = System.nanoTime();
          g.evaluate(evidence);
          evaluationDuration += (System.nanoTime() - evaluationStart);  //divide by 1000000 to get milliseconds.

          // This time, also differentiate.  Answers to many additional queries then
          // become available.  Inference time will still be linear in the size of the
          // arithmetic circuit, but the constant factor will be larger.

          // g.differentiate();

          // Now retrieve the result of inference.  The following method invocation
          // performs no inference, simply looking up the requested value that was
          // computed by OnlineEngine.evaluate().

          Complex amplitude = g.evaluationResults();
          double probability = amplitude.abs() * amplitude.abs();
          probabilitySum += probability;
          // double pN = probability * outputQubitString;

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

          // Finally, write the results to out file.
          if (amplitude.getImaginary()<0) {
            csv.write(outputQubitString+","+amplitude.getReal()+""+amplitude.getImaginary()+"j"+","+probabilitySum+","+evidenceDuration+","+evaluationDuration);
          } else {
            csv.write(outputQubitString+","+amplitude.getReal()+"+"+amplitude.getImaginary()+"j"+","+probabilitySum+","+evidenceDuration+","+evaluationDuration);
          }
          csv.newLine();

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
}
