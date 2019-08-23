package edu.ucla.belief.ace;
import java.io.*;
import java.util.concurrent.ThreadLocalRandom;
import org.apache.commons.math3.complex.*;

/**
 * An example of using the Ace evaluator API.  The files simple.net.lmap and
 * simple.net.ac must have been compiled using the SOP kind and must be in the
 * directory from which this program is executed.
 * <code>
 * Usage java edu.ucla.belief.Test
 * </code>
 *
 * @author Mark Chavira
 */

public class Test {

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
    // System.out.print("qubitCount = ");
    // System.out.println(qubitCount);

    // Obtain some objects representing variables in the network.  We are not
    // creating network variables here, just retrieving them by name from the
    // OnlineEngine.
    for (int qubit=0; qubit<qubitCount; qubit++) {
      String qubitName = String.format("q%01dn0000", qubit);
      int varForQubit = g.varForName(qubitName);
      // System.out.print(qubitName);
      // System.out.println(varForQubit);
    }

    int[] qubitFinalToVar = new int[qubitCount];
    for (int qubit=0; qubit<qubitCount; qubit++) {
      // System.out.print("Finding final node for qubit ");
      // System.out.println(qubit);

      boolean found = false;
      for (int varForQubit=g.numVariables()-1; !found; varForQubit--) {
        // System.out.print("varForQubit=");
        // System.out.println(varForQubit);
        String qubitName = g.nameForVar(varForQubit);
        // System.out.print("qubitName=");
        // System.out.println(qubitName);
        if (qubitName.startsWith(String.format("q%01d", qubit))) {
        // if (qubitName.endsWith(String.format("qubit%04d", qubit))) {
          qubitFinalToVar[qubit] = varForQubit;
          found = true;
          // System.out.println("found = true;");
        }
      }
    }

    try {
      while( g.readLiteralMap(lmReader, OnlineEngine.CompileKind.ALWAYS_SUM) != null ){

        BufferedWriter csv = new BufferedWriter(new FileWriter(g.hash_csv+".buf"));

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

          // Complex[][] potPartials = new Complex[g.numPotentials()][];
          // Complex[][] potMarginals = new Complex[g.numPotentials()][];
          // Complex[][] potPosteriors = new Complex[g.numPotentials()][];
          // for (int pot = 0; pot < g.numPotentials(); ++pot) {
          //   potPartials[pot] = g.potPartials(pot);
          //   potMarginals[pot] = g.potMarginals(pot);
          //   potPosteriors[pot] = g.potPosteriors(pot);
          // }

          // Finally, write the results to out file.
          if (amplitude.getImaginary()<0) {
            // System.out.println(outputQubitString+","+amplitude.getReal()+""+amplitude.getImaginary()+"j"+","+probabilitySum+","+evidenceDuration+","+evaluationDuration);
            csv.write(outputQubitString+","+amplitude.getReal()+""+amplitude.getImaginary()+"j"+","+probabilitySum+","+evidenceDuration+","+evaluationDuration);
          } else {
            // System.out.println(outputQubitString+","+amplitude.getReal()+"+"+amplitude.getImaginary()+"j"+","+probabilitySum+","+evidenceDuration+","+evaluationDuration);
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

          // for (int p = 0; p < g.numPotentials(); ++p) {
          //   System.out.println(
          //       "(PD wrt " + g.nameForPot(p) + ")(evidence) = " +
          //       Arrays.toString(potPartials[p]));
          // }
          // for (int p = 0; p < g.numPotentials(); ++p) {
          //   System.out.println(
          //       "Pr(" + g.nameForPot(p) + ", evidence) = " +
          //       Arrays.toString(potMarginals[p]));
          // }
          // for (int p = 0; p < g.numPotentials(); ++p) {
          //   System.out.println(
          //       "Pr(" + g.nameForPot(p) + " | evidence) = " +
          //       Arrays.toString(potPosteriors[p]));
          // }

        }

        csv.close();
        File oldfile = new File (g.hash_csv+".buf");
    		File newfile = new File (g.hash_csv+".csv");
    		oldfile.renameTo(newfile);
      }
    } catch(IOException e) {
        e.printStackTrace();
    } finally {
        lmReader.close();
    }
  }
}
