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
  static int noiseCount;
  static int[] noiseRVToVar;
  static Evidence evidence;
  static long evidenceDuration;
  static long evaluationDuration;
  static long amplitudeDuration;
  static long derivativesDuration;
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
    noiseCount = Integer.parseInt(args[3]);

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

        noiseRVToVar = new int[noiseCount];
        int noiseRVIndex = 0;
        for (int var = 0; var < g.numVariables(); var++) {
          if (g.nameForVar(var).endsWith("rv")) {
            noiseRVToVar[noiseRVIndex++] = var;
          }
        }
        assert noiseRVIndex == noiseCount;
        for (noiseRVIndex=0; noiseRVIndex<noiseCount; noiseRVIndex++) {
          System.out.println(g.nameForVar(noiseRVToVar[noiseRVIndex]));
        }

        // Construct evidence.
        evidence = new Evidence(g);
        evidenceDuration = 0L;
        evaluationDuration = 0L;
        amplitudeDuration = 0L;
        derivativesDuration = 0L;
        csv = new BufferedWriter(new FileWriter(g.basename+".buf"));
        // csv.write("outputQubitString,amplitude,evidenceDuration,evaluationDuration");
        // csv.newLine();

        if ( g.repetitions!=0 ) {
          long outputQubitString = ThreadLocalRandom.current().nextLong( 1L<<qubitCount );
          for ( int iter=0; iter<32; iter++ ) { // warmup
            Complex markovAmplitude = findAmplitude(0, outputQubitString, false); // TODO: enable noise
            outputQubitString = findDerivatives(outputQubitString);
          }
          for ( int iter=0; iter<g.repetitions; iter++ ) {
            long amplitudeStart = System.nanoTime();
            Complex markovAmplitude = findAmplitude(0, outputQubitString, true);  // TODO: enable noise
            amplitudeDuration += System.nanoTime()-amplitudeStart;

            long derivativesStart = System.nanoTime();
            outputQubitString = findDerivatives(outputQubitString);
            derivativesDuration += System.nanoTime()-derivativesStart;
          }
          // System.out.println( String.format("   evidence time=%16d",evidenceDuration) );
          // System.out.println( String.format(" evaluation time=%16d",evaluationDuration) );
          System.out.println( String.format("  amplitude time=%16d",amplitudeDuration) );
          System.out.println( String.format("derivatives time=%16d",derivativesDuration) );
        } else if (!g.bitstrings.isEmpty()) {
          for (int outputQubitString: g.bitstrings) {
            Complex amplitude = findAmplitude(0, outputQubitString, true); // TODO: enable noise
          }
        } else {
          for (long noiseString=0; noiseString<1L<<noiseCount; noiseString++) {
            double probabilitySum = 0.0;
            for (long outputQubitString=0; outputQubitString<1L<<qubitCount; outputQubitString++) {
              Complex amplitude = findAmplitude(noiseString, outputQubitString, true);
              probabilitySum += amplitude.abs() * amplitude.abs();
            }
            assert Math.abs(probabilitySum-1.0) < 1.0/65536.0;
          }
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

  static long prevQubitString = Long.MAX_VALUE;
  static Complex amplitude;
  private static Complex findAmplitude (
    long noiseString,
    long outputQubitString,
    boolean print
  ) throws Exception {

    // long evidenceStart = System.nanoTime();
    for (int noise=0; noise<noiseCount; noise++) {
      int varForNoise = noiseRVToVar[noise];
      // to adhere to Cirq's endian convention:
      evidence.varCommit(varForNoise, ((int)(noiseString>>(noiseCount-noise-1)))&1);
    }
    for (int qubit=0; qubit<qubitCount; qubit++) {
      int varForQubit = qubitFinalToVar[qubit];
      // to adhere to Cirq's endian convention:
      evidence.varCommit(varForQubit, ((int)(outputQubitString>>(qubitCount-qubit-1)))&1);
    }
    // evidenceDuration += System.nanoTime() - evidenceStart;  //divide by 1000000 to get milliseconds.

    // long evaluationStart = System.nanoTime();
    if (outputQubitString!=prevQubitString) {
      // Perform online inference in the context of the evidence set by
      // invoking OnlineEngine.evaluate().  Doing so will compute probability of
      // evidence.  Inference runs in time that is linear in the size of the
      // arithmetic circuit.
      g.evaluate(evidence);

      // Now retrieve the result of inference.  The following method invocation
      // performs no inference, simply looking up the requested value that was
      // computed by OnlineEngine.evaluate().
      amplitude = g.evaluationResults();
    }
    prevQubitString = outputQubitString;
    // evaluationDuration += System.nanoTime() - evaluationStart;  //divide by 1000000 to get milliseconds.

    // Finally, write the results to out file.
    if (print) {
      if ( amplitude.getImaginary()<0 ) {
        // csv.write(outputQubitString+","+amplitude.getReal()+""+amplitude.getImaginary()+"j"+","+evidenceDuration+","+evaluationDuration);
        csv.write(noiseString+","+outputQubitString+","+amplitude.getReal()+""+amplitude.getImaginary()+"j"+","+evidenceDuration+","+evaluationDuration);
      } else if ( amplitude.getImaginary()==0 ) {
        // csv.write(outputQubitString+","+amplitude.getReal()+","+evidenceDuration+","+evaluationDuration);
        csv.write(noiseString+","+outputQubitString+","+amplitude.getReal()+","+evidenceDuration+","+evaluationDuration);
      } else {
        // csv.write(outputQubitString+","+amplitude.getReal()+"+"+amplitude.getImaginary()+"j"+","+evidenceDuration+","+evaluationDuration);
        csv.write(noiseString+","+outputQubitString+","+amplitude.getReal()+"+"+amplitude.getImaginary()+"j"+","+evidenceDuration+","+evaluationDuration);
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
    if (!g.differentiateResultsAvailable()) {
      g.differentiate();
    }

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

    // evidence.varCommit(varForQubit, 0);
    // g.evaluate(evidence);
    // Complex amplitude_0 = g.evaluationResults();
    double partial_0 = g.varPartials(varForQubit)[0].abs() * g.varPartials(varForQubit)[0].abs();
    // double partial_0 = amplitude_0.abs() * amplitude_0.abs();

    // evidence.varCommit(varForQubit, 1);
    // g.evaluate(evidence);
    // Complex amplitude_1 = g.evaluationResults();
    double partial_1 = g.varPartials(varForQubit)[1].abs() * g.varPartials(varForQubit)[1].abs();
    // double partial_1 = amplitude_1.abs() * amplitude_1.abs();

    double probability = partial_1/(partial_0+partial_1);
    // System.out.println("probability = " + probability);
    if (Double.isNaN(probability)) {
      outputQubitString = ThreadLocalRandom.current().nextLong( 1L<<qubitCount );
      // throw new Exception("Gibbs sampling transition probability is NaN.");
    } else {
      if ( ThreadLocalRandom.current().nextDouble() <= probability ) {
          outputQubitString |=  (1L << (qubitCount-randomQubit-1));
      } else {
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
