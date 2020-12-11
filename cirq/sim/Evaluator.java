package edu.ucla.belief.ace;
import java.lang.Double;
import java.io.*;
import java.util.*;
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

  static short qubitCount;
  static int[] qubitFinalToVar;
  static Map<Integer,Integer> varToQubitFinal;

  static short noiseCount;
  static int[] noiseRVToVar;
  static Map<Integer,Integer> varToNoiseRV;

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
    qubitCount = Short.parseShort(args[2]);
    noiseCount = Short.parseShort(args[3]);

    // Obtain some objects representing variables in the network.  We are not
    // creating network variables here, just retrieving them by name from the
    // OnlineEngine.

    try {
      while( g.readLiteralMap(lmReader, OnlineEngine.CompileKind.ALWAYS_SUM) != null ){

        qubitFinalToVar = new int[qubitCount];
        varToQubitFinal = new HashMap<Integer,Integer>();
        for (int qubit=0; qubit<qubitCount; qubit++) {
          for (int trial=g.moment; ; trial--) {
            String qubitName = String.format("n%04dq%04d", trial, qubit);
            try {
              int varForQubit = g.varForName(qubitName);
              qubitFinalToVar[qubit] = varForQubit;
              varToQubitFinal.put(varForQubit,qubit);
              break;
            } catch (Exception e) {
            }
          }
        }

        noiseRVToVar = new int[noiseCount];
        varToNoiseRV = new HashMap<Integer,Integer>();
        int noiseRVIndex = 0;
        for (int var = 0; var < g.numVariables(); var++) {
          if (g.nameForVar(var).startsWith("rv")) {
            noiseRVToVar[noiseRVIndex++] = var;
            varToNoiseRV.put(var,noiseRVIndex);
          }
        }
        assert noiseRVIndex == noiseCount;
        for (noiseRVIndex=0; noiseRVIndex<noiseCount; noiseRVIndex++) {
          // System.out.println("noiseCount = " + noiseCount);
          // System.out.println(g.nameForVar(noiseRVToVar[noiseRVIndex]));
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

          // for ( int iter=0; iter<g.repetitions; iter++ ) {
          //
          //   // PROGRESSIVE SAMPLING APPROACH
          //
          //   evidence.retractAll();
          //
          //   // for (short qubitProgress = 0; qubitProgress<qubitCount; qubitProgress++) {
          //   //   int varForQubit = qubitFinalToVar[qubitProgress];
          //   //   if (g.fSrcVarToSrcValToIndicator[varForQubit][0]>0)
          //   //     g.logicVarToElimOp[g.fSrcVarToSrcValToIndicator[varForQubit][0]] = OnlineEngine.ElimOp.INVALID;
          //   //   g.logicVarToElimOp[g.fSrcVarToSrcValToIndicator[varForQubit][1]] = OnlineEngine.ElimOp.INVALID;
          //   // }
          //
          //   byte[] noiseArray = new byte[noiseCount];
          //
          //   // for (short qubitProgress = 0; qubitProgress<qubitCount; qubitProgress++) {
          //   for (short noiseProgress = (short)(noiseCount-1); 0<=noiseProgress; noiseProgress--) {
          //
          //     int varForNoise = noiseRVToVar[noiseProgress];
          //
          //     evidence.varCommit(varForNoise, 0);
          //     g.evaluate(evidence);
          //     Complex amplitude_0 = g.evaluationResults();
          //     double partial_0 = amplitude_0.abs() * amplitude_0.abs();
          //     // System.out.println("partial_0 = " + partial_0);
          //
          //     evidence.varCommit(varForNoise, 1);
          //     g.evaluate(evidence);
          //     Complex amplitude_1 = g.evaluationResults();
          //     double partial_1 = amplitude_1.abs() * amplitude_1.abs();
          //     // System.out.println("partial_1 = " + partial_1);
          //
          //     evidence.varCommit(varForNoise, 2);
          //     g.evaluate(evidence);
          //     Complex amplitude_2 = g.evaluationResults();
          //     double partial_2 = amplitude_2.abs() * amplitude_2.abs();
          //     // System.out.println("partial_2 = " + partial_2);
          //
          //     evidence.varCommit(varForNoise, 3);
          //     g.evaluate(evidence);
          //     Complex amplitude_3 = g.evaluationResults();
          //     double partial_3 = amplitude_3.abs() * amplitude_3.abs();
          //     // System.out.println("partial_3 = " + partial_3);
          //
          //     if (partial_0+partial_1+partial_2+partial_3==0.0) {
          //       throw new Exception("Progressive sampling probability is NaN.");
          //     }
          //
          //     double prob_0 = partial_0/(partial_0+partial_1+partial_2+partial_3);
          //     double prob_1 = partial_1/(partial_0+partial_1+partial_2+partial_3);
          //     double prob_2 = partial_2/(partial_0+partial_1+partial_2+partial_3);
          //     double prob_3 = partial_3/(partial_0+partial_1+partial_2+partial_3);
          //
          //     double rand = ThreadLocalRandom.current().nextDouble();
          //
          //     if ( rand <= prob_0 ) {
          //       // System.out.println("evidence.varCommit(varForNoise, 0);");
          //       evidence.varCommit(varForNoise, 0);
          //       noiseArray[noiseProgress] = 0;
          //     } else if ( rand <= prob_0+prob_1 ) {
          //       // System.out.println("evidence.varCommit(varForNoise, 1);");
          //       evidence.varCommit(varForNoise, 1);
          //       noiseArray[noiseProgress] = 1;
          //     } else if ( rand <= prob_0+prob_1+prob_2 ) {
          //       // System.out.println("evidence.varCommit(varForNoise, 2);");
          //       evidence.varCommit(varForNoise, 2);
          //       noiseArray[noiseProgress] = 2;
          //     } else {
          //       // System.out.println("evidence.varCommit(varForNoise, 3);");
          //       evidence.varCommit(varForNoise, 3);
          //       noiseArray[noiseProgress] = 3;
          //     }
          //   }
          //
          //   long outputQubitString = 0L;
          //
          //   // for (short qubitProgress = 0; qubitProgress<qubitCount; qubitProgress++) {
          //   for (short qubitProgress = (short)(qubitCount-1); 0<=qubitProgress; qubitProgress--) {
          //
          //     int varForQubit = qubitFinalToVar[qubitProgress];
          //     // if (g.fSrcVarToSrcValToIndicator[varForQubit][0]>0)
          //     //   g.logicVarToElimOp[g.fSrcVarToSrcValToIndicator[varForQubit][0]] = OnlineEngine.ElimOp.ADD;
          //     // g.logicVarToElimOp[g.fSrcVarToSrcValToIndicator[varForQubit][1]] = OnlineEngine.ElimOp.ADD;
          //
          //     evidence.varCommit(varForQubit, 0);
          //     g.evaluate(evidence);
          //     Complex amplitude_0 = g.evaluationResults();
          //     double partial_0 = amplitude_0.abs() * amplitude_0.abs();
          //     // System.out.println("partial_0 = " + partial_0);
          //
          //     evidence.varCommit(varForQubit, 1);
          //     g.evaluate(evidence);
          //     Complex amplitude_1 = g.evaluationResults();
          //     double partial_1 = amplitude_1.abs() * amplitude_1.abs();
          //     // System.out.println("partial_1 = " + partial_1);
          //
          //     double probability = partial_1/(partial_0+partial_1);
          //     // System.out.println("probability = " + probability);
          //
          //     if (Double.isNaN(probability)) {
          //       outputQubitString = ThreadLocalRandom.current().nextLong( 1L<<qubitCount );
          //       throw new Exception("Progressive sampling probability is NaN.");
          //     } else {
          //       if ( ThreadLocalRandom.current().nextDouble() <= probability ) {
          //         evidence.varCommit(varForQubit, 1);
          //         outputQubitString |=  (1L << (qubitCount-qubitProgress-1));
          //       } else {
          //         evidence.varCommit(varForQubit, 0);
          //         outputQubitString &= ~(1L << (qubitCount-qubitProgress-1));
          //       }
          //     }
          //
          //   }
          //
          //   // System.out.println(noiseArray+","+outputQubitString);
          //   csv.write(noiseArray+","+outputQubitString);
          //   csv.newLine();
          //
          // }

          // DIRECT SAMPLING APPROACH

          //   byte[] noiseArray = new byte[noiseCount];
          //   g.evaluate(evidence);
          //   // System.out.println("g.evaluationResults()=");
          //   // System.out.println(g.evaluationResults());
          //   List<Integer> list = g.fCalculator.dftSample(
          //       g.numAcNodes(),
          //       g.fNodeToType,
          //       g.logicVarToElimOp,
          //       g.fNodeToLit,
          //       g.fNodeToLastEdge,
          //       g.fEdgeToTailNode,
          //       evidence
          //   );
          //   long outputQubitString = 0L;
          //   for (int l : list) {
          //     int var = (l < 0 ? g.fAcVarToNegSrcVar : g.fAcVarToPosSrcVar)[Math.abs(l)];
          //     if (var!=-1) {
          //
          //       int val = (l < 0 ? g.fAcVarToNegSrcVal : g.fAcVarToPosSrcVal)[Math.abs(l)];
          //       // System.out.print("val=");
          //       // System.out.println(val);
          //
          //       // System.out.print("g.fSrcVarToName[var]=");
          //       // System.out.println(g.fSrcVarToName[var]);
          //
          //       if (varToQubitFinal.containsKey(var)) {
          //         int qubit = varToQubitFinal.get(var);
          //         outputQubitString += val<<(qubitCount-qubit-1);
          //       } else if (varToNoiseRV.containsKey(var)) {
          //       } else {
          //         System.out.println("Variable is neither final qubit state nor noise RV.");
          //         // throw new Exception("Variable is neither qubit nor noise RV.");
          //       }
          //
          //     }
          //   }
          //   // System.out.println(noiseArray+","+outputQubitString);
          //   csv.write(noiseArray+","+outputQubitString);
          //   csv.newLine();
          //
          // }

          // MCMC APPROACH

          byte[] noiseArray = new byte[noiseCount];
          for (short noise=0; noise<noiseCount; noise++) {
            noiseArray[noise] = (byte) ThreadLocalRandom.current().nextInt(4);
          }
          long outputQubitString = ThreadLocalRandom.current().nextLong( 1L<<qubitCount );

          for ( int iter=0; iter<Math.max(16,noiseCount); iter++ ) { // warmup
            // System.out.println("noiseArray="+Arrays.toString(noiseArray)+" outputQubitString"+outputQubitString);
            Complex markovAmplitude = findAmplitude(noiseArray, outputQubitString, false);
            if ( noiseCount!=0 ) {
              noiseArray = findDerivativesNoise(noiseArray);
              markovAmplitude = findAmplitude(noiseArray, outputQubitString, false);
            }
            outputQubitString = findDerivativesQubit(outputQubitString);
          }

          for ( int iter=0; iter<g.repetitions; iter++ ) {
            // System.out.println("noiseArray="+Arrays.toString(noiseArray)+" outputQubitString"+outputQubitString);

            long amplitudeStart = System.nanoTime();
            Complex markovAmplitude = findAmplitude(noiseArray, outputQubitString, true);
            amplitudeDuration += System.nanoTime()-amplitudeStart;

            if ( noiseCount!=0 ) {

              long derivativesStart = System.nanoTime();
              noiseArray = findDerivativesNoise(noiseArray);
              derivativesDuration += System.nanoTime()-derivativesStart;

              amplitudeStart = System.nanoTime();
              markovAmplitude = findAmplitude(noiseArray, outputQubitString, false);
              amplitudeDuration += System.nanoTime()-amplitudeStart;

            }

            long derivativesStart = System.nanoTime();
            outputQubitString = findDerivativesQubit(outputQubitString);
            derivativesDuration += System.nanoTime()-derivativesStart;

          }
          // System.out.println( String.format("   evidence time=%16d",evidenceDuration) );
          // System.out.println( String.format(" evaluation time=%16d",evaluationDuration) );
          // System.out.println( String.format("  amplitude time=%16d",amplitudeDuration) );
          // System.out.println( String.format("derivatives time=%16d",derivativesDuration) );

        } else if (!g.bitstrings.isEmpty()) {
          for (int outputQubitString: g.bitstrings) {
            Complex amplitude = findAmplitude(new byte[0], outputQubitString, true); // TODO: enable noise
          }
        } else {
          for (long noiseString=0; noiseString<1L<<(2*noiseCount); noiseString++) {

            byte[] noiseArray = new byte[noiseCount];
            for (short noise=0; noise<noiseCount; noise++) {
              noiseArray[noise] = (byte)( (noiseString>>(2*noise))&3 ) ;
            }

            double probabilitySum = 0.0;
            for (long outputQubitString=0; outputQubitString<1L<<qubitCount; outputQubitString++) {
              Complex amplitude = findAmplitude(noiseArray, outputQubitString, true);
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

  static byte[] prevNoiseArray;
  static long prevQubitString = Long.MAX_VALUE;
  static Complex amplitude;

  private static Complex findAmplitude (
    byte[] noiseArray,
    long outputQubitString,
    boolean print
  ) throws Exception {

    long evidenceStart = System.nanoTime();
    for (int noise=0; noise<noiseCount; noise++) {
      int varForNoise = noiseRVToVar[noise];
      evidence.varCommit(varForNoise, noiseArray[noise]);
    }
    for (int qubit=0; qubit<qubitCount; qubit++) {
      int varForQubit = qubitFinalToVar[qubit];
      // to adhere to Cirq's endian convention:
      evidence.varCommit(varForQubit, ((int)(outputQubitString>>(qubitCount-qubit-1)))&1);
    }
    evidenceDuration += System.nanoTime() - evidenceStart;  //divide by 1000000 to get milliseconds.

    long evaluationStart = System.nanoTime();
    if ( outputQubitString!=prevQubitString || !Arrays.equals(noiseArray,prevNoiseArray) ) {
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
    prevNoiseArray = noiseArray.clone();
    prevQubitString = outputQubitString;
    evaluationDuration += System.nanoTime() - evaluationStart;  //divide by 1000000 to get milliseconds.

    // Finally, write the results to out file.
    if (print) {
      if ( amplitude.getImaginary()<0 ) {
        csv.write(noiseArray+","+outputQubitString+","+amplitude.getReal()+""+amplitude.getImaginary()+"j"+","+evidenceDuration+","+evaluationDuration);
      } else if ( amplitude.getImaginary()==0 ) {
        csv.write(noiseArray+","+outputQubitString+","+amplitude.getReal()+","+evidenceDuration+","+evaluationDuration);
      } else {
        csv.write(noiseArray+","+outputQubitString+","+amplitude.getReal()+"+"+amplitude.getImaginary()+"j"+","+evidenceDuration+","+evaluationDuration);
      }
      csv.newLine();
    }

    return amplitude;
  }

  private static byte[] findDerivativesNoise (
    byte[] noiseArray
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

    int randomNoise = ThreadLocalRandom.current().nextInt(noiseCount);
    // System.out.println("noiseCount = " + noiseCount);
    // System.out.println("randomNoise = " + randomNoise);
    int varForNoise = noiseRVToVar[randomNoise];
    // System.out.println("varForNoise = " + varForNoise);

    Complex[] varPartials = g.varPartials(varForNoise);
    double partial_0 = varPartials[0].abs() * varPartials[0].abs();
    double partial_1 = varPartials[1].abs() * varPartials[1].abs();
    double partial_2 = varPartials[2].abs() * varPartials[2].abs();
    double partial_3 = varPartials[3].abs() * varPartials[3].abs();

    double prob_0 = partial_0/(partial_0+partial_1+partial_2+partial_3);
    double prob_1 = partial_1/(partial_0+partial_1+partial_2+partial_3);
    double prob_2 = partial_2/(partial_0+partial_1+partial_2+partial_3);
    double prob_3 = partial_3/(partial_0+partial_1+partial_2+partial_3);

    // if (partial_0+partial_1+partial_2+partial_3==0.0)
    //   throw new Exception("noise Gibbs sampling transition probability is NaN.");

    double rand = ThreadLocalRandom.current().nextDouble();
    if ( rand <= prob_0 ) {
      noiseArray[randomNoise] = 0;
    } else if ( rand <= prob_0+prob_1 ) {
      noiseArray[randomNoise] = 1;
    } else if ( rand <= prob_0+prob_1+prob_2 ) {
      noiseArray[randomNoise] = 2;
    } else {
      noiseArray[randomNoise] = 3;
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

    return noiseArray;
  }

  private static long findDerivativesQubit (
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
    // System.out.println("qubitCount = " + qubitCount);
    // System.out.println("randomQubit = " + randomQubit);
    int varForQubit = qubitFinalToVar[randomQubit];
    // System.out.println("varForQubit = " + varForQubit);

    Complex[] varPartials = g.varPartials(varForQubit);
    // evidence.varCommit(varForQubit, 0);
    // g.evaluate(evidence);
    // Complex amplitude_0 = g.evaluationResults();
    // double partial_0 = amplitude_0.abs() * amplitude_0.abs();
    double partial_0 = varPartials[0].abs() * varPartials[0].abs();

    // evidence.varCommit(varForQubit, 1);
    // g.evaluate(evidence);
    // Complex amplitude_1 = g.evaluationResults();
    // double partial_1 = amplitude_1.abs() * amplitude_1.abs();
    double partial_1 = varPartials[1].abs() * varPartials[1].abs();

    double probability = partial_1/(partial_0+partial_1);
    // System.out.println("partial_0 = " + partial_0);
    // System.out.println("partial_1 = " + partial_1);
    // System.out.println("probability = " + probability);
    if (Double.isNaN(probability)) {
      outputQubitString = ThreadLocalRandom.current().nextLong( 1L<<qubitCount );
      // throw new Exception("qubit Gibbs sampling transition probability is NaN.");
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
