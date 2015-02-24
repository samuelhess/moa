/*
 *    OFSP.java
 *    Copyright (C) 2015 Drexel University, Philadelphia, PA
 *    @author Gregory Ditzler (gregory{[dot]}ditzler{[at]}gmail{[dot]}com)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.featureselection;

import weka.core.Instance;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;

import java.util.Random;

/**
 * Online Feature Selection with Partial Inputs
 * 
 * @author Gregory Ditzler (gregory{[dot]}ditzler{[at]}gmail{[dot]}com)
 */
public class OFSP extends AbstractClassifier {

    /**
     * Serial ID
     */
    private static final long serialVersionUID = 1L;
    
    /**
     * Truncation size or the number of features to select. This is set up in 
     * the GUI menu
     */
    public IntOption numSelectOption = new IntOption("numSelect",
            'n', "The number of features to select.",
            10, 0, Integer.MAX_VALUE);
    
    /**
     * Step size
     */
    public FloatOption stepSizeOption = new FloatOption("stepSize",
            's', "The step size.",
            0.2, 0.00, Integer.MAX_VALUE);
    
    /**
     * Exploration parameter
     */
    public FloatOption searchOption = new FloatOption("search",
            'e', "Exploration parameter.",
            0.2, 0.00, Integer.MAX_VALUE);
    
    /**
     * Sets the L2-norm bound
     */
    public FloatOption boundOption = new FloatOption("bound",
            'b', "Bound on the l2-norm.",
            10, 0.00, Integer.MAX_VALUE);
    
    /**
     * Select the evaluation option: Full or Partial
     */
    public MultiChoiceOption evalOption = new MultiChoiceOption("evaluation",
            'p', "Evaluate on a full or partial information instance.",
            new String[]{"full", "partial"}, new String[]{"full", "partial"}, 0);
    
    /**
     * Parameter vector for prediction
     */
    protected double[] weights;
    
    /**
     * Bias parameters
     */
    protected double bias;
    
    /**
     * Class for generating random numbers
     */
    protected Random rand;

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isRandomizable() {
        //Not randomizable
        return false;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.weights == null) {
            return (inst.classAttribute().isNominal()) ? new double[2] : new double[1];
        }

        double[] result = (inst.classAttribute().isNominal()) ? new double[2] : new double[1];
        double f_t = 0;
        int[] indices = new int[this.numSelectOption.getValue()]; //Sets indices to all 0's

        if (this.evalOption.getChosenIndex() == 0) {
            f_t = dot(inst.toDoubleArray(), this.weights);
            f_t += this.bias;
        } else {
            for (int i = 0; i < this.numSelectOption.getValue(); i++) {
                indices[i] = this.rand.nextInt(inst.numAttributes());
            }
        }

        if (inst.classAttribute().isNumeric()) {
            result[0] = f_t;
            return result;
        }

        if (f_t <= 0) {
            result[0] = 1;
        } else {
            result[1] = 1;
        }

        return result;
    }

    /**
     * Resets the classifier. It is similar from starting the classifier from 
     * scratch.
     */
    @Override
    public void resetLearningImpl() {
        this.weights = null;     // we'll reset when we start learning
        this.bias = 0.0;
        this.rand = new Random();
    }

    /**
     * Trains this classifier incrementally using the given instance.
     * 
     * @param inst The instance to be used for training
     */
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        double y_t, f_t, denom, m_bias;
        int[] indices = new int[this.numSelectOption.getValue()];
        double[] m_weights;

        if (this.weights == null) {

            this.weights = new double[inst.numValues()];
            for (int i = 0; i < this.weights.length; i++) {
                this.weights[i] = this.rand.nextGaussian();
            }
            this.bias = 0.0;
            truncate(this.weights, this.numSelectOption.getValue());
        }

        if (inst.classAttribute().isNominal()) {
            y_t = (inst.classValue() == 0) ? -1 : 1;
        } else {
            y_t = inst.classValue();
        }
        double[] x_t = inst.toDoubleArray();
        double[] x_hat = inst.toDoubleArray();

        if (this.rand.nextDouble() < this.searchOption.getValue()) {
            int[] indices_perm = perm(inst.numAttributes());
            for (int i = 0; i < this.numSelectOption.getValue(); i++) {
                indices[i] = indices_perm[i];
            }

        } else {
            int[] sorted_indices = bubblesort_index(abs_vector(this.weights.clone()));

            for (int i = 0; i < inst.numAttributes() - this.numSelectOption.getValue(); i++) {
                x_hat[sorted_indices[i]] = 0.0;
            }

            for (int i = 0; i < this.numSelectOption.getValue(); i++) {
                indices[i] = sorted_indices[sorted_indices.length - i - 1];
            }
        }

        f_t = 0;
        for (int i = 0; i < this.numSelectOption.getValue(); i++) {
            f_t += this.weights[indices[i]] * x_t[indices[i]];
        }
        f_t += this.bias;

        if (f_t * y_t < 0) {

            for (int i = 0; i < x_hat.length; i++) {
                denom = this.numSelectOption.getValue() / x_hat.length * this.searchOption.getValue();
                if (this.weights[i] != 0) {
                    denom += (1 - this.searchOption.getValue());
                }
                if(denom != 0){
                    x_hat[i] /= denom;
                }
            }

            m_weights = scalar_vector(y_t * this.stepSizeOption.getValue(), x_hat);
            m_bias = y_t * this.stepSizeOption.getValue() * this.bias;
            m_weights = vector_add(m_weights, this.weights);
            m_bias += m_bias + this.bias;

            m_weights = l2_projection(m_weights, m_bias, this.boundOption.getValue());
            truncate(m_weights, this.numSelectOption.getValue());

            for (int i = 0; i < m_weights.length - 1; i++) {
                this.weights[i] = m_weights[i];
            }
            this.bias = m_weights[m_weights.length - 1];
        }

    }

    /**
     * Empty method - not supported.
     * 
     * @return null
     */
    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    /**
     * Empty method - not supported.
     */
    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    /**
     * Compute the dot product of two vectors.
     *
     * @param x input vector
     * @param y input vector
     * @return dot product between x & y.
     */
    public double dot(double[] x, double[] y) {
        double dp = 0.0;
        for (int i = 0; i < x.length; i++) {
            dp += x[i] * y[i];
        }
        return dp;
    }

    /**
     * Compute a*x where a is a scalar and x is a vector
     *
     * @param a scalar (double)
     * @param x vector (double[])
     * @return return the scalar product
     */
    public double[] scalar_vector(double a, double[] x) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = a * x[i];
        }
        return y;
    }

    /**
     * Add two double vectors. This method assumes that the vectors are of the
     * same size.
     *
     * @param x Vector x
     * @param y Vector y
     * @return The sum of two vectors
     */
    public double[] vector_add(double[] x, double[] y) {
        for (int i = 0; i < x.length; i++) {
            x[i] += y[i];
        }
        return x;
    }

    /**
     * 
     * 
     * @param x
     * @param bias
     * @param R
     * @return
     */
    public double[] l2_projection(double[] x, double bias, double R) {
        double norm = 0.0;
        double[] y = new double[x.length + 1];

        for (int i = 0; i < x.length; i++) {
            norm += x[i] * x[i];
        }
        norm = Math.sqrt(norm + bias * bias);
        double a = Math.sqrt(R) / norm;

        if (a < 1) {
            for (int i = 0; i < x.length; i++) {
                y[i] = a * x[i];
            }
            y[y.length - 1] = a * bias;
        } else {
            for (int i = 0; i < x.length; i++) {
                y[i] = x[i];
            }
            y[y.length - 1] = bias;
        }
        return y;
    }

    /**
     * Keep only the B entries of larger magnitudes in the given vector. The 
     * vector is manipulated via pass-by-reference, so it does not need to be 
     * returned.
     *
     * @param x Vector containing weights
     * @param B Number of larger magnitude weights to keep.
     */
    public void truncate(double[] x, int B) {
        int[] sorted_indices = bubblesort_index(abs_vector(x.clone()));
        for (int i = 0; i < x.length - B; i++) {
            x[sorted_indices[i]] = 0.0;
        }
    }

    /**
     * Sort the elements of a vector, and returns the indices that were sorted.
     * The sorted indices return the location of the smallest to largest weight 
     * values in the x array.
     * 
     * Note: x should not be sorted when this method is complete. 
     *
     * @param x vector to be sorted
     * @return vector of sorted indices
     */
    public int[] bubblesort_index(double[] x) {
        int[] y = new int[x.length];
        double[] temp = new double[x.length];
        
        //Create a copy of the x array so as to not change its values.
        System.arraycopy(x, 0, temp, 0, x.length);
        
        boolean flag = true;
        double t, r;

        // Initialize the indices array
        for (int i = 0; i < temp.length; i++) {
            y[i] = i;
        }

        while (flag) {
            flag = false;
            for (int j = 0; j < temp.length - 1; j++) {
                if (temp[j] > temp[j + 1]) {
                    t = temp[j];
                    r = y[j];

                    temp[j] = temp[j + 1];
                    y[j] = y[j + 1];

                    temp[j + 1] = t;
                    y[j + 1] = (int) r;
                    flag = true;
                }
            }
        }
        return y;
    }

    /**
     * Compute the absolute value of the elements in a vector
     *
     * @param x
     * @return
     */
    public double[] abs_vector(double[] x) {
        for (int i = 0; i < x.length; i++) {
            if (x[i] < 0) {
                x[i] = -x[i];
            }
        }
        return x;
    }

    /**
     * Create a randomly permutation sequence of a pre-specified length.
     *
     * @param length
     * @return
     */
    public int[] perm(int length) {

        // initialize array and fill it with {0,1,2...}
        int[] array = new int[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = i;
        }

        for (int i = 0; i < length; i++) {

	    // randomly chosen position in array whose element
            // will be swapped with the element in position i
            // note that when i = 0, any position can chosen (0 thru length-1)
            // when i = 1, only positions 1 through length -1
            // NOTE: r is an instance of java.util.Random
            int ran = i + this.rand.nextInt(length - i);

            // perform swap
            int temp = array[i];
            array[i] = array[ran];
            array[ran] = temp;
        }
        return array;
    }
}
