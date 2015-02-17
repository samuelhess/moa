/*
 *    OFSL.java
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

public class OFSL extends AbstractClassifier {
	
	/** serial ID */
	private static final long serialVersionUID = 1L;
	/** regularization parameter */
	public FloatOption stepSizeOption = new FloatOption("stepSize",
            's', "The step size.",
            0.2, 0.00, Integer.MAX_VALUE);
	/** step size */
	public FloatOption learningRateOption = new FloatOption("learningRate",
            'r', "Learning rate parameter.",
            0.2, 0.00, Integer.MAX_VALUE);
	/** truncation size or the number of features to select */
	public IntOption numSelectOption = new IntOption("numSelect",
            'n', "The number of features to select.",
            10, 0, Integer.MAX_VALUE);
	/** parameter vector for prediction */
	protected double[] weights;
	/** bias parameters */
	protected double bias;
	/** number of features */
	protected int n_features;
	
	
	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	
	@Override
	public double[] getVotesForInstance(Instance inst) {
		
		if (this.weights == null)
			return (inst.classAttribute().isNominal()) ? new double[2] : new double[1];
		
		double[] result = (inst.classAttribute().isNominal()) ? new double[2] : new double[1];
		double f_t = dot(inst.toDoubleArray(), this.weights);
		f_t += this.bias;
		
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

	
	@Override
	public void resetLearningImpl() {
		this.weights = null;     // we'll reset when we start learning
		this.bias = 0.0;
	}

	
	@Override
	public void trainOnInstanceImpl(Instance inst) {
		double y_t, m_bias_p1, m_bias_p2, m_bias;
		double[] m_weights_p1, m_weights_p2, m_weights;
		
		if (this.weights == null) {
			this.weights = new double[inst.numValues()];
			for (int i = 0; i < this.weights.length; i++)
				this.weights[i] = 0.0;
			this.bias = 0.0;
		}
		if (inst.classAttribute().isNominal()) {
			y_t = (inst.classValue() == 0) ? -1 : 1;
        } else {
        	y_t = inst.classValue();
        }
		
		double f_t = dot(inst.toDoubleArray(), this.weights);
		f_t += this.bias;
		
		if (y_t*f_t < 0){
			m_weights_p1 = scalar_vector(1.0 - this.stepSizeOption.getValue()*this.learningRateOption.getValue(), this.weights);
			m_bias_p1 = (1.0 - this.stepSizeOption.getValue()*this.learningRateOption.getValue())*this.bias;
			m_weights_p2 = scalar_vector(this.learningRateOption.getValue()*y_t, inst.toDoubleArray());
			m_bias_p2 = this.learningRateOption.getValue()*y_t;
			m_weights = vector_add(m_weights_p1, m_weights_p2);
			m_bias = m_bias_p1 + m_bias_p2;
			
			m_weights = l2_projection(m_weights, m_bias, this.learningRateOption.getValue());
			m_weights = truncate(m_weights, this.numSelectOption.getValue());
			
			for (int i = 0; i < m_weights_p1.length; i++)
				this.weights[i] = m_weights[i];
			this.bias = m_weights[m_weights.length-1];
		} else {
			this.weights = scalar_vector(1.0 - this.stepSizeOption.getValue()*this.learningRateOption.getValue(), this.weights);
			this.bias = (1.0 - this.stepSizeOption.getValue()*this.learningRateOption.getValue())*this.bias;
		}
		
		
	}

	
	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	
	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
		
	}
		
	
	/** 
	 * Compute the dot product of two vectors
	 * @param x input vector
	 * @param y input vector
	 * @return dot product between x & y.
	 */
	public double dot(double[] x, double[] y){
		double dp = 0.0;
		for (int i = 0; i < x.length; i++)
			dp += x[i]*y[i];
		return dp;
	}
	
	/**
	 * Compute a*x where a is a scalar and x is a vector
	 * @param a scalar (double)
	 * @param x vector (double[])
	 * @return return the scalar product
	 */
	public double[] scalar_vector(double a, double[] x) {
		double[] y = new double[x.length];
		for (int i = 0; i < x.length; i++)
			y[i] = a*x[i];
		return y;
	}
	
	/**
	 * Add two double vectors 
	 * @param x
	 * @param y
	 * @return
	 */
	public double[] vector_add(double[] x, double[] y) {
		for (int i = 0; i < x.length; i++) {
			x[i] += y[i];
		}
		return x;
	}
	
	/**
	 * 
	 * @param x
	 * @params bias
	 * @param lambda
	 * @return
	 */
	public double[] l2_projection(double[] x, double bias, double lambda) {
		double norm = 0.0;
		double[] y = new double[x.length+1];
		
		for (int i = 0; i < x.length; i++)
			norm += x[i]*x[i];
		norm = Math.sqrt(norm + bias*bias);
		double a = (1/Math.sqrt(lambda))/norm;
		
		if (a < 1) {
			for (int i = 0; i < x.length; i++) 
				y[i] = a*x[i];
			y[y.length-1] = a*bias;
		} else {
			for (int i = 0; i < x.length; i++) 
				y[i] = x[i];
			y[y.length-1] = bias;
		}
		return y;
	}
	
	
	/**
	 * Keep only the B largest entries in a vector x 
	 * @param x
	 * @param B
	 * @return
	 */
	public double[] truncate(double[] x, int B) {
		int[] sorted_indices = bubblesort_index(abs_vector(x));
		for (int i = 0; i < x.length - B - 1; i++)
			x[sorted_indices[i]] = 0.0;
		return x;
	}
	
	
	
	/** Sort the elements of a vector 
	 * @param x vector to be sorted 
	 * @return sorted vector
	 */
	private int[] bubblesort_index(double[] x){
		int[] y = new int[x.length];
		boolean flag = true;
		double t, r;
		
		/*initialize the indices*/
		for (int i = 0; i < x.length; i++)
			y[i] = i;
		
		while(flag){
			flag = false;
			for (int j = 0; j < x.length - 1; j++){
				if (x[j] < x[j + 1]){
					t = x[j];
					r = y[j];
					
					x[j] = x[j + 1];
					y[j] = y[j + 1];
					
					x[j + 1] = t;
					y[j + 1] = (int) r;
					flag = true;
				}
			}
		}	
		return y;
	}
	
	/**
	 * 
	 * @param x
	 * @return
	 */
	public double[] abs_vector(double[] x) {
		for (int i = 0; i < x.length; i++) {
			if (x[i] < 0)
				x[i] = -x[i];
		}
		return x;
	}
	
	
	

}
