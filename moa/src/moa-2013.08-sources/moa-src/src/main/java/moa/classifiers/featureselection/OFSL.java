/*
 *    OFSL.java
 *    Copyright (C) 2014 Drexel University, Philadelphia, PA
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
		// TODO Auto-generated method stub
		return null;
	}

	
	@Override
	public void resetLearningImpl() {
		this.weights = null;     // we'll reset when we start learning
		this.bias = 0.0;
	}

	
	@Override
	public void trainOnInstanceImpl(Instance inst) {
		
		if (this.weights == null) {
			this.weights = new double[inst.numValues()];
			for (int i = 0; i < this.weights.length; i++)
				this.weights[i] = 0.0;
			this.bias = 0.0;
		}
		
		double y_t = (inst.classValue() == 0) ? -1 : 1;
		double f_t = dot(inst.toDoubleArray(), this.weights);
		f_t += this.bias;
		
		if (y_t*f_t < 0){
			
		} else {
			
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
	
	
	/** Sort the elements of a vector 
	 * @param x vector to be sorted 
	 * @return sorted vector
	 */
	private double[] bubblesort(double[] x, double[] y){
		boolean flag = true;
		double t;
		while(flag){
			flag = false;
			for (int j = 0; j < x.length - 1; j++){
				if (x[j] < x[j + 1]){
					t = x[j];
					x[j] = x[j + 1];
					x[j + 1] = t;
					flag = true;
				}
			}
		}	
		return x;
	}
	
	/** Sort the elements of a vector 
	 * @param x vector to be sorted 
	 * @return sorted vector
	 */
	private double[] bubblesort_index(double[] x){
		double[] y = new double[x.length];
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
					y[j + 1] = r;
					flag = true;
				}
			}
		}	
		return y;
	}
	
	
	/** Compute the sign of the elements in a vector
	 * @param x vector of doubles
	 * @return sign of the vectors elements
	 */
	private double[] sign(double[] x){
		for (int j = 0; j < x.length; j++) {
			if (x[j] >= 0) 
				x[j] = 1.0;
			else
				x[j] = -1.0;
		}
		return x;
	}
	
	
	/** Compute the absolute value of a vector
	 * @param x input vector
	 * @return absolute value of the elements in x
	 */
	private double[] abs(double[] x){
		for (int j = 0; j < x.length; j++) {
			if (x[j] < 0) 
				x[j] = -x[j];
		}
		return x;
	}
	
	
	

}
