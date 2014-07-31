/*
 *    PAME.java
 *    Copyright (C) 2013, Drexel University 
 *    @author Gregory Ditzler (gregory.ditzler@gmail.com)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package moa.classifiers.meta;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import moa.options.FlagOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import weka.core.Instance;


public class PAME extends AbstractClassifier {

	/**
	 * declare the properties of the online learning algorithm. they are as follows:
	 *   :ensemble - vector of classifiers
	 *   :ensembleSizeOption - number of experts
	 *   :baseLearnerOption - base learning algorithm
	 *   :updateMethodOption - what kind of weight updates are we using (see paper)
	 *   :weights - vector of weights for the experts
	 */
	protected Classifier[] ensemble;
	public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of expert in the ensemble.", 10, 1, Integer.MAX_VALUE);
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");
	public MultiChoiceOption updateMethodOption = new MultiChoiceOption(
            "updateMethod", 'u', "The update method used.", new String[]{
                "PAME-I", "PAME-II", "PAME-III"}, new String[]{
                "Update +/- Weight",
                "Update + Weight clipping",
                "Formal Optimization"}, 0);
	public double[] weights;
	public double n_negativeWeights;
	public double rareCount;
	public double count;
	private double C = 0.01;
	private static final long serialVersionUID = 1L;
	public FlagOption overSampleOption = new FlagOption("overSample",
            'o', "Oversample class 0.");
	public FlagOption underSampleOption = new FlagOption("underSample",
            'm', "Undersample class 0.");
	public FlagOption logTransformOption = new FlagOption("logTransform",
            'z', "Log(1/p)");
	
	
	@Override
    public String getPurposeString() {
        return "Online Convex Weight Optimization Algorithm.";
    }
	
	/*
	 * set up some constants to make the coding thing a little bit less ambiguous
	 */
	private int PAME1 = 0;
	private int PAME2 = 1;
	private int PAME3 = 2;
	

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return true;
	}

	/*
	 * Compute the composite hypothesis of the classifier ensemble
	 */
	@Override
	public double[] getVotesForInstance(Instance inst) {
	    double[] wh = new double[this.ensemble.length];
		double H = 0.0;
		double[] Hout = {0.0, 0.0};

		for (int i = 0; i < this.ensemble.length; i++) {
			// get the weight of the current expert and get the vote as well. recall
			// that MOA will return a probabilistic interpretation of the the prediction.
			// we need to convert this to positive / negative to go with our formulation
			// of the learning problem. 
            double memberWeight = this.weights[i];            
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
            if (vote.sumOfValues() > 0.0) {
            	vote.normalize();
            }
            // index 0 = +1
            // index 1 = -1
            if (vote.getValue(0) >= vote.getValue(1)){
            	wh[i] = 1*memberWeight;
            }else{
            	wh[i] = -1*memberWeight;
            }
            
            // update the ensemble prediction
            H += wh[i];
        }

		// the sign of the result determines the final decision

		if (H >= 0){
			Hout[0] = 1.0;
			Hout[1] = 0.0;
		}else{
			Hout[0] = 0.0;
			Hout[1] = 1.0;
		}
		// convert the predictions to posterior probabilities with the soft-max function. 
		//Hout[0] = 1.0 / (1.0 + Math.exp(-1.0*H));
		//Hout[1] = 1 - Hout[0];
		return Hout;
	}

	@Override
	public void resetLearningImpl() {
		this.n_negativeWeights = 0;
		this.rareCount = 0.0;
		this.count = 0.0;
		
		// initialize the experts in the ensemble with null base learners
		this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
        }
        this.weights = new double[this.ensemble.length];
        // if the we are using PAOLN2 or OLNCVX in the update we will initialize the weights to 
        // be uniform and convex; however, if we are using PAOLN1 we will just use the zero
        // vector as a starting point. 
        if (this.updateMethodOption.getChosenIndex() == PAME2 || 
        		this.updateMethodOption.getChosenIndex() == PAME3){
        	for (int i = 0; i < this.ensemble.length; i++){
        		this.weights[i] = 1.0/this.ensemble.length;
        	}
        }else{ // this implements PAOLN1
        	for (int i = 0; i < this.ensemble.length; i++){
        		this.weights[i] = 0.001;
        	}
        }
	}
	
	private void pame1_weights(double[] ht, double yt){
		double alpha = (1.0 - yt*this.dot(ht, this.weights)) / this.dot(ht, ht);
		if (alpha < 0)
			alpha = 0;
		if (this.C < alpha)
			alpha = this.C;  // apply regularization

		this.weights = this.addvectors(this.weights, this.scalarvector(alpha*yt, ht));
	}

	private void pame2_weights(double[] ht, double yt){
		pame1_weights(ht, yt);
		//double zz = 0.0;
		for (int i = 0; i < this.weights.length; i++){
			if (this.weights[i] < 0.0)
				this.weights[i] = 0.0;   // clip the weights
			//zz += this.weights[i];
		}
		//System.out.printf("   sums: %f \n", zz);
	}

	private void pame3_weights(double[] ht, double yt){
		double K = (double) this.weights.length;

		/* create a vector of ones */
		double[] onesVec = new double[this.weights.length];
		for (int k = 1; k < onesVec.length; k++){
			onesVec[k] = 1.0;
		}

		/* see proof for this value */
		double hh = this.dot(ht, onesVec);
		double[] normVec = this.addvectors(ht, this.scalarvector(-1.0*hh/K, onesVec));
		double denom = this.dot(normVec, normVec);
		double alpha = (1.0 - yt*this.dot(ht, this.weights)) / denom;
		if (alpha < 0)
			alpha = 0;
		if (this.C < alpha)
			alpha = this.C;  // apply regularization

		double[] update = this.scalarvector(alpha*yt, this.addvectors(ht, 
				this.scalarvector(-1.0*hh/K, onesVec)));
		this.weights = this.addvectors(this.weights, update);

		/*
		 * project the current weight vector onto a probability simplex
		 */
		boolean bGet = Boolean.FALSE;
		double tmpsum = 0.0;
		double tmax = 0.0;
		double[] s = this.bubblesort(this.weights);	

		for (int i = 0; i < s.length - 1; i++){
			tmpsum += s[i];
			tmax = (tmpsum - 1.0)/ (i+1);
			if(tmax >= s[i + 1]){
				bGet = Boolean.TRUE;
				break;
			}
		}

		if (!bGet){
			tmax = (tmpsum + s[s.length-1] - 1) / K;
		}

		for (int k = 0; k < this.weights.length; k++){
			if (this.weights[k] - tmax < 0){
				this.weights[k] = 0;
			} else {
				this.weights[k] -= tmax;
			}
		}
	}

	/*
	 * This method is for training the experts in the ensemble and updating the 
	 * weights of each of the experts using an online convex optimization 
	 * algorithm. 
	 */
	@Override
	public void trainOnInstanceImpl(Instance inst) {

		// get the prediction vector back
		double[] ht = this.getPredictions(inst);
		double yt = inst.classValue();
		if (inst.classIndex() == 0){
			this.rareCount += 1.0;
		}
		this.count += 1.0;
		
		// convert to a positive / negative classification scenario
		if (yt == 0){
			yt = 1.0;
		}else{
			yt = -1.0;
		}

		/*
		 * update expert weights
		 */
		if (this.updateMethodOption.getChosenIndex() == PAME1){
			pame1_weights(ht, yt);
		} else if (this.updateMethodOption.getChosenIndex() == PAME2){
			pame2_weights(ht, yt);
		} else if (this.updateMethodOption.getChosenIndex() == PAME3){
			pame3_weights(ht, yt);
		}

		/*
		 * we are going to use an online bagging / boosting strategy to update the 
		 * experts. In the end our objective with the weight formulation is a bit
		 * more of a decision theoretic approach. 
		 */
		for (int i = 0; i < this.ensemble.length; i++) {
			// sample from a Poisson probability distribution as implemented in 
			// online bagging and boosting]
			double w;
			if (this.overSampleOption.isSet() && inst.classIndex() == 0){
				w = 1.0 / (this.rareCount/this.count);
				if (this.logTransformOption.isSet()){
					w = Math.log(w);
				}
			} else if (this.underSampleOption.isSet() && inst.classIndex() != 0){
				w = 1.0 - this.rareCount/this.count;
			} else {
				w = 1.0;
			}
			
            int k = MiscUtils.poisson(w, this.classifierRandom);
            
            // update the expert accordingly 
            if (k > 0) {
            	// this works by updating the expert k-times with the same example.
            	// thus is k = 4. the expert is trained updated on the same example
            	// 4 times in a row. pretty easy.
                Instance weightedInst = (Instance) inst.copy();
                weightedInst.setWeight(inst.weight() * k);       // set the # of training times
                this.ensemble[i].trainOnInstance(weightedInst);  // update expert
            }
        }

		this.n_negativeWeights = 0;
		for (int i = 0; i < this.weights.length; i++){
			if (this.weights[i] < 0.0)
				this.n_negativeWeights++;
		}
	}

	/*
	 * This method returns a vector of predictions by each of the experts. 
	 */
	public double[] getPredictions(Instance inst){
		double[] h = new double[this.ensemble.length]; // initialize vector of predictions

		for (int i = 0; i < this.ensemble.length; i++){
			// use only the posterior probability on the ``positive'' positive class 
			// in the binary prediction scenario. 
			DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
			if (vote.sumOfValues() > 0.0) {
                vote.normalize();
            }

			// index 0 = +1
			// index 1 = -1
			if (vote.getValue(0) >= vote.getValue(1)){
				h[i] = 1;
			}else{
				h[i] = -1;
			}
		}
		return h;
	}


	/*
	 * implement the dot product
	 */
	public double dot(double[] x, double[] y){
		double dp = 0.0;

		for (int i = 0; i < x.length; i++){
			dp += x[i]*y[i];
		}
		return dp;
	}


	/*
	 * multiply a vector by a scalar
	 */
	public double[] scalarvector(double a, double[] x){
		for (int i = 0; i < x.length; i++){
			x[i] = a*x[i];
		}
		return x;
	}


	/*
	 * add two vectors together
	 */
	public double[] addvectors(double[] x, double[] y){
		double[] z = new double[x.length];

		for (int i = 1; i < z.length; i++){
			z[i] = x[i] + y[i];
		}

		return z;
	}


	private double[] bubblesort(double[] x){
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


	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return new Measurement[]{new Measurement("ensemble size",
                this.ensemble != null ? this.ensemble.length : 0),
                new Measurement("negative weights",this.n_negativeWeights)};
	}


	/*
	 * Some of the other classes I looked at did not have anything for this method
	 * so I just left it alone. It really doesn't seem that important right now. 
	 */
	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
	}

}
