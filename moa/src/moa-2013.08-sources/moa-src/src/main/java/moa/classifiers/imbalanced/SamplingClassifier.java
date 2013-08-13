/*
 *    SamplingClassifier.java
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
package moa.classifiers.imbalanced;

import weka.core.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import moa.options.FlagOption;

public class SamplingClassifier extends AbstractClassifier {
	private static final long serialVersionUID = 1L;
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");
	public FlagOption overSampleOption = new FlagOption("overSample",
            'o', "Oversample class 0.");
	public FlagOption underSampleOption = new FlagOption("underSample",
            'm', "Undersample class 0.");
	public FlagOption logTransformOption = new FlagOption("logTransform",
            'z', "Log(1/p)");
	public double rareCount;
	public double count;
	protected Classifier classifier;
	
	
	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		return this.classifier.getVotesForInstance(inst);
	}

	@Override
	public void resetLearningImpl() {
		Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
		baseLearner.resetLearning();
		this.classifier = baseLearner.copy();
		this.rareCount = 0.0;
		this.count = 0.0;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (inst.classIndex() == 0){
			this.rareCount += 1.0;
		}
		this.count += 1.0;
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
		Instance weightedInst = (Instance) inst.copy();
		weightedInst.setWeight(inst.weight() * k);
		this.classifier.trainOnInstance(weightedInst);
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		Measurement[] m = new Measurement[0];
        return m;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
		
	}

}
