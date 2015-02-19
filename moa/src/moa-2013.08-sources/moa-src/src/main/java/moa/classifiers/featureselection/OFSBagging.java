/*
 *    OFSBagging.java
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
import moa.classifiers.Classifier;
import moa.core.Measurement;
import moa.classifiers.featureselection.OFSL;
import moa.classifiers.featureselection.OFSP;
import moa.options.ClassOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import moa.classifiers.Classifier;

public class OFSBagging extends AbstractClassifier {

	/**	serial id */
	private static final long serialVersionUID = 1L;
	public MultiChoiceOption evalOption = new MultiChoiceOption("evaluation", 
			'p', "Evaluate on a full or partial information instance.", 
			new String[] {"full","partial"}, new String[] {"full","partial"}, 0);
	public MultiChoiceOption learningSettingOption = new MultiChoiceOption("learningSetting", 
			'l', "Use full or partial information for learning.", 
			new String[] {"full","partial"}, new String[] {"full","partial"}, 0);
	public IntOption ensembleSizeOption = new IntOption("ensembleSize",
            'n', "Size of the ensemble.",
            10, 0, Integer.MAX_VALUE);
	public IntOption numSelectOption = new IntOption("numSelect",
            'n', "The number of features to select.",
            10, 0, Integer.MAX_VALUE);
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train (Must be OFSL or OFSP).", Classifier.class, 
            "featureselection.OFSL");
	protected Classifier[] ensemble;
	
	
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
		//
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// TODO Auto-generated method stub
		
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

}
