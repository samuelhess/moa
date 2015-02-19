package moa.classifiers.featureselection;

import weka.core.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.Measurement;

public class OFSBoosting extends AbstractClassifier {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
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
		// TODO Auto-generated method stub
		
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
