package de.ebamberg.djl.lib.timeseries;

import java.util.Collections;
import java.util.function.BiFunction;

import ai.djl.inference.Predictor;
import ai.djl.translate.TranslateException;

public class FeedbackPredictor<I, O> {

	private Predictor<I, O> wrappedPredictor;
	
	private BiFunction<I,O,I> transferFunction;

	public FeedbackPredictor(Predictor<I, O> wrappedPredictor, BiFunction<I, O, I> transferFunction) {
		super();
		this.wrappedPredictor = wrappedPredictor;
		this.transferFunction = transferFunction;
	} 
	
    public O predict(I input) throws TranslateException {
        return wrappedPredictor.batchPredict(Collections.singletonList(input)).get(0);
    }
	
}
