package de.ebamberg.djl.lib.timeseries;

import static de.ebamberg.djl.demo.timeseries.utils.NDArraySerializer.stringToNDArray;

import java.util.Collections;
import java.util.Map;
import java.util.WeakHashMap;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import de.ebamberg.djl.demo.timeseries.utils.MinMaxScaler;
import static de.ebamberg.djl.lib.core.StandardModelProperties.PROPERTY_MIN_SCALE;
import static de.ebamberg.djl.lib.core.StandardModelProperties.PROPERTY_MAX_SCALE;
public class ForecastTranslator implements Translator<float[][], Forecast> {

	private static Map<String, MinMaxScaler> scalerCache=Collections.synchronizedMap(new WeakHashMap<>());
	
	@Override
	public NDList processInput(TranslatorContext ctx, float[][] input) throws Exception {
			var manager = ctx.getNDManager();
			
			//TODO refresh Scaler when the version of model changed
			var scaler=scalerCache.computeIfAbsent(ctx.getModel().getName(), (k)-> recreateScalerFromModel(ctx, manager).detach() );
			ctx.setAttachment("scaler", scaler);
			
			var array = manager.create(input);
			scaler.transformi(array);
			array = array.reshape(new Shape(1, array.getShape().get(0), array.getShape().get(1)));
		
		return new NDList(array);
	}



	@Override
	public Forecast processOutput(TranslatorContext ctx, NDList list) throws Exception {
		var resultArray = list.get(0);
		resultArray = ((MinMaxScaler)ctx.getAttachment("scaler")).inverseTransform(resultArray);
		return new Forecast(resultArray.toFloatArray());
	}

	@Override
	public Batchifier getBatchifier() {
		return null;
	}
	
	private MinMaxScaler recreateScalerFromModel(TranslatorContext ctx, NDManager manager) {
		MinMaxScaler scaler = new MinMaxScaler();
		var min = stringToNDArray(manager,ctx.getModel().getProperty(PROPERTY_MIN_SCALE));
		var max = stringToNDArray(manager,ctx.getModel().getProperty(PROPERTY_MAX_SCALE));				
		scaler.fitToRange(min, max);
		ctx.setAttachment("scaler", scaler);
		return scaler;
	}

}
