package de.ebamberg.djl.demo.timeseries.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.fasterxml.jackson.databind.ObjectMapper;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.BlockFactory;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import de.ebamberg.djl.demo.timeseries.training.LSTMTimeseriesForcastBlockFactory;
import de.ebamberg.djl.demo.timeseries.utils.MinMaxScaler;

@Configuration
public class DJLConfiguration {

	@Bean
	public BlockFactory blockFactory() {
		return new LSTMTimeseriesForcastBlockFactory();
	}

	@Bean
	public Translator<float[][], float[]> translator(ObjectMapper jsonMapper) {
		return new Translator<float[][], float[]>() {

			@Override
			public NDList processInput(TranslatorContext ctx, float[][] input) throws Exception {
					MinMaxScaler scaler = new MinMaxScaler();
					String minScaleProp = ctx.getModel().getProperty("minScale");
					String maxScaleProp = ctx.getModel().getProperty("maxScale");
					var min = jsonMapper.readValue(minScaleProp, float[].class);
					var max = jsonMapper.readValue(maxScaleProp, float[].class);
					var manager = ctx.getNDManager();
					scaler.fitToRange(manager.create(min), manager.create(max));
					ctx.setAttachment("scaler", scaler);

					var array = ctx.getNDManager().create(input);
					scaler.transformi(array);
					array = array.reshape(new Shape(1, array.getShape().get(0), array.getShape().get(1)));
				
				return new NDList(array);
			}

			@Override
			public float[] processOutput(TranslatorContext ctx, NDList list) throws Exception {
				var resultArray = list.get(0);
				resultArray = ((MinMaxScaler)ctx.getAttachment("scaler")).inverseTransform(resultArray);
				return resultArray.toFloatArray();
			}

			@Override
			public Batchifier getBatchifier() {
				return null;
			}

		};

	}

}
