package de.ebamberg.djl.demo.timeseries.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.fasterxml.jackson.databind.ObjectMapper;

import ai.djl.nn.BlockFactory;
import ai.djl.translate.Translator;
import de.ebamberg.djl.demo.timeseries.training.LSTMTimeseriesForcastBlockFactory;
import de.ebamberg.djl.lib.timeseries.Forecast;
import de.ebamberg.djl.lib.timeseries.ForecastTranslator;

@Configuration
public class DJLConfiguration {

	@Bean
	public BlockFactory blockFactory() {
		return new LSTMTimeseriesForcastBlockFactory();
	}

	@Bean
	public Translator<float[][], Forecast> translator() {
		return new ForecastTranslator();
	}

}
