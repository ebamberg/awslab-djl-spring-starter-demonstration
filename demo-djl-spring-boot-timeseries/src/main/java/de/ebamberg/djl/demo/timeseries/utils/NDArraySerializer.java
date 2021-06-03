package de.ebamberg.djl.demo.timeseries.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public class NDArraySerializer {

	private static Logger log=LoggerFactory.getLogger(NDArraySerializer.class);
	
	private static final ObjectMapper mapper = new ObjectMapper();

	
	public static String ndarrayAsString(NDArray array) {
		try {
			return mapper.writeValueAsString(array.toFloatArray());
		} catch (JsonProcessingException e) {
			log.error("error while serialize ndarray to string",e);
		}
		return null;
		
	}
	
	public static NDArray stringToNDArray(NDManager manager,String input) {
			return manager.create(stringToFloatArray(input));
	}

	public static float[] stringToFloatArray(String input) {
		try {
			return mapper.readValue(input,float[].class);
		} catch (JsonProcessingException e) {
				log.error("error while serialize ndarray to string",e);
		}
		return new float[] {};
	}
	
	public static float[][] stringToNDimFloatArray(String input) {
		try {
			return mapper.readValue(input,float[][].class);
		} catch (JsonProcessingException e) {
				log.error("error while serialize ndarray to string",e);
		}
		return new float[][] {};
	}
	
}
