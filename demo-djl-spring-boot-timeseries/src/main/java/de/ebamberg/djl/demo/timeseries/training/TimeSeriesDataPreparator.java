package de.ebamberg.djl.demo.timeseries.training;

import com.fasterxml.jackson.databind.ObjectMapper;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import de.ebamberg.djl.demo.timeseries.utils.MinMaxScaler;
import static de.ebamberg.djl.lib.core.StandardModelProperties.PROPERTY_LOOKBACK;
import static de.ebamberg.djl.lib.core.StandardModelProperties.PROPERTY_FEATURES;

public class TimeSeriesDataPreparator {
	

	private NDManager manager;
	private Model model;
	private MinMaxScaler scaler;
	
	public TimeSeriesDataPreparator(NDManager manager, Model model, MinMaxScaler scaler) {
		super();
		this.manager = manager;
		this.model = model;
		this.scaler = scaler;
	}


	
	public NDList  prepare(float [][] dataInput,int lookback) {
		    NDArray data=manager.create(dataInput);
		    scaler.fit(data).transformi(data);			// fit and transform inplace
		    
		    var timesteps = data.getShape().get(0);
		    var number_of_features = data.getShape().get(1);
		    
		    model.setProperty(PROPERTY_FEATURES, String.valueOf(number_of_features));
		    model.setProperty(PROPERTY_LOOKBACK, String.valueOf(lookback));
		    
		    var listOfXYData=temporize(data,lookback);
		    
		    return listOfXYData;
	}
	
	
	private NDList temporize(NDArray data, int lookback) {
		
		var  timesteps = data.getShape().get(0);
	    long number_of_samples = timesteps-lookback;

	    NDList temporizedData=new NDList((int) number_of_samples);
	    NDList labels=new NDList((int) number_of_samples);
	    
	    long predictOffset=1;
	    for (long i=timesteps-number_of_samples;i <=timesteps-predictOffset; i++) {
	    		System.out.println(String.format("picking sample #%d\tend #%d",i,(timesteps-predictOffset) ) );
	    	    NDArray sample=data.get(new NDIndex("{}:{}, : ", (i-lookback),i));
	    	    System.out.println(scaler.inverseTransform(sample));
	    	    System.out.println(sample);
	    	    NDArray label=data.get(new NDIndex("{}, 0 ",i));
	    	    System.out.println(label);
	    	    temporizedData.add(sample);
	    	    labels.add(label);
	    }
	    NDArray  temporizedDataAsArray =NDArrays.stack(temporizedData);
	    NDArray labelsAsArray = NDArrays.stack(labels);
	    temporizedDataAsArray.setName("X");
	    labelsAsArray.setName("y");
	    return new NDList(temporizedDataAsArray,labelsAsArray);
	}
	
	
}
