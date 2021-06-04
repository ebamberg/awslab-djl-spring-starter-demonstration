package de.ebamberg.djl.demo.timeseries.http;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.ConcurrentHashMap;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.BlockFactory;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import de.ebamberg.djl.demo.timeseries.datasource.CSVDataSource;
import de.ebamberg.djl.demo.timeseries.repository.ModelRepository;
import de.ebamberg.djl.demo.timeseries.training.TimeSeriesDataPreparator;
import de.ebamberg.djl.demo.timeseries.training.TrainerService;
import de.ebamberg.djl.demo.timeseries.utils.MinMaxScaler;
import de.ebamberg.djl.lib.core.ModelStore;
import de.ebamberg.djl.lib.timeseries.Forecast;

import static de.ebamberg.djl.demo.timeseries.utils.NDArraySerializer.ndarrayAsString; 
import static de.ebamberg.djl.demo.timeseries.utils.NDArraySerializer.stringToFloatArray;

@RestController
public class StandardController {

	private static Logger log=LoggerFactory.getLogger(StandardController.class);
	
	@Autowired
	private CSVDataSource dataSource;

	@Autowired
	private TrainerService trainerService;
	
	@Autowired
	private ModelStore modelStore;
	
	@Autowired
	private BlockFactory blockFactory;
	
	@Autowired
	private Translator<float[][],Forecast> translator;
	
	@Value ("${timeseries.lookback:4}")
	private int lookback;
	
	@Value ("${timeseries.epochs:200}")
	private int epochs;


	
	
	/**
	 * curl -X POST http://127.0.0.1:8080/timeseries/airlinepassangers -F "data=@./src/main/resources/airline-passengers.csv"
 
	 * @param file
	 * @return
	 * @throws Exception
	 */
	@PostMapping (value="timeseries/{modelName}")
	public String upload(@RequestParam("data") MultipartFile file, @PathVariable String modelName) throws Exception {
		float[][] rawData=dataSource.read(file.getInputStream());
		String result="";
		try(var manager = NDManager.newBaseManager(); var model=modelStore.createEmptyModel( modelName ) ) {
			// preparing the dataset
			var dataset=buildFramedTimeSeriesDataSet(manager, model,  rawData, lookback);			
			//train the dataset
			trainerService.train(model, dataset,epochs);
			// store the model 
			modelStore.store(model);

		}
		return result;
	}


	
	/**
	 * curl -X GET http://127.0.0.1:8080/timeseries/airlinepassangers?lookforward=5
	 * @return
	 * @throws TranslateException
	 * @throws IOException 
	 * @throws MalformedModelException 
	 */
	@GetMapping (value="timeseries/{modelName}")
	public List<Forecast> predict(@RequestParam (defaultValue = "5" ,required = false) int lookforward,@PathVariable String modelName) throws TranslateException, MalformedModelException, IOException {

		var modelForPrediction=modelStore.findModel(modelName);
		
		var lastSample= stringToFloatArray( modelForPrediction.getProperty("lastSample") );
		
		var data=new float[lastSample.length][] ;
		for (int i=0; i<lastSample.length; i++) {
			data[i]=new float[] {lastSample[i]};
		}
		
		var predictor=modelForPrediction.newPredictor(translator);
		
		var result=new ArrayList<Forecast>(lookforward);
		for (int i=0;i<lookforward;i++) {
			Forecast forecast=predictor.predict(data);
			result.add(forecast);
			
			for (int j=1;j<data.length;j++) {
				data[j-1]=data[j];
			}
			data[data.length-1]=forecast.getForecastValue();
			
		}
		return result;
	}
	

	private Dataset buildFramedTimeSeriesDataSet(NDManager manager , Model model, float[][] rawData,int lookback) throws IOException, TranslateException {
		MinMaxScaler scaler=new MinMaxScaler();
		TimeSeriesDataPreparator timeSeriesDataPreparator=new TimeSeriesDataPreparator(manager, model,scaler);
		var data= timeSeriesDataPreparator.prepare(rawData, lookback);
		model.setProperty("minScale", ndarrayAsString(scaler.getMin()));
		model.setProperty("maxScale", ndarrayAsString(scaler.getMax()));
		model.setProperty("lastSample", ndarrayAsString( scaler.inverseTransform(data.get(0).get(-1))) );
		
		var	 dataset = new ArrayDataset.Builder()
		       .setData(data.get(0))
		       .optLabels(data.get(1))
		       .setSampling(20, true)
		       .build();
		dataset.prepare();
		return dataset;
	}
	
    

	
	
}
