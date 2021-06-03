package de.ebamberg.djl.demo.timeseries.http;

import java.io.FileNotFoundException;
import java.io.IOException;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

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

@RestController
public class StandardController {

	private static Logger log=LoggerFactory.getLogger(StandardController.class);
	
	@Autowired
	private CSVDataSource dataSource;

	@Autowired
	private TrainerService trainerService;
	
	@Autowired
	private ModelRepository modelRepository;
	
	@Autowired
	private BlockFactory blockFactory;
	
	@Autowired
	private Translator<float[][],float[]> translator;
	
	@Value ("${timeseries.lookback:4}")
	private int lookback;
	
	@Value ("${timeseries.epochs:200}")
	private int epochs;
	
	@Value ("${timeseries.modelname:savedmodel}")
	private String modelName;
	
	@Autowired
	private ObjectMapper jsonMapper;
	

	private Model modelForPrediction;

	
	@PostConstruct
	public void loadModelOnStartup() throws IOException, MalformedModelException {
		modelForPrediction=createEmptyModel( modelName );
		try {
		modelRepository.load(modelForPrediction);
		} catch (FileNotFoundException ex) {
			log.warn("no saved model found");
		}
	}
	
	@PreDestroy
	public void unloadModelBeforeShutdown() {
		if (modelForPrediction!=null) {
			modelForPrediction.close();
		}
	}
	
	
	/**
	 * curl -X POST http://127.0.0.1:8080/timeseries/train -F "data=@./src/main/resources/airline-passengers.csv"
 
	 * @param file
	 * @return
	 * @throws Exception
	 */
	@PostMapping (value="timeseries/train")
	public String upload(@RequestParam("data") MultipartFile file) throws Exception {
		float[][] rawData=dataSource.read(file.getInputStream());
		String result="";
		try(var manager = NDManager.newBaseManager(); var model=createEmptyModel( modelName ) ) {
			// preparing the dataset
			var dataset=buildFramedTimeSeriesDataSet(manager, model,  rawData, lookback);			
			//train the dataset
			trainerService.train(model, dataset,epochs);
			// store the model 
			modelRepository.store(model);

		}
		return result;
	}

	private Model createEmptyModel(String name) {
		var model=Model.newInstance(name);
		model.setBlock(blockFactory.newBlock(null));
		return model;
	}
	
	/**
	 * curl -X GET http://127.0.0.1:8080/timeseries/predict
	 * @return
	 * @throws TranslateException
	 * @throws JsonProcessingException 
	 * @throws JsonMappingException 
	 */
	@GetMapping (value="timeseries/predict")
	public float[][] predict() throws TranslateException, JsonMappingException, JsonProcessingException {
		int lookforward=5;
		
		var lastSample=jsonMapper.readValue(modelForPrediction.getProperty("lastSample"),float[].class);
		
		var data=new float[lastSample.length][] ;
		for (int i=0; i<lastSample.length; i++) {
			data[i]=new float[] {lastSample[i]};
		}
		
	//	data=new float[][] {{472.0f}, {535.0f}, {622.0f}, {606.0f}};

		var predictor=modelForPrediction.newPredictor(translator);
		
		float[][] result=new float[lookforward][];
		for (int i=0;i<lookforward;i++) {
			result[i]=predictor.predict(data);
			
			for (int j=1;j<data.length;j++) {
				data[j-1]=data[j];
			}
			data[data.length-1]=result[i];
			
		}
		return result;
	}
	

	private Dataset buildFramedTimeSeriesDataSet(NDManager manager , Model model, float[][] rawData,int lookback) throws IOException, TranslateException {
		MinMaxScaler scaler=new MinMaxScaler();
		TimeSeriesDataPreparator timeSeriesDataPreparator=new TimeSeriesDataPreparator(manager, model,scaler);
		var data= timeSeriesDataPreparator.prepare(rawData, lookback);
		model.setProperty("minScale", jsonMapper.writeValueAsString(scaler.getMin().toFloatArray()));
		model.setProperty("maxScale", jsonMapper.writeValueAsString(scaler.getMax().toFloatArray()));
		model.setProperty("lastSample", jsonMapper.writeValueAsString( scaler.inverseTransform(data.get(0).get(-1)).toFloatArray() ));
		
		var	 dataset = new ArrayDataset.Builder()
		       .setData(data.get(0))
		       .optLabels(data.get(1))
		       .setSampling(20, true)
		       .build();
		dataset.prepare();
		return dataset;
	}
	
    

	
	
}
