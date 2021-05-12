package com.example.demo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

@SpringBootApplication
@RestController
public class DemoDjlSpringBootTextSimilarityApplication {

    private static final String[] internetExample = new String [] {
        	"How can I increase the speed of my internet connection while using a VPN?",
        	"How can i increase speed of internet ?",
        	"Why is my internet so slow?",
        	"Where is London?"
        };
        
        private static final String[] newspaperExample = new String [] {
        		null,
    	    	"Obama speaks to the media in Illinois",
    	    	"The President greets the press in Chicago",
    	    	"The President speaks to the newspapers in Illinois",
    	    	"The President speaks to the media in Washington",
    	    	"Obama greets the press in Illinois",
    	    	"The Queen of England writes a letter to Obama",
    	    	"Boris Johnson wrote on twitter."
    	    };
	
	public static void main(String[] args) {
		SpringApplication.run(DemoDjlSpringBootTextSimilarityApplication.class, args);
	}
	

	@GetMapping (value="isSimilar")
	public Map<String,Float> isSimilar(@RequestParam String sentence) throws Exception {
        String[] inputs = newspaperExample;
        
        
        
        var inputsAndSentence=inputs.clone();
        inputsAndSentence[0]=sentence;

    	
        String modelUrl =
                "https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder/4.tar.gz";

        Criteria<String[], List> criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.TEXT_EMBEDDING)
                        .setTypes(String[].class, List.class)
                        .optModelUrls(modelUrl)
                        .optTranslator(new CosineSimilarityTranslator())
                        .optProgress(new ProgressBar())
                        .build();
        try (ZooModel<String[], List> model = ModelZoo.loadModel(criteria);
                Predictor<String[], List> predictor = model.newPredictor()) {
            List result=predictor.predict(inputsAndSentence);
            
            var resultMap = mapResultToMap(inputsAndSentence, result);
            return resultMap;
        }

	}


	private HashMap<String, Float> mapResultToMap(String[] inputs, List result) {
		var resultMap=new HashMap<String,Float>();
		System.out.println("similarities to sentence:"+inputs[0]);
		IntStream.range(1, inputs.length).forEach( i -> {
			System.out.println(result.get(i)+"\t=>\t"+"\t"+inputs[i]);
			resultMap.put(inputs[i], (Float)result.get(i));
		});
		return resultMap;
	}
	
	
	
    private static final class CosineSimilarityTranslator implements Translator<String[], List> {

        @Override
        public NDList processInput(TranslatorContext ctx, String[] inputs) {
            // manually stack for faster batch inference
            NDManager manager = ctx.getNDManager();
            NDList inputsList =
                    new NDList(
                            Arrays.stream(inputs)
                                    .map(manager::create)
                                    .collect(Collectors.toList()));
            return new NDList(NDArrays.stack(inputsList));
        }

        @Override
        public List<Float> processOutput(TranslatorContext ctx, NDList list) {
            NDList result = new NDList();
            long numOutputs = list.singletonOrThrow().getShape().get(0);
            for (int i = 0; i < numOutputs; i++) {
                result.add(list.singletonOrThrow().get(i));
            }
            
            List<Float> similarities=new ArrayList<>(result.size());
           
            similarities.add(1f);
            for (int i=1; i<result.size(); i++) {
        	similarities.add(cosineSimilarity(result.get(0),result.get(i)));
            }
            
            return similarities;
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
        
        public static float cosineSimilarity(NDArray vectorA, NDArray vectorB) {
            // NDArray dotProduct = vectorA.dot(vectorB); // not implemented in tensorflow
            NDArray dotProduct = vectorA.mul(vectorB).sum();    
            NDArray normA = vectorA.pow(2).sum().sqrt();
            NDArray normB = vectorB.pow(2).sum().sqrt(); 
            NDArray cosineSim=dotProduct.div (normA.mul(normB));
            return cosineSim.getFloat();        
        }
        
        
    }

	
	
}
