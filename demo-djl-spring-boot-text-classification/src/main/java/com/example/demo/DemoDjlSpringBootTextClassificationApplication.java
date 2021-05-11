package com.example.demo;

import java.util.function.Supplier;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications.Classification;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.translate.TranslateException;

@SpringBootApplication
@RestController
public class DemoDjlSpringBootTextClassificationApplication {

    private static final String input="The girl eat the apple";
	
	public static void main(String[] args) {
		SpringApplication.run(DemoDjlSpringBootTextClassificationApplication.class, args);
	}
	
	@javax.annotation.Resource
	private Supplier<Predictor<String,Classification>> predictorProvider;

	@GetMapping (value="predict")
	public void predict() throws TranslateException {
		try (var predictor=predictorProvider.get()) {
			var classification=predictor.predict(input);
		}
	}
	
}
