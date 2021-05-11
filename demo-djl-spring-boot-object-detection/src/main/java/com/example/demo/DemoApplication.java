package com.example.demo;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Supplier;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.core.io.ClassPathResource;

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;

@SpringBootApplication
public class DemoApplication implements CommandLineRunner {

	public static void main(String[] args) {
		SpringApplication.run(DemoApplication.class, args);
	}

	@javax.annotation.Resource
	private Supplier<Predictor<Image,DetectedObjects>> predictorProvider;
	
	@Override
	public void run(String... args) throws Exception {
		System.out.println("Hello Detection");
		
		var resource=new ClassPathResource("/bus-lane-1.jpg");
		var image=ImageFactory.getInstance().fromInputStream(resource.getInputStream());
		
		try ( var predictor=predictorProvider.get()) {
			var detectedObjects=predictor.predict(image);
			detectedObjects.items().forEach( item -> System.out.println(item.getClassName()));
			image.drawBoundingBoxes(detectedObjects);
			image.save(Files.newOutputStream(Path.of("output.png")), "png");
		}
		
		
		
	}

}
