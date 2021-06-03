package de.ebamberg.djl.demo.timeseries.training;

import java.io.IOException;
import java.util.concurrent.CompletableFuture;

import org.springframework.stereotype.Component;

import ai.djl.Model;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.translate.TranslateException;

@Component
public class TrainerService {

	public Float train(Model model, Dataset dataset,int epochs) throws IOException, TranslateException {
		
				long timestepsPerSample=Long.valueOf(model.getProperty("lookback"));
				long number_of_features=Long.valueOf(model.getProperty("features"));
		
		 	   TrainingConfig trainingConfig=new DefaultTrainingConfig(Loss.l2Loss())
					.optOptimizer(Optimizer.adam().build())
					.addTrainingListeners(TrainingListener.Defaults.logging());
		 	   
		 	  Float loss;
		 	   /** now create a trainer */
		 	   try (Trainer trainer=model.newTrainer(trainingConfig)) {
		 	       trainer.initialize(new Shape(1,timestepsPerSample,number_of_features));
		 	       EasyTrain.fit(trainer, epochs, dataset,null);
		 	       loss=trainer.getTrainingResult().getTrainLoss();
		 	   }  
		 	   return loss;
	}
	
}
