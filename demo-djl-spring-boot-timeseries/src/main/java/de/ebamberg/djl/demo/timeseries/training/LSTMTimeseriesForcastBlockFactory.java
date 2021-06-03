package de.ebamberg.djl.demo.timeseries.training;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.BlockFactory;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.LSTM;

public class LSTMTimeseriesForcastBlockFactory implements BlockFactory {

	private static final long serialVersionUID = -2352354802294420849L;

	@Override
	public Block newBlock(NDManager manager) {
		SequentialBlock block = new SequentialBlock();
		block.add(LSTM.builder().setStateSize(128).setNumLayers(1).optDropRate(0).optReturnState(false).build());
		block.add(Activation.reluBlock());
		block.add(Blocks.batchFlattenBlock());
		block.add(Linear.builder().setUnits(1).build());
		return block;
	}

}
