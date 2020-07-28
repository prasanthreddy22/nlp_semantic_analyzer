


from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader
import logging
from datetime import datetime
import sys

def nlptrain(premodel,ver,tr_data,te_data):
	
#### Just some code to print debug information to stdout
	logging.basicConfig(format='%(asctime)s - %(message)s',
                    	datefmt='%Y-%m-%d %H:%M:%S',
                    	level=logging.INFO,
                    	handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
	model_name = 'roberta-large-nli-stsb-mean-tokens'
	train_batch_size = 16
	num_epochs = 4
	model_save_path = ver
	sts_reader = STSDataReader('kt_datasets/kt_benchmark', normalize_scores=True)

# Load a pre-trained sentence transformer model
	model = SentenceTransformer(premodel)

# Convert the dataset to a DataLoader ready for training
	logging.info("")
	train_data = SentencesDataset(sts_reader.get_examples(tr_data), model)
	train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
	train_loss = losses.CosineSimilarityLoss(model=model)


	logging.info("")
	dev_data = SentencesDataset(examples=sts_reader.get_examples(te_data), model=model)
	dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
	evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)


# Configure the training. We skip evaluation in this example
	warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
	logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
	model.fit(train_objectives=[(train_dataloader, train_loss)],
          	evaluator=evaluator,
          	epochs=num_epochs,
          	evaluation_steps=1000,
         	warmup_steps=warmup_steps,
          	output_path=model_save_path)

	list=['model saved in '+ ver+' directory']

	return(list)





