from datasets import load_dataset
from lightautoml.automl.presets.text_presets import TabularNLPAutoML
from lightautoml.tasks import Task
import numpy as np

# Load the dataset from Hugging Face
dataset = load_dataset("DeepPavlov/banking77")

# Split the data into train and test sets
data_train = dataset['train'].to_pandas()
data_test = dataset['test'].to_pandas()

# Initialize and train LightAutoML
roles = {'target': 'label'}  # Define the target column

automl = TabularNLPAutoML(
    task=Task("multiclass"),
    timeout=3600,
    cpu_limit=1,
    gpu_ids='0',
    general_params={
        'nested_cv': False,
        'use_algos': [['nn']]
    },
    autonlp_params={
        'sent_scaler': 'l2'
    },
    text_params={
        'lang': 'en',
        'bert_model': 'prajjwal1/bert-tiny'
    },
    nn_params={
        'opt_params': {'lr': 1e-5},
        'max_length': 128,
        'bs': 32,
        'n_epochs': 7,
    }

)


model = automl.fit_predict(data_train, roles=roles)

# Predict on the test set
test_predictions = automl.predict(data_test)

# Evaluate the results
import code; code.interact(local=locals())
evaluation_results = (data_test["label"].to_numpy() == np.argmax(test_predictions.data, axis=1)).mean()
print("Evaluation Results:", evaluation_results)
