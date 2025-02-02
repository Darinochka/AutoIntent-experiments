from autointent import load_dataset
import json
import pandas as pd
import os

from autointent.generation.utterances.evolution.evolver import UtteranceEvolver
from autointent.generation.utterances.generator import Generator
from autointent.generation.utterances.evolution.chat_templates import (
    AbstractEvolution,
    ConcreteEvolution,
    FormalEvolution,
    FunnyEvolution,
    GoofyEvolution,
    InformalEvolution,
    ReasoningEvolution,
)
import os

os.environ["OPENAI_MODEL_NAME"] = 'Qwen/Qwen2.5-7B-Instruct-AWQ'
os.environ["OPENAI_BASE_URL"] = 'http://localhost:8000/v1'
os.environ["OPENAI_API_KEY"] = 'sth'


def create_subset(dataset_name: str, max_utterances_per_intent: int = 10):
    output_file = f"datasets/{dataset_name.split('/')[-1]}.json"
    output_file_subset = output_file.replace(".json", f"_subset_{max_utterances_per_intent}.json")

    if os.path.isfile(output_file_subset):
        return output_file, output_file_subset
    
    dataset = load_dataset(dataset_name)
    dataset.to_json(output_file)
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data["train"])
    sampled_df = df.groupby("label").head(max_utterances_per_intent)
    data["train"] = sampled_df.to_dict(orient="records")

    with open(output_file_subset, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return output_file, output_file_subset

datasets = [
    "AutoIntent/banking77_ru",
    "AutoIntent/clinc150_ru",
    "AutoIntent/banking77",
    "AutoIntent/snips",
    "AutoIntent/snips_ru"
]

datasets_subset = []

for dataset_name in datasets:
    _, output_file_subset = create_subset(dataset_name, max_utterances_per_intent=10)
    datasets_subset.append(output_file_subset)

evolutions = [
        AbstractEvolution(),
        ConcreteEvolution(),
        FormalEvolution(),
        FunnyEvolution(),
        GoofyEvolution(),
        InformalEvolution(),
        ReasoningEvolution()
]
seed = 42
split = "train"
n_evolutions = 10


for dataset_subset in datasets_subset:
    print(dataset_subset)
    dataset = load_dataset(dataset_subset)
    n_before = len(dataset[split])
    output_path = dataset_subset.replace(".json", "_augmented.json") 

    generator = UtteranceEvolver(Generator(), evolutions, seed, async_mode=True)
    _ = generator.augment(dataset, split_name=split, n_evolutions=n_evolutions, batch_size=1)

    dataset.to_json(output_path)