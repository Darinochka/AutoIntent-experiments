from autointent import load_dataset
import json
import pandas as pd
import os

from autointent.generation.utterances import UtteranceEvolver
from autointent.generation import Generator
from autointent.generation.chat_templates import EVOLUTION_MAPPING


model_name = "gpt-3.5-turbo-0125"
os.environ["OPENAI_MODEL_NAME"] = model_name
os.environ["OPENAI_BASE_URL"] = 'http://193.187.173.33:8002/api/providers/openai/v1'
os.environ["OPENAI_API_KEY"] = 'InnPracAutoIntent:Darina_Rustamova:69fb2c7dbd044ede970b02132d5ea9bb'


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
    "DeepPavlov/banking77_ru",
    "DeepPavlov/clinc150_ru",
    "DeepPavlov/banking77",
    "DeepPavlov/snips",
    "DeepPavlov/snips_ru"
]

datasets_subset = []

for dataset_name in datasets:
    _, output_file_subset = create_subset(dataset_name, max_utterances_per_intent=10)
    datasets_subset.append(output_file_subset)

evolution_names = ["abstract", "concrete", "formal", "funny", "goofy", "informal", "reasoning"]
evolution_templates = [EVOLUTION_MAPPING[name] for name in evolution_names]
seed = 42
split = "train"
n_evolutions = 10

if not os.path.exists(f"datasets/{model_name.split('/')[-1]}"):
    os.makedirs(f"datasets/{model_name.split('/')[-1]}")

for dataset_subset in datasets_subset:
    print(dataset_subset)
    dataset = load_dataset(dataset_subset)
    n_before = len(dataset[split])
    output_path = f"datasets/{model_name.split('/')[-1]}/{dataset_subset.split('/')[-1].replace('.json', '_augmented.json')}"
    
    if os.path.isfile(output_path):
        continue
    
    generator = UtteranceEvolver(Generator(), evolution_templates, seed, async_mode=True)
    _ = generator.augment(dataset, split_name=split, n_evolutions=n_evolutions, batch_size=32)

    dataset.to_json(output_path)