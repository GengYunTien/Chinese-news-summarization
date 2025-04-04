import torch
import argparse
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from datasets import Dataset

import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import set_seed, read_data

parser = argparse.ArgumentParser(description="Inference a transformers model on a title generation task")
parser.add_argument("--test_file", type=str, default=None, help="A csv or a json file containing the training data.")
parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
parser.add_argument("--max_source_length", type=int, default=512)
parser.add_argument("--max_title_length", type=int, default=64)
parser.add_argument("--output_file", type=str, default=None, help="Where to store the final model.")
parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
args = parser.parse_args()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Seed for reproducibility
set_seed(args.seed)

# load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)

# load data
testFile = read_data(args.test_file)
test_dataloader = DataLoader(testFile, batch_size=1)

gen_kwargs = {
    "max_length": args.max_title_length,
    "num_beams": 5,
}

results = []
model.eval()
for batch in test_dataloader:
    input_text = batch["maintext"]
    ID = batch["id"]
    inputs = tokenizer(input_text, max_length = args.max_source_length, return_tensors="pt", padding='max_length', truncation=True).to(model.device)

    with torch.no_grad():
        generated_tokens = model.generate(**inputs, **gen_kwargs)
        decoded_pred = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    result = {
        "title": decoded_pred,
        "id": ID[0]
    }
    results.append(result)

with open(args.output_file, "w", encoding='utf-8') as f:
    for result in results:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')