import torch
import argparse
import os
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from datasets import Dataset
from tw_rouge import get_rouge

import transformers
from transformers import AdamW, Adafactor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, get_scheduler

from modules.utils import log, set_seed, read_data

parser = argparse.ArgumentParser(description="Finetune a transformers model on a title generation task")
parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
parser.add_argument("--max_text_length", type=int, default=512)
parser.add_argument("--max_title_length", type=int, default=64)
parser.add_argument("--source_prefix", type=str, default=None)
parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_restarts", help="The type of learning rate scheduler to use.")
parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate (after the potential warmup period) to use.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
parser.add_argument("--num_train_epochs", type=int, default=24, help="Total number of training epochs to perform.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--logging_step", type=int, default=100)
parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
args = parser.parse_args()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Seed for reproducibility
set_seed(args.seed)

# load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)

# accelerator
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="fp16")

# load data
trainFile = read_data(args.train_file)
train_dataset = Dataset.from_list(trainFile)

# tokenize
prefix = args.source_prefix
def preprocess_function(example):
    inputs = example['maintext']
    targets = example['title']
    inputs = [prefix + doc for doc in inputs]
    model_inputs = tokenizer(inputs, max_length = args.max_text_length, padding = 'max_length', truncation = True)
    labels = tokenizer(targets, max_length = args.max_title_length, padding = 'max_length', truncation = True)

    if args.ignore_pad_token_for_loss:
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

columns = ['date_publish', 'title', 'source_domain', 'maintext', 'split', 'id']
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=columns)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=data_collator)

# Optimizer
optimizer = Adafactor(model.parameters(), lr=args.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)

# Scheduler
num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
num_warmup_steps = 0.1 * max_train_steps
scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps * accelerator.num_processes,
    num_training_steps=max_train_steps
)

# Training
model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)

log('Strat training')
for epoch in range(args.num_train_epochs):
    step = 1
    train_loss = 0
    model.train()
    preds, labels = [], []
    for batch in tqdm(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            train_loss += loss.item()

            decoded_preds = tokenizer.batch_decode(outputs.logits.argmax(-1), skip_special_tokens=True)
            filtered_labels = [[token for token in label if token != -100] for label in batch['labels']]
            decoded_labels = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)
            preds.extend(decoded_preds)
            labels.extend(decoded_labels)
        
        if step % args.logging_step == 0:
            rouge_scores = get_rouge(preds, labels)
            log(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss / args.logging_step:.3f} | ROUGE-1 = {rouge_scores['rouge-1']['f']:.3f} | ROUGE-2 = {rouge_scores['rouge-2']['f']:.3f} | ROUGE-L = {rouge_scores['rouge-l']['f']:.3f}")
            train_loss = 0
            preds, labels = [], []

    log("Save model")
    root = args.output_dir
    model.save_pretrained(os.path.join(root, f'checkpoint_{epoch}'))