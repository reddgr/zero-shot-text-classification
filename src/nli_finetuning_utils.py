import pandas as pd
import random
from datasets import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import cuda

class NLIModelFineTuner:
    def __init__(self, dataset, labels, model, tokenizer):
        self.dataset = dataset
        self.labels = labels
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = None
        self.eval_dataset = None


    def tokenize_and_format_nli_dataset(self, template="This example is a {} prompt.", max_length=128, eval_proportion=0.4):
        
        # Convert the dataset to a Pandas DataFrame
        df = self.dataset.to_pandas()

        input_ids, attention_masks, labels, input_sentences = [], [], [], []

        for index, row in df.iterrows():
            text = row["text"]
            category = row["class"] 
            label = row["label"]

            # Construct text with template
            nli_text = f"<prompt>{text}</prompt> {template.format(category)}"
            input_sentences.append(nli_text)

            # Tokenize example
            encoding = self.tokenizer(
                nli_text,
                max_length=max_length - 1,  # Leave space for <eos> token
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )

            input_ids.append(encoding["input_ids"].squeeze(0))
            attention_masks.append(encoding["attention_mask"].squeeze(0))
            labels.append(label)

        # Create tokenized dataset
        full_dataset = Dataset.from_dict({
            "input_ids": [seq.tolist() for seq in input_ids],
            "attention_mask": [mask.tolist() for mask in attention_masks],
            "labels": labels,
            "input_sentence": input_sentences,
        })

        # Split dataset
        shuffled_dataset = full_dataset.shuffle(seed=42)
        split_index = int(len(shuffled_dataset) * (1 - eval_proportion))
        
        self.train_dataset = shuffled_dataset.select(range(split_index))
        self.eval_dataset = shuffled_dataset.select(range(split_index, len(shuffled_dataset)))

        return self.train_dataset, self.eval_dataset, full_dataset



    def tokenize_and_create_contradictions(self, template="This example is {}.", num_contradictions=2, max_length=128, eval_proportion=0.2):
        """
        Tokenizes and formats the dataset for fine-tuning, creating contradiction examples.

        Args:
            template (str): Template for entailment and contradiction examples.
            num_contradictions (int): Number of extra contradiction examples per input.
            max_length (int): Maximum sequence length for padding/truncation.
            eval_proportion (float): Proportion of data to use for evaluation (0-1).
            random_seed (int): Seed for reproducible splitting.

        Returns:
            tuple: (train_dataset, eval_dataset)
        """
        # Convert the dataset to a Pandas DataFrame
        df = self.dataset.to_pandas()
        input_ids, attention_masks, labels, input_sentences = [], [], [], []

        for index, row in df.iterrows():
            text = row["text"]
            label = row["class"]

            # Construct original entailment sentence
            entailment_text = f"<prompt>{text}</prompt> {template.format(label)}"
            input_sentences.append(entailment_text)

            # Tokenize entailment example
            entailment_encoding = self.tokenizer(
                entailment_text,
                max_length=max_length - 1,  # Leave space for <eos> token
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )

            input_ids.append(entailment_encoding["input_ids"].squeeze(0))
            attention_masks.append(entailment_encoding["attention_mask"].squeeze(0))
            labels.append(2)  # Entailment label

            # Construct and tokenize contradiction examples
            possible_contradictions = [x for x in self.labels if x != label]
            selected_contradictions = random.sample(possible_contradictions, num_contradictions)
            for contradiction_label in selected_contradictions:
                contradiction_text = f"<prompt>{text}</prompt> {template.format(contradiction_label)}"
                input_sentences.append(contradiction_text)

                contradiction_encoding = self.tokenizer(
                    contradiction_text,
                    max_length=max_length - 1,  # Leave space for <eos> token
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                )

                input_ids.append(contradiction_encoding["input_ids"].squeeze(0))
                attention_masks.append(contradiction_encoding["attention_mask"].squeeze(0))
                labels.append(0)  # Contradiction label

        # Create full dataset
        full_dataset = Dataset.from_dict({
            "input_ids": [seq.tolist() for seq in input_ids],
            "attention_mask": [mask.tolist() for mask in attention_masks],
            "labels": labels,
            "input_sentence": input_sentences,
        })

        # Split dataset
        shuffled_dataset = full_dataset.shuffle(seed=42)
        split_index = int(len(shuffled_dataset) * (1 - eval_proportion))
        
        self.train_dataset = shuffled_dataset.select(range(split_index))
        self.eval_dataset = shuffled_dataset.select(range(split_index, len(shuffled_dataset)))

        return self.train_dataset, self.eval_dataset, full_dataset
    
    def fine_tune(self, output_dir="./results", epochs=3, batch_size=8, learning_rate=5e-5):
        """
        Fine-tune the NLI model.

        Args:
            output_dir (str): Directory to save model checkpoints.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
        """
        device = "cuda:0" if cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.model = self.model.to(device)

        # Verify that the dataset is PyTorch-compatible
        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10,
            save_total_limit=2,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            eval_strategy="epoch",
            #report_to="none",
        )

        # Create GenerationConfig to avoid warning
        # generation_config = GenerationConfig(forced_eos_token_id=2)
        # self.model.generation_config = generation_config

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )

        print("Fine-tuning in progress...")
        trainer.train()
        print(f"Fine-tuning complete. Model saved to {output_dir}.")

        print(f"Last checkpoint {trainer.state.global_step}")

        return trainer
    
    def fine_tune_with_cosine_annealing(self, output_dir="./results", epochs=6, batch_size=8, base_learning_rate=1e-4, t_max=3):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.model = self.model.to(device)

        # Prepare dataset
        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Set up optimizer
        optimizer = AdamW(self.model.parameters(), lr=base_learning_rate)

        # Set up cosine annealing scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
        
        # Custom callback to log learning rate
        class LogLearningRateCallback(TrainerCallback):
            def __init__(self, scheduler):
                self.scheduler = scheduler
            def on_epoch_begin(self, args, state, control, **kwargs): 
                # self.scheduler = kwargs.get("scheduler")
                if self.scheduler is not None:
                    print("Last learning rate:", end=" ")
                    lr = self.scheduler.get_last_lr()[0]
                    print(lr)
 

        # Custom training loop
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10,
            save_total_limit=2,
            logging_dir=f"{output_dir}/logs",
            eval_strategy="epoch"
        )

        print_lr_callback = LogLearningRateCallback(scheduler)

        # Custom callback to print epoch-end metrics
        class EpochEndCallback(TrainerCallback):
            def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
                print(f"\nEpoch {state.epoch} complete.")
                # Doesn't print anything currently. 
                # Need to modify Trainer class so train loss is calculated in between epochs
                # https://github.com/huggingface/transformers
                if metrics:
                    print(f"\nEpoch {state.epoch} metrics:")
                    for key, value in metrics.items():
                        print(f"{key}: {value}")
                    print("-" * 50)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            optimizers=(optimizer, scheduler),  # Pass both optimizer and scheduler
            callbacks=[print_lr_callback, EpochEndCallback()]
        )

        print("Fine-tuning in progress...")
        train_results = trainer.train()
        print(f"Fine-tuning complete. Model saved to {output_dir}.")

        print(f"Last checkpoint {trainer.state.global_step}")

        return trainer, train_results