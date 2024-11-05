from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

model_name = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("jinaai/code_exercises", split="train[:200]")

synthetic_data = []
for example in ds:
    problem = example["problem"]
    solution = example["solution"]
    kotlin_translation = {
        "input": f"// Task: {problem}\nfun main() {{\n{solution}\n}}"
    }
    synthetic_data.append(kotlin_translation)

def tokenize_data(data, tokenizer):
    inputs = [item['input'] for item in data]
    tokenized_inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")
    labels = tokenized_inputs["input_ids"].clone()
    labels[tokenized_inputs["attention_mask"] == 0] = -100  # Mask padding tokens
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_data = tokenize_data(synthetic_data, tokenizer)

class KotlinDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return {key: torch.tensor(value) for key, value in item.items()}

train_dataset = KotlinDataset(tokenized_data)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

test_data = ["fun sum(a: Int, b: Int): Int = a + b"]

model.eval()
for task in test_data:
    inputs = tokenizer(task, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        print("Generated solution:", tokenizer.decode(outputs[0], skip_special_tokens=True))
