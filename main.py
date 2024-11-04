from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

# Загрузка легковесной модели и токенизатора
model_name = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Устанавливаем токен паддинга (используем eos_token в качестве pad_token)
tokenizer.pad_token = tokenizer.eos_token

# Загрузка части датасета
ds = load_dataset("jinaai/code_exercises", split="train[:200]")  # Загружаем только 1% для быстрого эксперимента

# Преобразование примеров на Kotlin
synthetic_data = []
for example in ds:
    problem = example["problem"]
    solution = example["solution"]

    kotlin_translation = {
        "input": f"// Задача: {problem}\nfun main() {{\n{solution}\n}}"
    }
    synthetic_data.append(kotlin_translation)


# Функция для токенизации данных с выравненными метками
def tokenize_data(data, tokenizer):
    inputs = [item['input'] for item in data]
    tokenized_inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")

    # Создаем метки, используя те же токены, что и для входных данных
    labels = tokenized_inputs["input_ids"].clone()
    labels[tokenized_inputs["attention_mask"] == 0] = -100  # Маскируем токены паддинга

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Токенизация с выравненными метками
tokenized_data = tokenize_data(synthetic_data, tokenizer)


# Создание пользовательского Dataset
class KotlinDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return {key: torch.tensor(value) for key, value in item.items()}


train_dataset = KotlinDataset(tokenized_data)

# Настройка Trainer для дообучения модели
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,  # Для быстрого эксперимента можно начать с 1 эпохи
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Запуск обучения
trainer.train()

# Пример задачи для оценки
test_data = ["fun sum(a: Int, b: Int): Int = a + b"]

# Оценка производительности модели
model.eval()
for task in test_data:
    inputs = tokenizer(task, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        print("Generated solution:", tokenizer.decode(outputs[0], skip_special_tokens=True))
