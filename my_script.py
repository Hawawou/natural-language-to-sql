import pandas as pd
import transformers
import torch
from datasets import load_dataset
import json

dataset = load_dataset("Salesforce/wikisql", trust_remote_code=True)

# split the dataset into training, validation and test
dataset_train = dataset['train'].shuffle().select(range(3000))
dataset_val = dataset['validation'].shuffle().select(range(1000))
dataset_test = dataset['test'].shuffle().select(range(1000))

# split the dataset into training and validation
#dataset = dataset_test['train'].train_test_split(test_size=0.2)

"""## Mistral 7b

## Preprocessing
"""

# forma data
import json
system_message = """You are a natural language to sql query translator model. Users will ask you a question in English and you will generate a SQL query based on the table provided: {table}"""

def format_data(dataset):

    # format table
    try:
        table_str = json.dumps(dataset["table"], indent=4)
        return {
        "messages": [
            {"role": "system", "content": system_message.format(table=table_str)},
            {"role": "user", "content": dataset["question"]},
            {"role": "assistant", "content": dataset["sql"]["human_readable"]}
        ]}
    except KeyError as e:
        print("Missing key in dataset: {e}")
        return None

train_data = dataset_train.map(format_data)
val_data = dataset_val.map(format_data)
test_data = dataset_test.map(format_data)

df = pd.DataFrame(train_data)
df2 = pd.DataFrame(val_data)
df3 = pd.DataFrame(test_data)

train_data = df["messages"].to_list()
val_data = df2["messages"].to_list()
test_data = df3["messages"].to_list()


from huggingface_hub import login

login(token='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')

from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, BitsAndBytesConfig, get_scheduler, DataCollatorWithPadding
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
from accelerate import Accelerator, init_empty_weights, infer_auto_device_map
from tqdm.auto import tqdm

# enable distributed training
accelerate = Accelerator()

checkpoint = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# oov
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# with init_empty_weights():
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=bnb_config, device_map={"":0})
# device_map = infer_auto_device_map(model, max_memory={"cuda": "2GiB", "cpu": "16GiB"})
# model = model.to(device_map)

from trl import setup_chat_format
model, tokenizer = setup_chat_format(model, tokenizer)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# parameters
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Lora config
lora_config = LoraConfig(
    r = 256,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

#layers = model.state_dict().keys()
#for name in layers:
#    print(name)


from torch.utils.data import DataLoader, Dataset


# tokenisation
def tokenize_function(dataset):
    encoding = tokenizer.apply_chat_template(dataset,
            tokenize=True,
            padding=True,
            truncation=True,
            max_length=512,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True)
    encoding["labels"] = encoding["input_ids"].clone()
    return encoding

train_data = tokenize_function(train_data)
#val_data = tokenize_function(val_data)
test_data = tokenize_function(test_data)


class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_data.items()}
        return item

train_dataset = TokenizedDataset(train_data)
#val_dataset = TokenizedDataset(val_data)
test_dataset = TokenizedDataset(test_data)



#add special tokens to shorter sequences
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=data_collator)
#val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=False, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False, collate_fn=data_collator)

#check
for batch in train_dataloader:
   {k: v.shape for k,v in batch.items()}


"""## Training"""

from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_train_steps = len(train_data) * num_epochs

accelerator = Accelerator()
train_data, model, optimizer = accelerator.prepare(train_data, model, optimizer)
#Training



lr_scheduler = get_scheduler(
            "linear",
            optimizer = optimizer,
            num_warmup_steps = 0,
            num_training_steps = num_train_steps,
        )

progress_bar = tqdm(range(num_train_steps))


model.train()
tr_loss = []
for epoch in range(num_epochs):
    for batch in train_dataloader:
        #batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
    
        #clear gradients
        optimizer.zero_grad()

        #compute gradients
        accelerator.backward(loss)
        #update weights
        optimizer.step()
        #update lr
        lr_scheduler.step()
       # scaler.update()
        progress_bar.update(1)

        tr_loss.append(loss.item())
    print(f"Epoch {epoch+1}/{num_epochs}, {loss.item()}")
progress_bar.close()

torch.save(model.state_dict(), 'natural_sql.pt')
