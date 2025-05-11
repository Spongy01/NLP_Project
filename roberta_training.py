import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    AutoConfig,
    AutoModel,
)
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


model_id = "roberta-base"
# dataset_id = "qiaojin/PubMedQA"
dataset = load_dataset("qiaojin/PubMedQA","pqa_artificial")

dataset = dataset["train"].select(range(18000))
dataset
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
label2id = {"yes": 0, "no": 1, "maybe": 2}
id2label = {v: k for k, v in label2id.items()}

class RoBERTaModel(torch.nn.Module):
  def __init__(self, num_labels):
    super().__init__()
    self.base_model = AutoModel.from_pretrained('distilroberta-base')
    self.classifier = torch.nn.Linear(self.base_model.config.hidden_size, num_labels)

  def forward(self, input_ids, attention_mask):
    outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state
    pooled_output = hidden_states[:, 0, :]
    logits = self.classifier(pooled_output)
    return logits 
  

def preprocess(example):

    final_prompt = f"{example['question']}\n{example['long_answer']}"
    inputs = tokenizer(
        final_prompt,
    )
    inputs["labels"] = label2id[example["final_decision"]]
    return inputs

encoded_dataset = dataset.map(preprocess)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


model = RoBERTaModel(num_labels=len(label2id))
model.to("cuda")

def finetune_roberta_classifier(model, train_loader, num_epochs, lr, weight_decay, device="cuda"):
    """
    Function to fine-tune a distilRoberta model for a classification task
    Args:
    model: instance of distilRoberta
    train_loader: Dataloader for the BoolQ training set
    num_epochs: Number of epochs for training
    lr: Learning rate
    weight_decay: Weight decay
    ...: Any other arguments you may need

    Returns:
    model: Fine-tuned model
    batch_losses: List of losses for each mini-batch
    """
    #<FILL IN>
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    batch_losses = []

    # use binary cross entropy loss
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)




    for epoch in range(num_epochs):
      for batch in tqdm(train_loader):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.amp.autocast(device_type=device, dtype=torch.float16):
          logits = model(input_ids=input_ids, attention_mask=attention_mask)

          optimizer.zero_grad()
          # logits = logits.view(-1)
          # labels = labels.type(torch.long)
          loss = criterion(logits, labels)
          batch_losses.append(loss.item())
          loss.backward()
          optimizer.step()

    return model, batch_losses



def make_tokenize(tokenizer):
    def tokenize_boolq_evaluation(examples):
        return tokenizer(f"{examples['question']}.\n{examples['long_answer']}",truncation=True)
    return tokenize_boolq_evaluation

def roberta_collator(batch, tokenizer):
  input_ids = pad_sequence([torch.tensor(x['input_ids']) for x in batch], batch_first=True,padding_value=tokenizer.pad_token_id)
  attention_mask = pad_sequence([torch.tensor(x['attention_mask']) for x in batch], batch_first=True, padding_value=0)
  labels = torch.tensor([x['labels'] for x in batch])

  return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}



# tokenized_train_set = dataset.map(make_tokenize(tokenizer), batched=False)
train_loader = DataLoader(encoded_dataset, batch_size=8, collate_fn=lambda batch: roberta_collator(batch, tokenizer))

model, batch_losses = finetune_roberta_classifier(model, train_loader, num_epochs=1, lr=1e-5, weight_decay=2e-4)