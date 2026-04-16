import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset

class TransformerVARegressor(nn.Module):
    '''
    A BERT-based regressor for predicting Valence and Arousal scores.

    - Uses a pretrained BERT backbone to encode text.
    - Takes the [CLS] token representation as sentence-level embedding.
    - Adds a dropout layer and a linear head to output 2 values: [Valence, Arousal].
    - Includes helper methods for one training epoch and one evaluation epoch.

    Args:
        model_name (str): HuggingFace model name, default "bert-base-multilingual-cased".
        dropout (float): Dropout rate before the regression head.

    Methods:
        train_epoch(dataloader, optimizer, loss_fn, device):
            Train the model for one epoch.
            Returns average training loss.

        eval_epoch(dataloader, loss_fn, device):
            Evaluate the model for one epoch (no gradient).
            Returns average validation loss.
    '''
    def __init__(self, model_name, dropout=0.1, use_aspect_pooling=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(self.backbone.config.hidden_size, 2)  # Valence + Arousal
        self.use_aspect_pooling = use_aspect_pooling

    def forward(self, input_ids, attention_mask, aspect_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        if self.use_aspect_pooling and aspect_mask is not None:
            # Aspect Masked Mean Pooling
            aspect_mask = aspect_mask.unsqueeze(-1).expand_as(hidden_states).float()
            pooled = (hidden_states * aspect_mask).sum(dim=1) / (aspect_mask.sum(dim=1) + 1e-9)
        else:
            # 原始 [CLS] pooling
            pooled = hidden_states[:, 0]

        x = self.dropout(pooled)
        return self.reg_head(x)

    def train_epoch(self, dataloader, optimizer, loss_fn, device):
        self.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = self(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
        return total_loss / len(dataloader)

    def eval_epoch(self, dataloader, loss_fn, device):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                outputs = self(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                total_loss += loss.detach().item()
                # total_loss += loss.item()
        return total_loss / len(dataloader)

class VADataset(Dataset):
    '''
    A PyTorch Dataset for Valence–Arousal regression.

    - Combines aspect and text into a single input (e.g., "keyboard: The keyboard is good").
    - Tokenizes the input using a HuggingFace tokenizer.
    - Returns:
        * input_ids: token IDs, shape [max_len]
        * attention_mask: mask, shape [max_len]
        * labels: [Valence, Arousal], shape [2], float tensor

    Args:
        dataframe (pd.DataFrame): must contain "Text", "Aspect", "Valence", "Arousal".
        tokenizer: HuggingFace tokenizer.
        max_len (int): max sequence length.
    '''
    def __init__(self, dataframe, tokenizer, use_aspect_pooling=False, max_len=128):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()
        self.labels = dataframe[["Valence", "Arousal"]].values.astype(float)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_aspect_pooling = use_aspect_pooling

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        aspect = self.aspects[idx]
        text = self.sentences[idx]
        
        input_text = f"{aspect} : {text}"
        
        if self.use_aspect_pooling:
            encoded = self.tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
                return_offsets_mapping=True
            )
        else:
            encoded = self.tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }
        if self.use_aspect_pooling:
            aspect_mask = self._create_aspect_mask(
                encoded, 
                aspect, 
                input_text
            )
            item["aspect_mask"] = aspect_mask
        return item

    def _create_aspect_mask(self, encoded, aspect, input_text):
        offset_mapping = encoded["offset_mapping"].squeeze(0)  # [seq_len, 2]
        input_ids = encoded["input_ids"].squeeze(0)
        
        aspect_mask = torch.zeros(self.max_len, dtype=torch.long)
        
        aspect_tokens = self.tokenizer.encode(aspect, add_special_tokens=False)
        
        if len(aspect_tokens) == 0:
            return aspect_mask

        for i in range(len(input_ids) - len(aspect_tokens) + 1):
            if input_ids[i:i+len(aspect_tokens)].tolist() == aspect_tokens:
                aspect_mask[i:i+len(aspect_tokens)] = 1
                break

        if aspect_mask.sum() == 0:
            aspect_lower = aspect.lower()
            text_lower = input_text.lower()
            start = text_lower.find(aspect_lower)
            if start != -1:
                for j in range(len(offset_mapping)):
                    if offset_mapping[j][0] <= start < offset_mapping[j][1]:
                        end_idx = min(j + 8, self.max_len)
                        aspect_mask[j:end_idx] = 1
                        break

        return aspect_mask