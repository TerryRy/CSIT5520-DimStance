from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import pandas as pd

from utils import load_jsonl, jsonl_to_df, get_prd, df_to_jsonl, evaluate_predictions_task1, plot_loss_curve, plot_scatter
from model import TransformerVARegressor, VADataset

if __name__ == '__main__':
    #task config
    subtask = "subtask_1"#don't change
    task = "task1"#don't change
    lang = "eng" #chang the language you want to test
    domain = "environmental_protection" #change what domain you want to test
    train_url = f"./task-dataset/track_b/{subtask}/{lang}/{lang}_{domain}_train_task1.jsonl"
    predict_url = f"./task-dataset/track_b/{subtask}/{lang}/{lang}_{domain}_dev_{task}.jsonl"

    #model config
    model_name = "bert-base-multilingual-cased"
    # model_name = "answerdotai/ModernBERT-base"
    # model_name = "microsoft/deberta-v3-base"
    is_eval = False
    pooling = False
    batch_size = 64
    # 由于我事先没想到要加batch_size的组，所以保存文件名上没有对无掩码32组和64组进行区分。现有的文件是生成后手动修改文件名，还请注意不要误替换

    safe_model_name = model_name.replace("/", "_")
    checkpoint_path = f"./checkpoints/{safe_model_name}_{pooling}.pth"
    # chage your transformer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVARegressor(model_name=model_name, use_aspect_pooling=pooling)
    if (is_eval):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model.eval()
    model = model.to(device)
    tokenizer = model.tokenizer
    
    # Dataloader building
    train_raw = load_jsonl(train_url)
    predict_raw = load_jsonl(predict_url)

    train_df = jsonl_to_df(train_raw)
    predict_df = jsonl_to_df(predict_raw)

    # split 10% for dev
    train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    train_dataset = VADataset(train_df, tokenizer, use_aspect_pooling=pooling)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    dev_dataset = VADataset(dev_df, tokenizer, use_aspect_pooling=pooling)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    
    # paramter
    lr = 1e-5 #learning rate
    epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if (not is_eval):
        # training
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            train_loss = model.train_epoch(train_loader, optimizer, loss_fn, device)
            val_loss = model.eval_epoch(dev_loader, loss_fn, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"model:{safe_model_name} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
            
        torch.save(model.state_dict(), checkpoint_path)

        # 保存 loss 数据
        loss_df = pd.DataFrame({
            'Epoch': range(1, len(train_losses) + 1),
            'Train_Loss': train_losses,
            'Val_Loss': val_losses
        })
        loss_df.to_csv(f"./loss_curve/{safe_model_name}_{pooling}.csv", index=False)

        plot_loss_curve(loss_df, safe_model_name, pooling)

    # evaluate
    pred_v, pred_a, gold_v, gold_a = get_prd(model, dev_loader, device, type="dev")
    eval_score = evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)
    print(f"{safe_model_name} dev_eval: {eval_score}")

    df = pd.DataFrame({
        'Gold_Valence': gold_v,
        'Gold_Arousal': gold_a,
        'Pred_Valence': pred_v,
        'Pred_Arousal': pred_a
    })
    
    plot_scatter(df, eval_score, safe_model_name, pooling)

    # testing
    pred_dataset = VADataset(predict_df, tokenizer, use_aspect_pooling=pooling)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    pred_v, pred_a, = get_prd(model, pred_loader, device, type="pred")

    predict_df["Valence"] = pred_v
    predict_df["Arousal"] = pred_a

    # saving
    df_to_jsonl(predict_df, f"pred_{lang}_{domain}/{safe_model_name}_{pooling}.jsonl")