import json
from typing import List, Dict
import pandas as pd
import re
import torch
import numpy as np
from scipy.stats import pearsonr
import math
import matplotlib.pyplot as plt
import seaborn as sns

def load_jsonl(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def jsonl_to_df(data):
    if 'Quadruplet' in data[0]:
        df = pd.json_normalize(data, 'Quadruplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Category', 'Opinion'])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Triplet' in data[0]:
        df = pd.json_normalize(data, 'Triplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Opinion'])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Aspect_VA' in data[0]:
        df = pd.json_normalize(data, 'Aspect_VA', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})  # rename to Aspect
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Aspect' in data[0]:
        df = pd.json_normalize(data, 'Aspect', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})  # rename to Aspect
        df['Valence'] = 0  # default value
        df['Arousal'] = 0  # default value

    else:
        raise ValueError("Invalid format: must include 'Quadruplet' or 'Triplet' or 'Aspect'")

    return df

def extract_num(s):
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1

def df_to_jsonl(df, out_path):
    df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
    grouped = df_sorted.groupby("ID", sort=False)

    with open(out_path, "w", encoding="utf-8") as f:
        for gid, gdf in grouped:
            record = {
                "ID": gid,
                "Aspect_VA": []
            }
            for _, row in gdf.iterrows():
                record["Aspect_VA"].append({
                    "Aspect": row["Aspect"],
                    "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
                })
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def get_prd(model, dataloder, device, type ="dev"):
    if type == "dev":
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dataloder:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].cpu().numpy()
                outputs = model(input_ids, attention_mask).cpu().numpy()
                all_preds.append(outputs)
                all_labels.append(labels)
        preds = np.vstack(all_preds)
        lables = np.vstack(all_labels)

        pred_v = preds[:,0]
        pred_a = preds[:,1]

        gold_v = lables[:,0]
        gold_a = lables[:,1]

        return pred_v, pred_a, gold_v, gold_a

    elif type == "pred":
        all_preds = []
        with torch.no_grad():
            for batch in dataloder:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask).cpu().numpy()
                all_preds.append(outputs)
        preds = np.vstack(all_preds)

        pred_v = preds[:, 0]
        pred_a = preds[:, 1]

        return pred_v, pred_a

def evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v, is_norm = False):
    if not (all(1 <= x <= 9 for x in pred_v) and all(1 <= x <= 9 for x in pred_a)):
        print(f"Warning: Some predicted values are out of the numerical range.")
    pcc_v = pearsonr(pred_v,gold_v)[0]
    pcc_a = pearsonr(pred_a,gold_a)[0]

    gold_va = list(gold_v) + list(gold_a)
    pred_va = list(pred_v) + list(pred_a)
    def rmse_norm(gold_va, pred_va, is_normalization = True):
        result = [(a - b)**2 for a, b in zip(gold_va, pred_va)]
        if is_normalization:
            return math.sqrt(sum(result)/len(gold_v))/math.sqrt(128)
        return math.sqrt(sum(result)/len(gold_v))
    rmse_va = rmse_norm(gold_va, pred_va, is_norm)
    return {
        'PCC_V': pcc_v,
        'PCC_A': pcc_a,
        'RMSE_VA': rmse_va,
    }
    
def plot_loss_curve(loss_df, safe_model_name, pooling = False):
    if loss_df is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_df['Epoch'], loss_df['Train_Loss'], label='Train Loss', marker='o')
        plt.plot(loss_df['Epoch'], loss_df['Val_Loss'], label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'Training and Validation Loss Curve\n{safe_model_name}_{pooling}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'loss_curve/{safe_model_name}_{pooling}.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_scatter(df, eval_score, safe_model_name, pooling = False):
    plt.figure(figsize=(11, 9))

    sns.scatterplot(
        data=df, 
        x='Gold_Valence', 
        y='Gold_Arousal', 
        label='Gold Labels', 
        alpha=0.75, 
        s=70, 
        color='#1f77b4'
    )

    sns.scatterplot(
        data=df, 
        x='Pred_Valence', 
        y='Pred_Arousal', 
        label='Model Predictions (Red)', 
        alpha=0.75, 
        s=70, 
        color='#d62728'
    )

    plt.xlabel('Valence (1 = Very Negative  →  9 = Very Positive)', fontsize=13)
    plt.ylabel('Arousal (1 = Calm  →  9 = Very Intense)', fontsize=13)
    plt.title(f'Valence-Arousal: Gold vs Predictions\n{safe_model_name}_{pooling} | RMSE_VA = {eval_score["RMSE_VA"]:.3f}', 
            fontsize=15, pad=20)

    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 9)
    plt.ylim(1, 9)

    plt.tight_layout()
    plt.savefig(f'./pred_vs_gold_scatter/{safe_model_name}_{pooling}.png', dpi=400, bbox_inches='tight')
    plt.show()
