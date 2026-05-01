import os
import json
import time
from tqdm import tqdm
from openai import AzureOpenAI
import numpy as np
import json
import time
import pandas as pd

from tqdm import tqdm
from openai import AzureOpenAI
from utils import evaluate_predictions_task1, plot_scatter, load_jsonl

#task config
subtask = "subtask_1"#don't change
task = "task1"#don't change
lang = "eng" #chang the language you want to test
domain = "environmental_protection" #change what domain you want to test
# TEST_TYPE = "zero-shot"
# TEST_TYPE = "1-shot"
# TEST_TYPE = "3-shot"
TEST_TYPE = "5-shot"
SYSTEM_PROMPT = open("system_prompt/base.txt", encoding="utf-8").read() + "\n\n" + open(f"system_prompt/{TEST_TYPE}.txt", encoding="utf-8").read()

TEST_DATA_PATH = f"./task-dataset/track_b/{subtask}/{lang}/{lang}_{domain}_train_task1.jsonl"  # 原项目测试集路径

AZURE_ENDPOINT = "https://hkust.azure-api.net/"
AZURE_API_KEY = "447dcc80393b4d1c914666138495e0d2"
AZURE_API_VERSION = "2025-02-01-preview"
DEPLOYMENT_NAME = "gpt-4o-mini"  # 你的部署名

TEMPERATURE = 0.0
MAX_TOKENS = 128
TOP_P = 1.0

safe_model_name = f"{DEPLOYMENT_NAME}_{TEST_TYPE}"

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

def predict_stance(text: str, aspects: list):
    """调用 Azure OpenAI 预测单个样本的维度立场"""
    # 构造用户输入
    user_content = f"Text: {text}\nAspects: {', '.join(aspects)}"
    
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_content}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        res = json.loads(response.choices[0].message.content.strip())
        pred_v_list = []
        pred_a_list = []
        for item in res["Aspect_VA"]:
            pred_v = float(item["VA"].split("#")[0])
            pred_a = float(item["VA"].split("#")[1])
            pred_v_list.append(pred_v)
            pred_a_list.append(pred_a)
    except Exception as e:
        print(f"Error: {e}")
        pred_v_list = [0] * len(aspects)
        pred_a_list = [0] * len(aspects)

    return pred_v_list, pred_a_list

def get_prd_llm(test_data):

    all_pred_v = []
    all_pred_a = []
    all_gold_v = []
    all_gold_a = []

    # count = 0
    for sample in tqdm(test_data):
        text = sample["Text"]
        aspects_list = []
        gold_v_list = []
        gold_a_list = []

        for item in sample["Aspect_VA"]:
            aspect = item["Aspect"]
            va_str = item["VA"]
            
            v_str, a_str = va_str.split("#")
            gold_v = float(v_str)
            gold_a = float(a_str)
            
            aspects_list.append(aspect)  # 顺序保存所有方面
            gold_v_list.append(gold_v)   # 顺序保存所有V
            gold_a_list.append(gold_a)   # 顺序保存所有A
            
        pred_v, pred_a = predict_stance(text, aspects_list)
        
        if pred_v[0] == 0: continue
        if len(pred_v) != len(gold_v_list): 
            print("Error: 数目不同")
            continue
        
        all_pred_v.extend(pred_v)    # append → extend
        all_pred_a.extend(pred_a)
        all_gold_v.extend(gold_v_list)
        all_gold_a.extend(gold_a_list)

        time.sleep(0.2) 
        
        # 测试用中断控制
        # count += 1
        # if count == 30: break

    # 转成和原模型一样的 numpy 数组
    pred_v = np.array(all_pred_v)
    pred_a = np.array(all_pred_a)
    gold_v = np.array(all_gold_v)
    gold_a = np.array(all_gold_a)

    return pred_v, pred_a, gold_v, gold_a

if __name__ == "__main__":
    # 1. 加载测试数据
    test_data = load_jsonl(TEST_DATA_PATH)

    pred_v, pred_a, gold_v, gold_a = get_prd_llm(test_data)
    eval_score = evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)
    print(f"{safe_model_name} dev_eval: {eval_score}")

    df = pd.DataFrame({
        'Gold_Valence': gold_v,
        'Gold_Arousal': gold_a,
        'Pred_Valence': pred_v,
        'Pred_Arousal': pred_a
    })
    
    plot_scatter(df, eval_score, safe_model_name)
