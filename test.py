from components.dataset import OcrDataset # OCR용 데이터셋
from torch.utils.data import DataLoader # 멀티배치 학습을 위한 데이터로더
from components.tokenizer import Tokenizer
from components.crnn_model import CRNN

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from tqdm import tqdm
"""
학습된 모델을 불러와서 평가합니다.
"""

params = {
    'batch_size': 512,
    'model_path': './model/state_crnn_big.pth', # 모델 저장 위치
    'result_path': './result.csv'
}

# 데이터셋 경로 설정 =============
base_dir = '../soma/font/dataset/'
test_dir = base_dir + 'test_dict/'
# ===========================

# 데이터셋 생성 ================
test_ocr = OcrDataset(test_dir, is_val=False)
test_dataset = DataLoader(test_ocr, batch_size=params['batch_size'], shuffle=True)
# ===========================

# 학습에 필요한 파일 생성 ========
tokenizer = Tokenizer(seq_len=10, one_hot=False)
model = CRNN()
criterion = nn.CTCLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ==========================

if __name__ == "__main__":

    print("========== Start Testing ... ========== ")
    print(f"tokenizer word size : {len(tokenizer.word2id)}")

    # 학습된 model 불러오기 기능
    print("Load model from ", params['model_path'])

    result = []

    checkpoint = torch.load(params['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.eval()
    torch.no_grad()

    test_loss = 0.0
    for idx, data in tqdm(enumerate(test_dataset)):

        xs, original = data # 데이터 분리

        ys, y_len = tokenizer.encode(original) # 데이터 인코딩

        xs = xs.to(device)

        output = model(xs) # model.forward

        seq_shape = torch.full(size=(output.shape[1],), fill_value=output.shape[0])

        loss = criterion(output, ys, seq_shape, y_len) # evaluate loss

        test_loss += loss.item()
        
        output = output.permute(1,0,2) # (batch, seq, word_len)
        for i, sent in enumerate(output):
            result.append({'predict' : tokenizer.untokenize(sent), 'answer': original[i]})

            
    print(f"Test loss : {(test_loss / len(test_dataset) ):0.5f}")
    df = pd.DataFrame(result)
    df.to_csv(params['result_path']) # 데이터 저장
    print(f"Result Data was saved at {params['result_path']}")

    accuracy = (df['predict'] == df['answer']).values.sum() / len(df)
    print(f"accuarcy :{accuracy * 100}%")


    

