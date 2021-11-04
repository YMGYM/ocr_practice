import sys
sys.path.append('../minjun_ocr')
from collections import Counter

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
    'model_path': './model/model_font_fit.pth', # 모델 저장 위치
    'result_path': './result_test_small.csv'
}

crnn_params = {
    'conv1_out': 128,
    'conv1_kernel_size' : 5,
    'conv2_out': 128,
    'conv2_kernel_size': 3,
    'dropout_ratio': 0.4694,
    'rnn_hidden_size': 1024,
    'rnn_bidirectional': True, # bidirectional LSTM 사용 유무
    'rnn_num_layers': 3, # RNN 계층을 몇개 쌓을 것인지
    'num_words': 1482, #  tokenizer의 word2id의 길이와 동일해야 함
}


# 데이터셋 경로 설정 =============
base_dir = '../soma/font/dataset/'
test_dir = base_dir + 'realTesting/'
# ===========================

# 데이터셋 생성 ================
test_ocr = OcrDataset(test_dir, is_val=True)
test_dataset = DataLoader(test_ocr, batch_size=params['batch_size'], shuffle=False)
# ===========================

# 학습에 필요한 파일 생성 ========
tokenizer = Tokenizer(seq_len=10, one_hot=False)
model = CRNN(crnn_params)
criterion = nn.CTCLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ==========================

if __name__ == "__main__":
    totalCounter = Counter() # 모든 글자를 저장하는 카운터
    trueCounter = Counter() # 맞춘 글자를 저장하는 카운터
    falseCounter = Counter() # 오답 글자를 저장하는 카운터

    print("========== Start Testing ... ========== ")
    print(f"tokenizer word size : {len(tokenizer.word2id)}")

    # 학습된 model 불러오기 기능
    print("Load model from ", params['model_path'])

    result = []

    checkpoint = torch.load(params['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
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
            result.append({'predict' : tokenizer.untokenize(sent), 'answer': tokenizer.make_answer_word(original[i]), 'original': original[i],})

            
    print(f"Test loss : {(test_loss / len(test_dataset) ):0.5f}")
    df = pd.DataFrame(result)
    df.to_csv(params['result_path']) # 데이터 저장
    print(f"Result Data was saved at {params['result_path']}")

    # 단어 별로 정답률을 체크합니다.
    accuracy = (df['predict'] == df['original']).values.sum() / len(df)
    print(f"accuracy(word):{accuracy * 100}%")

    for row in df.iterrows():
        for charIdx, char in enumerate(row[1]['original']):
            if char in tokenizer.target_text:
                totalCounter[char] += 1

                if char in row[1]['predict']: # 정답인경우
                    trueCounter[char] += 1
                else:
                    falseCounter[char] += 1
            else:
                totalCounter['<OOV>'] += 1
                falseCounter['<OOV>'] += 1
    print(f"accuracy(character) :{sum(trueCounter.values())/sum(totalCounter.values()) * 100}%")
    

