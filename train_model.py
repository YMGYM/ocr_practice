from components.dataset import OcrDataset # OCR용 데이터셋
from torch.utils.data import DataLoader # 멀티배치 학습을 위한 데이터로더
from components.tokenizer import Tokenizer
from components.trainer import Trainer
from components.model import Model

import torch.optim as optim
import torch.nn as nn

"""
모델을 학습합니다.
"""

params = {
    'epochs': 100,
    'log_interval': 30, # validation 확인 정도
    'sent_interval': 10, # 학습 중 모델 인식 결과물을 찍어 보는 주기
    'batch_size': 2048,
    'load_model': True, # 모델을 이어서 학습할 것인지
    'save_path': './model/state_long_cnn.pth' # 모델 저장 위치
}

# 데이터셋 경로 설정 =============
base_dir = '../soma/font/dataset/'
train_dir = base_dir + 'train/'
val_dir = base_dir + 'val/'
# ===========================


# 데이터셋 생성 ================
train_ocr = OcrDataset(train_dir, is_val=False)
train_dataset = DataLoader(train_ocr, batch_size=params['batch_size'], shuffle=True)

val_ocr = OcrDataset(val_dir, is_val=True)
val_dataset = DataLoader(val_ocr, batch_size=params['batch_size'], shuffle=True)
# ===========================

# 학습에 필요한 파일 생성 ========
tokenizer = Tokenizer(seq_len=10, one_hot=False)
model = Model(tokenizer)
PAD_IDX = tokenizer.word2id['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
optimizer = optim.Adam # 클래스 정보만 넘겨줌
trainer = Trainer(model, train_dataset, val_dataset, criterion, optimizer, tokenizer )
# ==========================

if __name__ == "__main__":
    

    print(f"{trainer.count_parameters()} trainable parameters")
    trainer.train_model(epoch_num=params['epochs'], log_interval=params['log_interval'], sent_interval=params['sent_interval'], save_path=params['save_path'], load_model=params['load_model'])