from components.dataset import OcrDataset # OCR용 데이터셋
from torch.utils.data import DataLoader # 멀티배치 학습을 위한 데이터로더
from components.tokenizer import Tokenizer
from components.trainer import Trainer
from components.crnn_model import CRNN

import torch.optim as optim
import torch.nn as nn

"""
모델을 학습합니다.
"""

params = {
    'epochs': 60,
    'log_interval': 40, # validation 확인 정도
    'sent_interval': 10, # 학습 중 모델 인식 결과물을 찍어 보는 주기
    'batch_size': 512,
    'load_model': False, # 모델을 이어서 학습할 것인지
    'save_path': './model/model_drop.pth' # 모델 저장 위치
}

# 데이터셋 경로 설정 =============
base_dir = '../soma/font/dataset/'
train_dir = base_dir + 'train_dict_small/'
val_dir = base_dir + 'val_dict_small/'
# ===========================


# 데이터셋 생성 ================
train_ocr = OcrDataset(train_dir, is_val=False)
train_dataset = DataLoader(train_ocr, batch_size=params['batch_size'], shuffle=True)

val_ocr = OcrDataset(val_dir, is_val=True)
val_dataset = DataLoader(val_ocr, batch_size=params['batch_size'], shuffle=True)
# ===========================

# 학습에 필요한 파일 생성 ========
tokenizer = Tokenizer(seq_len=10, one_hot=False)
model = CRNN()
criterion = nn.CTCLoss()
optimizer = optim.Adam # 클래스 정보만 넘겨줌
trainer = Trainer(model, train_dataset, val_dataset, criterion, optimizer, tokenizer )
# ==========================

if __name__ == "__main__":
    

    print(f"{trainer.count_parameters()} trainable parameters")
    print(f"tokenizer word size : {len(tokenizer.word2id)}")
    trainer.train_model(epoch_num=params['epochs'], log_interval=params['log_interval'], sent_interval=params['sent_interval'], save_path=params['save_path'], load_model=params['load_model'])