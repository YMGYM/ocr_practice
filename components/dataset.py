import os, sys
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from .tokenizer import Tokenizer


""" 하이퍼파라미터 설정 """
transform_resize_size = (32, 100) # (h, w) 크기
transform_resize_interpolation = 0 # Image.NEAREST (0), Image.LANCZOS (1), Image.BILINEAR (2), Image.BICUBIC (3), Image.BOX (4) or Image.HAMMING (5)
mean, std = 0.5, 0.5 # Normalize 평균, 표준편차
degree = 45 # 회전 각도

""" 하이퍼파라미터 끝 """


""" transform 설정 """
train_transforms = transforms.Compose([
    transforms.Resize(size=transform_resize_size, interpolation=transform_resize_interpolation),
    transforms.RandomAffine(degree),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transforms = transforms.Compose([
    transforms.Resize(size=transform_resize_size, interpolation=transform_resize_interpolation),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
""" transform 끝 """


class OcrDataset(Dataset): 
    """
    dataset 클래스입니다.
    생성자 파라미터
    data_dir : 타겟 디렉토리 폴더 주소
    is_val : validation 인지 여부 (transforms에 차이가 있습니다)

    결과값:
    x : 이미지 np.array()
    y : 정답 string
    """
    def __init__(self, data_dir, is_val = False, seq_len = 10):
        self.datas = os.listdir(data_dir)
        self.isVal = is_val
        self.data_dir = data_dir
        self.tokenizer = Tokenizer(seq_len = seq_len)

        # train 인지 val인지에 따라 변형을 다르게 적용함
        if is_val:
            self.transforms = val_transforms
        else:
            self.transforms = train_transforms


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        target = self.datas[idx]
        img = Image.open(self.data_dir + '/' + target)
        img = self.transforms(img) # 이미지 변형 적용

        x = np.array(img) # 변형된 이미지

        imgName = target.split('_')[0] # 파일에서 이미지 이름 추출
        y = self.tokenizer(imgName) # y 토크나이징

        return x, y

