import os, sys
from PIL import Image
import numpy as np
from torch.functional import norm

from torch.utils.data import Dataset
from torchvision import transforms

import cv2


""" 하이퍼파라미터 설정 """
transform_resize_size = (32, 70) # (h, w) 크기
transform_interpolation = 0 # Image.NEAREST (0), Image.LANCZOS (1), Image.BILINEAR (2), Image.BICUBIC (3), Image.BOX (4) or Image.HAMMING (5)
mean, std = 0.5, 0.5 # Normalize 평균, 표준편차
""" 하이퍼파라미터 끝 """


""" transform 설정 """
train_transforms = transforms.Compose([
    transforms.Resize(size=transform_resize_size, interpolation=transform_interpolation),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transforms = transforms.Compose([
    transforms.Resize(size=transform_resize_size, interpolation=transform_interpolation),
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
    def __init__(self, data_dir, is_val = False):
        self.datas = os.listdir(data_dir)
        self.isVal = is_val
        self.data_dir = data_dir


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

        if self.isVal: # 
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            norm_gray = (((gray - gray.min())/(gray.max() - gray.min()))*255).astype(np.uint8)

            # threshold
            thresh = cv2.threshold(norm_gray, -1, 255, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)[1]

            # get the (largest) contour
            contours = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            # draw white filled contour on black background
            img2 = np.full_like(norm_gray, 255, np.uint8)
            _ = cv2.drawContours(img2, contours, -1, (0,0,0), cv2.FILLED)

            if img2[0,0] == 0:
                img2 = 255-img2

            img = Image.fromarray(img2)

        img = self.transforms(img) # 이미지 변형 적용

        x = np.array(img) # 변형된 이미지

        y = target.split('_')[0] # 파일에서 이미지 이름 추출
        

        return x, y

