from .components.tokenizer import Tokenizer
from .components.crnn_model import CRNN
from torchvision import transforms
import torch
from PIL import Image
import numpy as np

"""
학습된 모델을 불러와서 평가합니다.
"""

params = {
    'batch_size': 512,
    'model_path': '../src/model/new_ocr/model/state_crnn_big.pth', # 모델 저장 위치
    'transform_resize_size' : (32, 100), # (h, w) 크기
    'transform_interpolation' : 0,
    'mean' : 0.5,
    'std' :  0.5, # Normalize 평균, 표준편차
}



transforms = transforms.Compose([
    transforms.Resize(size=params['transform_resize_size'], interpolation=params['transform_interpolation']),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(params['mean'], params['std']),
])


# 학습에 필요한 파일 생성 ========
tokenizer = Tokenizer(seq_len=10, one_hot=False)
model = CRNN()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load(params['model_path'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
# ==========================

def crnn_predict(bbox):
    """
    주어진 bbox 배열을 순회하면서 텍스트를 인식합니다.
    text_image가 np.array의 단일 이미지라고 가정하고 실행합니다.
    text_image는 bbox 단위로 crop된 글자 이미지 배열이 들어갑니다. (단일 컷의 모든 bounding box입니다.)

    """
    model.eval()
    torch.no_grad()

    bboxes = []

    for box in bbox:
        box = Image.fromarray(box)
        box = transforms(box)
        box = torch.tensor(np.expand_dims(box, axis=0)).to(device)

        output = model(box) # model.forward
        output = output.permute(1,0,2)
        result = tokenizer.decode(output)

        bboxes.append(*result)

    return bboxes