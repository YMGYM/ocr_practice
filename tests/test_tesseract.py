from PIL import Image
import cv2
from pytesseract import *
import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm


params = {
    'result_path': './result_test_tesseract.csv'
}


BASE_DIR = "/home/jun/myWorks/soma/font/dataset/realTesting/"
img_list = os.listdir(BASE_DIR)

result = []

for img in tqdm(img_list):
    op_img = Image.open(BASE_DIR + img)
    text = pytesseract.image_to_string(op_img,lang='kor') #한글은 'kor'

    result.append({
        'predict': text,
        'answer': img.split('_')[0]
    })
    
df = pd.DataFrame(result)
df.to_csv(params['result_path']) # 데이터 저장
print(f"Result Data was saved at {params['result_path']}")

# 단어 별로 정답률을 체크합니다.
accuracy = (df['predict'] == df['answer']).values.sum() / len(df)
print(f"accuracy(word):{accuracy * 100}%")

# 글자별 정확도 체크
totalCounter = Counter() # 모든 글자를 저장하는 카운터
trueCounter = Counter() # 맞춘 글자를 저장하는 카운터
falseCounter = Counter() # 오답 글자를 저장하는 카운터

for row in df.iterrows():
    for charIdx, char in enumerate(row[1]['answer']):
        totalCounter[char] += 1
        if charIdx < len(row[1]['predict']):
            if char in row[1]['predict']: # 정답인경우
                trueCounter[char] += 1
            else:
                falseCounter[char] += 1
        else:
            falseCounter[char] += 1

print(f"accuracy(character) :{sum(trueCounter.values())/sum(totalCounter.values()) * 100}%")


