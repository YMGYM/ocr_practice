# OCR_Practice

SW 마에스트로 활동으로 작성된 코드입니다.

pytorch를 활용하여 OCR 인공지능을 제작하는 것이 목표입니다.

# Run

`train_model.py` 내부의 파라미터 조정 후

```bash
python train_model.py
```

를 실행하여 모델 학습

클래스별로 confusion matrix

# 성능 평가
- 웹툰에서 추출된 웹툰 대사 이미지 100장으로 성능 테스트
- 파이썬 오픈소스 라이브러리인 `tesseract` 한국어 버전과 성능을 테스트함

## 결과
||자체 모델|tesseract|
|:---:|:---:|:---:|
|단어별 정확도(%)|26%|0%|
|글자별 정확도(%)|61%|21%|
