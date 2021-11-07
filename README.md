# OCR_Practice

SW 마에스트로 활동으로 작성된 코드입니다.
웹툰 영역 글자 인식에 맞는 학습 가능 모듈 및 경량화 모듈을 작성하였습니다.

# 학습

`train_model.py` 내부의 파라미터 조정 후

```bash
python train_model.py
```
를 실행하여 모델 학습을 실시합니다.

# 예측
`predict.py` 내부에 실제 예측을 위한 코드가 저장되어 있습니다. 이를 실행하여 예측이 가능합니다.

# Components
`Components` 에는 OCR에 필요한 각각의 모듈들이 저장되어 있습니다. 
- `crnn_model` : 실제 사용된 모델 파일입니다.
- `dataset`: 데이터 입력을 위한 dataset 클래스
- `tokenizer` : 예측 결과값을 문장으로 바꿔 주는 클래스
- `trainer` : 학습을 위한 코드들을 담고 있는 모듈

# 성능 평가
- 웹툰에서 추출된 웹툰 대사 이미지 100장으로 성능 테스트
- 파이썬 오픈소스 라이브러리인 `tesseract` 한국어 버전과 성능을 테스트함

## 결과
||자체 모델|tesseract|
|:---:|:---:|:---:|
|단어별 정확도(%)|26%|0%|
|글자별 정확도(%)|61%|21%|
