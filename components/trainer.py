"""
Trainer
생성된 모든 객체들을 입력으로 받아 모델을 학습시킵니다.
학습된 모델을 사용해서 OCR이미지로부터 문장을 추출할 수 있습니다.
"""

"""하이퍼파라미터 설정"""
trainer_params = {
    'epoch_num': 5, # 학습시킬 에폭 수
    'log_interval': 1000, # 로그 찍어볼 미니배치 반복 수
    'is_string' : True, # 추론 결과물을 디코딩된 문장으로 반환할지, 토큰의 배열로 반환할지를 설정합니다.
}

""" 하이퍼파라미터 설정 끝 """

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, criterion, optimizer):

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimzier = optimizer(self.model.parameters())

        self.tokenizer = self.dataset.tokenizer # 토크나이저 설정


    def train_model(self, epoch_num=trainer_params['epoch_num'], log_interval = trainer_params['log_interval']):

        assert self.model is not None, "Model is None"
        assert self.train_dataset is not None, "train_dataset is None"
        assert self.val_dataset is not None, "val_dataset is None"
        assert self.criterion is not None, "criterion is None"
        assert self.optimizer is not None, "optimizer is None"
        assert self.tokenizer is not None, "tokenizer is None"

        print("========== Start Training Model.... ==========")
    
        for epoch in range(epoch_num): # epoch 만큼 확인합니다.
            training_loss = 0.0

            for idx, data in enumerate(self.train_dataset):
                self.model.train()

                xs, ys = data # 데이터 분리

                self.optimzer.zero_grad() # 그래디언트 파라미터를 0으로 설정

                output = self.model(xs) # model.forward
                loss = self.criterion(output, ys) # evaluate loss
                loss.backward() # backward pass
                self.optimizer.step()


                training_loss += loss.item()

                if idx % log_interval == 0:
                    print(f"Training Epoch [{idx + 1}/{epoch_num}] loss: {(training_loss / log_interval):.3f}")
                    self.validation()
                    


    def validation(self):
        self.model.eval()
        print("========== Start Validation ... ========== ")
        validation_loss = 0.0
        for idx, data in enumerate(self.val_dataset):
            xs, ys = data # 데이터 분리

            self.optimzer.zero_grad() # 그래디언트 파라미터를 0으로 설정

            output = self.model(xs) # model.forward
            loss = self.criterion(output, ys) # evaluate loss
            self.optimizer.step()


            validation_loss += loss.item()

        print(f"validation loss : {(validation_loss/idx):0.5f}")
        for i in range(10):
            print("====== sentence ======")
            print(f"Answer : {ys[i]} / Predicted : {output[i]}")


    def predict(self, data, is_string = True):
        """
        data를 입력으로 받아 결과값을 예측합니다.
        data는 numpy 배일로 가정합니다.
        is_string 이 True 이면 디코딩된 문장으로 반환, False면 토크나이즈된 배열로 반환합니다.
        """
        output = self.model(data)

        if is_string:
            output = self.tokenizer.decode(output)

        return output


    def set_params(self, **params): # 비어있는 파라미터를 세팅합니다.
        if params.get('model'): self.model = params['model']
        if params.get('train_dataset'): self.train_dataset = params['train_dataset']
        if params.get('val_dataset'): self.val_dataset = params['val_dataset']
        if params.get('criterion'): self.criterion = params['criterion']
        if params.get('optimizer'): self.optimizer = params['optimizer']
        if params.get('tokenizer'): self.tokenizer = params['tokenizer']



            