import torch
"""
Trainer
생성된 모든 객체들을 입력으로 받아 모델을 학습시킵니다.
학습된 모델을 사용해서 OCR이미지로부터 문장을 추출할 수 있습니다.
"""

"""하이퍼파라미터 설정"""
trainer_params = {
    'epoch_num': 5, # 학습시킬 에폭 수
    'log_interval': 100, # 로그 찍어볼 미니배치 반복 수
    'is_string' : True, # 추론 결과물을 디코딩된 문장으로 반환할지, 토큰의 배열로 반환할지를 설정합니다.
    'seq_len': 10, # 모델 학습 기간
}

""" 하이퍼파라미터 설정 끝 """

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, criterion, optimizer, tokenizer):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters())

        self.tokenizer = tokenizer # 토크나이저 설정

        self.best_loss = 100

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

                ys = self.tokenizer.encode(ys) # 데이터 인코딩

                xs, ys = xs.to(self.device), ys.to(self.device)

                self.optimizer.zero_grad() # 그래디언트 파라미터를 0으로 설정

                output = self.model(xs, ys) # model.forward

                # loss 계산을 위한 차원 변경
                output = output.view(-1, self.model.output_dim)
                ys = ys.view(-1)

                loss = self.criterion(output, ys) # evaluate loss
                loss.backward() # backward pass
                self.optimizer.step()


                training_loss += loss.item()

                if idx % log_interval == log_interval-1:
                    print(f"Training Epoch [{idx + 1}/{epoch_num}] iter : {idx} loss: {(training_loss / log_interval):.3f}")
                    result = self.validation()

                    if result:
                        self.save_state('model/state', epoch, training_loss)
                        
                    training_loss = 0.0

                


    def validation(self):
        self.model.eval()
        print("========== Start Validation ... ========== ")
        validation_loss = 0.0
        for idx, data in enumerate(self.val_dataset):
            xs, original = data # 데이터 분리

            ys = self.tokenizer.encode(original) # 데이터 인코딩

            xs, new_ys = xs.to(self.device), ys.to(self.device)

            self.optimizer.zero_grad() # 그래디언트 파라미터를 0으로 설정

            output = self.model(xs, teacher_force=False) # model.forward

            new_output = output.view(-1, self.model.output_dim)
            new_ys = new_ys.view(-1)

            loss = self.criterion(new_output, new_ys) # evaluate loss
            validation_loss += loss.item()

            if idx > 2: break # 3번만 하고 브레이크
            
        print(f"validation loss : {(validation_loss/(idx+1)):0.5f}")
        output = output.permute(1,0,2).to('cpu') # 차원 변환 [batch, seq_len, class], cpu로 전송
        print("====== sentence ======")
        for i in range(10):
            print(f"Answer : {original[i]} / Predicted : {''.join(self.tokenizer.untokenize(output[i]))}")
        
        if validation_loss / (idx+1) < self.best_loss:
            return True # 저장하라고 알림
        else:
            return False

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


    def __repr__(self):
        """
        Trainer 를 호출했을때 실행되는 메소드입니다.
        """

        return f"model: {self.model}\ncriterion: {self.criterion}\noptimizer : {self.optimizer}"

    def save_model(self, save_path):
        print("Save Model To ", save_path)
        torch.save(self.model.state_dict(), save_path)

    def save_state(self, save_path, epoch, loss):
        print("Save state to ", save_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            })

