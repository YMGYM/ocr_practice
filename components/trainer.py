from .tokenizer import Tokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
"""
Trainer
생성된 모든 객체들을 입력으로 받아 모델을 학습시킵니다.
학습된 모델을 사용해서 OCR이미지로부터 문장을 추출할 수 있습니다.
"""

"""하이퍼파라미터 설정"""
trainer_params = {
    'epoch_num': 100, # 학습시킬 에폭 수
    'log_interval': 100, # 로그 찍어볼 미니배치 반복 수
    'is_string' : True, # 추론 결과물을 디코딩된 문장으로 반환할지, 토큰의 배열로 반환할지를 설정합니다.
    'seq_len': 10, # 모델의 sequence 길이
    'sent_interval': 10, # training 시에 문장 출력 에폭
    'optimizer_lr': 0.0005, # optimizer learning rate
    'save_path' : './model/state.pth', # 모델 저장 경로 -> 차차 오버라이딩됨
    'is_save' : True,
}

""" 하이퍼파라미터 설정 끝 """

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, criterion, optimizer, tokenizer):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=trainer_params['optimizer_lr'])

        self.tokenizer = tokenizer

        self.best_loss = 100 # 임의의 큰 값으로 입력

        # uniform 가중치 입력
        self.init_parameters()

        # tensorboard 객체 생성
        if trainer_params['is_save']:
            self.writer = SummaryWriter()

    def train_model(self, epoch_num=trainer_params['epoch_num'], log_interval = trainer_params['log_interval'], sent_interval = trainer_params['sent_interval'], save_path= trainer_params['save_path'], load_model=True):

        assert self.model is not None, "Model is None"
        assert self.train_dataset is not None, "train_dataset is None"
        assert self.val_dataset is not None, "val_dataset is None"
        assert self.criterion is not None, "criterion is None"
        assert self.optimizer is not None, "optimizer is None"
        assert self.tokenizer is not None, "tokenizer is None"

        # 학습된 모델을 불러옵니다.
        if load_model:
            epoch = self.load_model(save_path)
        else:
            epoch = 0

        print("========== Start Training Model.... ==========")
    
        for epoch in range(epoch_num): # epoch 만큼 확인합니다.
            training_loss = 0.0

            for idx, data in enumerate(self.train_dataset):
                self.model.train()

                xs, original = data # 데이터 분리

                ys, y_len = self.tokenizer.encode(original) # 데이터 인코딩
                xs = xs.to(self.device)

                self.optimizer.zero_grad() # 그래디언트 파라미터를 0으로 설정

                output = self.model(xs) # model.forward output : (10, batch, num_words)
                # print(output)
                seq_shape = torch.full(size=(output.shape[1],), fill_value=trainer_params['seq_len'], dtype=torch.int32)

                loss = self.criterion(output, ys, seq_shape, y_len) # evaluate loss
        
                loss.backward() # backward pass
                self.optimizer.step()

                training_loss += loss.item()

                if idx % log_interval == log_interval-1:
                    print(f"Training Epoch [{epoch + 1}/{epoch_num}] iter : {idx + 1} loss: {(training_loss / (idx+1)):.3f}")
                    result = self.validation(idx*(epoch+1))

                    if result:
                        self.save_state(save_path, epoch, training_loss)

                if idx % sent_interval == 0:
                    print(f"Training Epoch [{epoch + 1}/{epoch_num}] iter : {idx + 1} loss: {(training_loss / (idx+1)):.3f}")
                    print("====== Train Sentence ======")
                    output = output.permute(1,0,2) # [batch, sequence, output_len]
                    for i in range(2):
                        print(f"Answer : {original[i]} / Predicted : {''.join(self.tokenizer.untokenize(output[i]))}")
                        

                    # Tensorboard 에 training_loss 기록
                    if trainer_params['is_save']:
                        self.writer.add_scalar('Loss/Train', (training_loss / (idx+1)), idx*(epoch+1)) # loss 기록



    def validation(self, iter_count):
        self.model.eval()
        print("========== Start Validation ... ========== ")
        validation_loss = 0.0
        for idx, data in enumerate(self.val_dataset):

            if idx > 2: break # 3번만 하고 브레이크

            xs, original = data # 데이터 분리

            ys, y_len = self.tokenizer.encode(original) # 데이터 인코딩

            xs = xs.to(self.device)
            
            self.optimizer.zero_grad() # 그래디언트 파라미터를 0으로 설정

            output = self.model(xs) # model.forward

            seq_shape = torch.full(size=(output.shape[1],), fill_value=output.shape[0])

            loss = self.criterion(output, ys, seq_shape, y_len) # evaluate loss
            validation_loss += loss.item()

            
            
        print(f"validation loss : {(validation_loss/(idx+1)):0.5f}")
        output = output.permute(1,0,2).to('cpu') # 차원 변환 [batch, seq_len, class], cpu로 전송


        print("====== sentence ======")
        for i in range(5):
            print(f"Answer : {original[i]} / Predicted : {''.join(self.tokenizer.untokenize(output[i]))}")
            

        # Tensorboard 에 validation_loss 기록
        if trainer_params['is_save']:
            self.writer.add_scalar('Loss/Validation', (validation_loss/(idx+1)), iter_count) # loss 기록


        print("========== Finish Validation ========== ")


        if validation_loss / (idx+1) < self.best_loss:
            self.best_loss = validation_loss / (idx+1)
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

    def save_state(self, save_path, epoch, loss): # 상태를 통으로 저장합니다.
        print("Save state to ", save_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, save_path)


    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    def init_parameters(self):
        for name, param in self.model.named_parameters():
            torch.nn.init.uniform_(param.data, -0.08, 0.08)


    # 모델과 가중치를 load 합니다.
    # is_state : 모델 저장을 save_state로 했는지(True), save_model로 했는지 (False)
    def load_model(self, load_path, is_state=True):
        print("Load model from ", load_path)

        if is_state:
            checkpoint = torch.load(load_path)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            self.best_loss = checkpoint['loss']
            return epoch
            
        else: # 모델만 저장함
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint)
            return None
        

    