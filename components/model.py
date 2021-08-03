import torch
from torch import nn
import torch.nn.functional as F

"""
Encoder 와 Decoder로 구성된 img2seq 모델.
인코더는 cnn을 임베딩해서 하나의 linear 텐서를 반환
디코더는 인코더 값을 적용하여 sequence 를 생성
"""


""" 하이퍼파라미터 설정 """
encoder_params = {
    'conv1_out': 64,
    'conv1_kernel_size' : 4,
    'conv2_out': 16,
    'conv2_kernel_size': 4,
    'fc_out': 128,
    'dropout_ratio': 0.5
}

decoder_params = {
    'num_words': 1017, # dataset의 word2id의 길이와 동일해야 함
    'embedding_dim': 256,
    'rnn_hidden_size': 128, # encoder의 fc_out과 동일해야 함
    'dropout_ratio': 0.5
}

teacher_force = False # 모델 학습 시 정답을 강제로 입력하는지 여부
""" 하이퍼파라미터 끝"""

class Encoder(nn.Module):
    def __init__(self, encoder_params):
        super(Encoder, self).__init__()
        self.params = encoder_params
        self.conv1 = nn.Conv2d(1, encoder_params['conv1_out'], encoder_params['conv1_kernel_size']) # in_channels, out_channels, kernel_size, stride
        # output = (x + 2*paddint - dilation * (kernel_size - 1)-1) / stride
        # output = (64, 28, 96)
        self.pool = nn.MaxPool2d(2, 2) # output = (64, 14, 48)
        self.drop1 = nn.Dropout(encoder_params['dropout_ratio'])
        self.conv2 = nn.Conv2d(encoder_params['conv1_out'], encoder_params['conv2_out'], encoder_params['conv2_kernel_size']) # output = (16, 10, 44) => (16, 5, 22)
        self.drop2 = nn.Dropout(encoder_params['dropout_ratio'])

        self.fc = nn.Linear(16*5*22, encoder_params['fc_out'])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)

        x = torch.flatten(x, 1) # 배치를 제외하고 flatten() 하기 때문에 1
        x = F.relu(self.fc(x))

        return x


class Decoder(nn.Module):
    def __init__(self, decoder_params):
        super(Decoder, self).__init__()

        self.params = decoder_params
        self.embedding = nn.Embedding(num_embeddings= decoder_params['num_words'], embedding_dim= decoder_params['embedding_dim'])
        self.dropout = nn.Dropout(decoder_params['dropout_ratio'])
        self.rnn = nn.GRU(input_size= decoder_params['embedding_dim'], hidden_size = decoder_params['rnn_hidden_size'])

        self.fc_out = nn.Linear(128, decoder_params['num_words'])


    def forward(self, x, hidden):
        # input : [batch,]
        x = self.dropout(self.embedding(x)).unsqueeze(0) # output : [1, batch, embedding_dim]

        rnn_out, out_hidden = self.rnn(x, hidden)

        # rnn_out : [1, batch, h_out]
        # out_hidden : [1, batch, h_out]

        output = self.fc_out(rnn_out) # out : [batch, num_words]

        return output, out_hidden

class Model(nn.Module):
    def __init__(self, tokenizer, seq_len=10):
        """
        파라미터 정리
        device : 트레이너에서 넘어온 cuda, cpu 정보
        tokenizer : Tokenizer 모듈. 트레이너에서 넘어옴
        seq_len : 문장에서 쓰는 글자의 개수. tokenizer의 seq_len과 일치해야함. 기본값 10
        """
        super(Model, self).__init__()
        self.encoder = Encoder(encoder_params)
        self.decoder = Decoder(decoder_params)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.output_dim = decoder_params['num_words'] # loss 계산을 위해 flatten 하는 과정에서 사용
        """
        ASSERT
        인코더의 out_state가 디코더의 hidden_state로 들어가기 때문에 크기가 일치해야 합니다.
        임베딩 벡터의 길이와 토크나이저의 단어 길이가 일치해야 합니다.
        토크나이저의 seq_len과 모델의 seq_len이 일치해야 합니다.
        """

        assert encoder_params['fc_out'] == decoder_params['rnn_hidden_size'], "encoder['fc_out'] and decoder['rnn_hidden_size'] must be same!!"
        assert decoder_params['num_words'] == len(self.tokenizer.word2id) + 1, "'embedding_dim' dosen't have full words"
        assert self.tokenizer.seq_len == self.seq_len, "seq_len dosen't match!"

    def forward(self, x, y = None, teacher_force = teacher_force):
        """
        x : numpy 이미지
        y : 정답 one-hot 인코딩
        teacher_force : 정답 지도 학습 (다음 input으로 예측값이 아닌 정답 정보가 들어갑니다.)
        """
        batch_size = x.shape[0]

        hidden = self.encoder(x).unsqueeze(0)

        
        outputs = torch.zeros(self.seq_len, batch_size, decoder_params['num_words']).to(self.device)

        input = self.tokenizer.word2id.get('<SOS>') # start of sentence 토큰
        
        assert input is not None, "Can't fine <SOS> in tokenizer. "

        input = torch.tensor(input).repeat(batch_size).to(self.device) # batch 크기만큼 반복해서 증가시킴

        for c in range(0, self.seq_len):
            output, hidden = self.decoder(input, hidden)

            outputs[c] = output

            if teacher_force:
                assert y is not None, "y should be needed if teacher_force"
                input = y[:c]
            else:
                input = output.squeeze(0).argmax(dim= 1) # softmax 대용?

        return outputs