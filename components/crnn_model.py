import torch
from torch import nn
import torch.nn.functional as F

"""
CRNN 모델을 구현합니다.
"""

"""하이퍼파라미터 설정"""
crnn_params = {
    'conv1_out': 64,
    'conv1_kernel_size' : 5,
    'conv2_out': 256,
    'conv2_kernel_size': 5,
    'dropout_ratio': 0.5,
    'rnn_hidden_size': 512,
    'rnn_bidirectional': True, # bidirectional LSTM 사용 유무
    'rnn_num_layers': 2, # RNN 계층을 몇개 쌓을 것인지
    'num_words': 1016, #  tokenizer의 word2id의 길이와 동일해야 함
}

"""하이퍼파라미터 설정 종료"""

class CRNN(nn.Module):

    def __init__(self, crnn_params=crnn_params):
        super(CRNN, self).__init__()

        self.params = crnn_params
        self.conv1 = nn.Conv2d(1, crnn_params['conv1_out'], crnn_params['conv1_kernel_size']) # in_channels, out_channels, kernel_size, stride
        # output = (x + 2*paddint - dilation * (kernel_size - 1)-1) / stride + 1
        # output = (64, 28, 76)
        self.pool = nn.MaxPool2d(2, 2) # output = (64, 14, 38)
        self.drop1 = nn.Dropout(crnn_params['dropout_ratio'])
        self.conv2 = nn.Conv2d(crnn_params['conv1_out'], crnn_params['conv2_out'], crnn_params['conv2_kernel_size'])
        # output = (256, 10, 34) : feature, width, height
        self.drop2 = nn.Dropout(crnn_params['dropout_ratio'])
        self.pool = nn.MaxPool2d(2, 2) # output = (256, 5, 17)

        self.rnn1 = nn.GRU(input_size= 256 * 17, hidden_size = crnn_params['rnn_hidden_size'], batch_first=True, num_layers=crnn_params['rnn_num_layers'], bidirectional=crnn_params['rnn_bidirectional'])
        # rnn output : (batch, 5, 256)

        self.fc_out = nn.Linear(crnn_params['rnn_hidden_size'] * 2, crnn_params['num_words'])
        # output : (batch, 5, 1015)

    def forward(self, x):
        
        # cnn 통과
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x) # (128, 10, 44)

        # flatten 실시
        x = x.permute(0, 2, 1, 3) # (batch, width, filter, height)
        x = torch.flatten(x, start_dim=2) # rnn input에 맞게 변형 : height 제거

        # rnn 통과
        x, _ = self.rnn1(x)

        # fc 통과
        
        x = self.fc_out(x)
        
        # softmax 통과
        x = x.permute(1, 0, 2)
        x = F.log_softmax(x, 2) # (seq, batch, word_num)
        return x