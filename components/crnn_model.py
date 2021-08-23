import torch
from torch import nn
import torch.nn.functional as F

"""
CRNN 모델을 구현합니다.
"""

"""하이퍼파라미터 설정"""
crnn_params = {
    'conv1_out': 64,
    'conv1_kernel_size' : 4,
    'conv2_out': 128,
    'conv2_kernel_size': 4,
    'dropout_ratio': 0.5,
    'rnn_hidden_size': 256,
    'rnn_bidirectional': True, # bidirectional LSTM 사용 유무
    'num_words': 1014, #  dataset의 word2id의 길이와 동일해야 함
}

"""하이퍼파라미터 설정 종료"""

class CRNN(nn.Module):

    def __init__(self, crnn_params):
        super(CRNN, self).__init__()

        self.params = crnn_params
        self.conv1 = nn.Conv2d(1, crnn_params['conv1_out'], crnn_params['conv1_kernel_size']) # in_channels, out_channels, kernel_size, stride
        # output = (x + 2*paddint - dilation * (kernel_size - 1)-1) / stride
        # output = (64, 28, 96)
        self.pool = nn.MaxPool2d(2, 2) # output = (64, 14, 48)
        self.drop1 = nn.Dropout(crnn_params['dropout_ratio'])
        self.conv2 = nn.Conv2d(crnn_params['conv1_out'], crnn_params['conv2_out'], crnn_params['conv2_kernel_size'])
        # output = (128, 10, 44) => (128, 5, 22) : feature, width, height
        self.drop2 = nn.Dropout(crnn_params['dropout_ratio'])


        self.rnn1 = nn.GRU(input_size= 128 * 22, hidden_size = crnn_params['rnn_hidden_size'])
        # rnn output : (5, batch, 256)

        self.fc_out = nn.Linear(crnn_params['rnn_hidden_size']*2, crnn_params['num_words'])
        # output : (batch, 1014)

    def forward(self, x):
        
        # cnn 통과
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x) # (128, 5, 22)

        # flatten 실시
        x = x.permute(2, 0, 1, 3)
        x = x.view(-1, 5, 128*22) # rnn input에 맞게 변형

        # rnn 통과
        x = self.rnn1(x)

        # fc 통과
        x = self.fc_out(x)

        return x