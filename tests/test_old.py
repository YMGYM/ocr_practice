import os
from collections import Counter
from pathlib import Path
sys.path.append(str(Path('../../soma/ai/src').absolute()))

from model.vedastr.tools.inference import recognition_api

from PIL import Image

BASE_DIR = '/home/jun/myWorks/soma/font/dataset/realTesting'

open_img = []
name_list = []

for im in os.listdir(BASE_DIR):
    open_img.append(np.array(Image.open(BASE_DIR + '/' + im)))
    name_list.append(im.split('_')[0])


if __name__ == "__main__":
    totalCounter = Counter() # 모든 글자를 저장하는 카운터
    trueCounter = Counter() # 맞춘 글자를 저장하는 카운터
    falseCounter = Counter() # 오답 글자를 저장하는 카운터

    print("========== Start Testing ... ========== ")
    print(f"tokenizer word size : {len(tokenizer.word2id)}")

    # 학습된 model 불러오기 기능
    print("Load model from ", params['model_path'])

    result = []

    checkpoint = torch.load(params['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)

    model.eval()
    torch.no_grad()

    test_loss = 0.0
    for idx, data in tqdm(enumerate(test_dataset)):

        xs, original = data # 데이터 분리

        ys, y_len = tokenizer.encode(original) # 데이터 인코딩

        xs = xs.to(device)

        output = model(xs) # model.forward

        seq_shape = torch.full(size=(output.shape[1],), fill_value=output.shape[0])

        loss = criterion(output, ys, seq_shape, y_len) # evaluate loss

        test_loss += loss.item()
        
        output = output.permute(1,0,2) # (batch, seq, word_len)
        for i, sent in enumerate(output):
            result.append({'predict' : tokenizer.untokenize(sent), 'answer': tokenizer.make_answer_word(original[i]), 'original': original[i],})

            
    print(f"Test loss : {(test_loss / len(test_dataset) ):0.5f}")
    df = pd.DataFrame(result)
    df.to_csv(params['result_path']) # 데이터 저장
    print(f"Result Data was saved at {params['result_path']}")

    # 단어 별로 정답률을 체크합니다.
    accuracy = (df['predict'] == df['original']).values.sum() / len(df)
    print(f"accuracy(word):{accuracy * 100}%")

    for row in df.iterrows():
        for charIdx, char in enumerate(row[1]['original']):
            if char in tokenizer.target_text:
                totalCounter[char] += 1
                if charIdx < len(row[1]['predict']):
                    if char == row[1]['predict'][charIdx]: # 정답인경우
                        trueCounter[char] += 1
                    else:
                        falseCounter[char] += 1
                else:
                    falseCounter[char] += 1
            else:
                totalCounter['<OOV>'] += 1
                falseCounter['<OOV>'] += 1

    print(f"accuracy(character) :{sum(trueCounter.values())/sum(totalCounter.values()) * 100}%")
    

