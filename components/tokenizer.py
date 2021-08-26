import torch
import torch.nn.functional as F

# 예측 해야하는 글자
target_text = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑똥뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없엇엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘ㅋㅎㄱㅅ"

class Tokenizer:
    def __init__(self, seq_len = 10, one_hot = False):
        self.seq_len = seq_len
        self.one_hot = one_hot
        self.word2id = {}
        self.id2word = {}

        self.word2id['<BLANK>'] = 0 # CTC loss blank word
        self.word2id['<UNK>'] = 1 # unknown token
        self.word2id['<PAD>'] = 2 # padding token

        self.id2word[0] = '<BLANK>'
        self.id2word[1] = '<UNK>'
        self.id2word[2] = '<PAD>'
        


        # 글자를 한 글자씩 둘러보면서 사전에 저장
        for idx, c in enumerate(target_text):
            self.word2id[c] = idx + 3
            self.id2word[idx + 3] = c
        
        


    def encode(self, input_str): # dataset output을 토크나이즈함

        result = []
        sent_len = []

        for sent in input_str: # 배치를 1문장씩 쪼개서 반복
            sent = sent[:self.seq_len - 1] # input_str를 잘라서 최대 글자에 맞춘다.
            
            token_sent = []
            # 입력 텍스트를 한 글자씩 돌리면서 번호로 교체
            for c in sent:
                if self.word2id.get(c) is not None: # 사전에 글자가 있으면 토크나이징
                    token_sent.append(self.word2id[c])
                else: # 없으면 <UNK> 토큰 배정
                    token_sent.append(self.word2id['<UNK>'])

            sent_len.append(len(token_sent)) # 현재 문장의 길이를 sent_len에 입력합니다.
            result += token_sent
        
        sent_len = torch.tensor(sent_len, dtype=torch.int32)
        result = torch.tensor(result, dtype=torch.int32)
        return result, sent_len

    def __call__(self, input_str):
        return self.tokenize(input_str)

    def tokenize(self, input_str): # 하나의 문장을 tokenize 할 때 사용. 입력값을 단순 스트링으로 가정
        result = []
        input_str = input_str[:self.seq_len] # input_str를 잘라서 최대 글자에 맞춘다.
        
        # 입력 텍스트를 한 글자씩 돌리면서 번호로 교체
        for c in input_str:
            if self.word2id.get(c) is not None: # 사전에 글자가 있으면 토크나이징
                result.append(self.word2id[c])
            else: # 없으면 <UNK> 토큰 배정
                result.append(self.word2id['<UNK>'])

        if self.one_hot:
            result = F.one_hot(torch.tensor(result), num_classes=len(self.word2id) + 1) # num_classes 는 0을 포함하므로 하나 더 더해주어야 한다.
        else:
            result = torch.tensor(result)

        return result


    def decode(self, input_vec): # 주어진 문장을 string 값으로 변경함 (one-hot 형식의 다차원([batch, seq_len, num_words]) softmax 결과값을 예상합니다.)
        result = []
        for sent in input_vec: # 배치 값별로 반복
            word = ""
            for char in sent:
                word += self.id2word[char.argmax().item()]

            result.append(word)
        return result

    def untokenize(self, input_vec): # 단어 하나의 값을 예측 (입력값은 상수 배열 [seq_len, num_words])
        result = ""

        for char in input_vec: # <EOS>를 만날 때 까지 계속함
            result += self.id2word[char.argmax().item()]
                
        return result