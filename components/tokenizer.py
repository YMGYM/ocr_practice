import torch
import torch.nn.functional as F
import re

# 예측 해야하는 글자
target_text = ' !"#%\'()*+,-./0123456789:<>?ABCDEFGKLMNOPQRSTUVXYZ[]^acdeghiklmnoprstuvx~–‘’“”…─ㄱㄴㅂㅇㅋㅎ가각간갇갈갉감갑값갓갔강갖갗같갚갛개객갠갯갱갸걀걔걘거걱건걷걸검겁것겉게겐겠겨격겪견결겸겹겼경곁계고곡곤곧골곪곯곰곱곳공과곽관괄광괘괜괴굉교구국군굳굴굵굶굼굽궁궂권궐궤귀귓규균그극근글긁금급긋긍기긱긴길김깁깃깄깊까깍깎깐깔깜깝깟깡깥깨깬깰깻깼깽꺄꺼꺽꺾껄껍껏껐껑께껴꼈꼬꼭꼰꼴꼼꼽꽁꽂꽃꽈꽉꽝꽤꾀꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿨꿰뀌뀐뀔끄끅끈끊끌끓끔끗끙끝끼끽낀낄낌낍낑나낙낚난날낡남납낫났낭낮낯낱낳내낸낼냄냅냈냉냐냔냥너넉넋넌널넓넘넝넣네넥넨넬넵넷넸녀녁년념녔녕노녹논놀놈농높놓놔놨뇌뇨누눅눈눌눕눠눴뉘뉜뉴느늑는늘늙늠능늦늪늬니닉닌닐님닙닛닝다닥닦단닫달닭닮닳담답닷당닿대댄댈댑댓댔댕더덕던덜덟덤덥덧덩덮데덴델뎌뎠도독돈돋돌돔돕돗동돼됐되된될됨됩두둑둔둘둠둡둥둬뒀뒈뒤뒷뒹드득든듣들듬듭듯등디딘딛딜딨딩딪따딱딴딸땀땄땅때땐땠땡떠떡떤떨떳떴떵떻떼뗀뗄뗏또똑똘똥똬뚜뚝뚤뚫뚱뛰뛴뛸뜁뜨뜩뜬뜯뜰뜸뜹뜻띄띈띠띤띵라락란랄람랍랏랐랑랗래랙랜램랩랫랬랭략량러럭런럴럼럽럿렀렁렇레렉렌렐렘렛려력련렬렴렵렷렸령례로록론롤롭롯롱롸뢰료룡루룩룬룰룸룹룻뤄류륙륜률륭르륵른를름릅릇릉릎리릭린릴림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맴맵맸맹맺머먹먼멀멈멋멍멎멓메멘멜멤멩며멱면멸명몇모목몫몬몰몸몹못몽뫼묘묠무묵묶문묻물뭇뭉뭍뭐뭔뭘뭡뭣뮤뮨뮬므미민믿밀밌밍및밑바박밖반받발밝밟밤밥밧방밭배백밸뱀뱃뱅뱉버벅번벌범법벗벙베벤벨벼벽변별볍볐병볕보복볶본볼봄봅봇봉봐봤봬뵈뵙부북분불붉붐붓붕붙뷔뷰브블비빅빈빌빕빗빙빚빛빠빡빤빨빳빴빵빼빽뺄뺏뺨뻐뻑뻔뻗뻘뻣뻥뼈뼘뼛뽀뽐뽑뽕뾰뿅뿌뿐뿔뿜쁘쁜쁠쁨삐삑사삭산살삶삼삽삿샀상샅새색샌샐샘샛생샤서석섞선섣설섬섭섯섰성세센셀셈셉셋셔션셨셰소속손솔솜솟송쇄쇠쇼숑수숙순숟술숨숫숭숲숴쉈쉬쉰쉴쉽쉿슈슉슛스슥슨슬슴습슷승시식신실싫심십싯싱싶싸싹싼쌀쌌쌍쌓쌘써썩썰썹썼썽쎄쏘쏙쏜쏟쏠쏴쐐쑤쑥쓰쓱쓴쓸씀씁씌씨씩씬씸씹씻아악안앉않알앓암압앗았앙앞애액앤앨앰앱야약얀얄얇얌양얕얗얘얜어억언얹얻얼얽엄업없엇었엉엎에엑엔엘엣엥여역엮연열염엽엾엿였영옅옆옇예옙옛오옥온올옭옮옳옴옵옷옹와왁완왈왓왔왕왜왠왱외왼요욕용우욱운울움웁웃웅워원월웠웨웩웬웹위윈윌윗윙유육윤율융으윽은을읊음읍읏응의이익인일읽잃임입잇있잉잊잎자작잔잖잘잠잡잣잤장잦재잴잼잽잿쟁쟤쟨저적전절젊점접젓정젖제젝젠젯져졌조족존졸좀좁종좆좇좋좌죄죗죠주죽준줄줌줍중줘줬쥐쥔쥘쥡즈즉즌즐즘즙증지직진질짊짐집짓징짖짙짚짜짝짠짤짧짬짭짱째쨌쨍쩌쩍쩐쩔쩡쪼쪽쫄쫓쬐쭈쭉쭤쯤쯧찌찍찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챘챙처척천철첨첩첫청체첸쳇쳐쳤초촉촌총촤촬최쵸추축춘출춤춥충춰췄취츠측츨츰츳층치칙친칠침칩칫칭카칵칸칼캄캐캔캠캡캬커컥컨컫컬컴컷컸컹케켄켈켓켜켠켰코콕콘콜콤콥콧콩콰콱콸쾅쾌쿄쿠쿡쿨쿵퀀퀘퀴퀸큐크큭큰클큼큿킁키킥킨킬킴킵킷킹타탁탄탈탐탑탓탔탕태택탰탱터턱턴털텀텁텅테텍텐텔템텨텼토톡톤톰톱톳통퇴투툭툰툴툼퉁퉤튀튄튈튕튜트특튼틀틈틋틔티틱틴틸팀팁팅파팍판팔팝팟팠팡패팩팬팻팽퍼퍽펀펄펐펑페펙펜펠펫펴편펼폈평폐포폭폰폴폼표푸푹푼풀품풋풍퓨프픈플픔피픽핀필핍핏핑하학한할핥함합핫항해핵햄햇했행향허헉헌헐험헛헝헤헥헬헴헷혀혁현혈혐협혓혔형혜호혹혼홀홈홉홍화확환활황홰홱횃회획횟횡효후훅훈훌훑훔훗훤훨훼휑휘휩휴흉흐흑흔흘흙흠흡흣흥흩희흰흴히힌힐힘힙'

class Tokenizer:
    def __init__(self, seq_len = 14, one_hot = False):
        self.target_text = target_text
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
            temp = ''

            for idx, char in enumerate(sent):
                dc_char = self.id2word[char.argmax().item()]
                if idx == 0:
                    temp = dc_char
                    word += dc_char
                else:
                    if temp == dc_char:
                        continue
                    else:
                        word += dc_char
                        temp = dc_char

            # <Blank 토큰 제거>
            word = re.sub(r"<BLANK>","",word)

            result.append(word)

        return result

    def untokenize(self, input_vec): # 단어 하나의 값을 예측 (입력값은 상수 배열 [seq_len, num_words])
        result = ""
        temp = ''
        for idx, char in enumerate(input_vec):
            dc_char = self.id2word[char.argmax().item()]
            if idx == 0:
                temp = dc_char
                result += dc_char
            else:
                if temp == dc_char:
                    continue
                else:
                    result += dc_char
                    temp = dc_char

        # <Blank 토큰 제거>
        result = re.sub(r"<BLANK>","",result)
                
        return result

    
    def make_answer_word(self, sent):
        # 입력받은 정답 단어에서 없는 글자만 UNK 토큰으로 변경합니다.
        # 정답 데이터를 유추할 때 사용할 예정.
        # 중복 제거 리펙토링이 필요할 것 같습니다.

        translated_sent = ""
        for c in sent:
                if self.word2id.get(c) is not None: # 사전에 글자가 있으면 토크나이징
                    translated_sent += c
                else: # 없으면 <UNK> 토큰 배정
                    translated_sent += '<UNK>'
                    
        return translated_sent