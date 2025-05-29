from models.blip.med import BertConfig, BertModel, BertLMHeadModel
from models.blip.blip_captioning import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.prune_utils import inherit_encoder_decoder_masks

class BLIPVQA(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/blip/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                   
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """         
        print("[Debug] blip_vqa.py : BLIPVQA 클래스 init() 함수 호출")      
        super().__init__()
        # Vision encoder 생성
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()
        # Text encoder 생성
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False) 
        # Text decoder 생성
        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)


    def forward(self, image, question, answer=None, k=None, weights=None, train=True, inference='rank', k_test=128):
        print("[Debug] blip_vqa.py : BLIPVQA 클래스 forward() 함수 실행")
        image_embeds = self.visual_encoder(image)   # Image 임베딩 -> Question 임베딩과 cross-attention에서 상호작용할 때 쓰임

        # 이미지 임베딩 벡터의 어텐션 마스크 : 중요한 임베딩에 1로 표시하여 나타냄
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device) 
        
        # question = self.tokenizer(question, padding='longest', truncation=True, max_length=35, 
        #                           return_tensors="pt").to(image.device)

        # Question 전처리 : Tokenize된 Question 텍스트의 첫 번째 시작 token을 enc_token_id로 교체 -> 질문의 시작을 특수한 Token으로 명시하기 위한 전략
        question.input_ids[:,0] = self.tokenizer.enc_token_id
        
        if train:  # Train 모드가 True일 때           
            '''
            n: number of answers for each question
            weights: weight for each answer
            '''                     
            # answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device) 
            # Answer 전처리 : Tokenize된 Answer 텍스트의 첫 번째 시작 token을 bos_token_id로 교체 -> 정답 시퀀스의 시작점임을 명시하기 위한 전략
            answer.input_ids[:,0] = self.tokenizer.bos_token_id
            # 패딩 token의 위치는 학습 시에 loss에 반영하지 않기 위해 -100으로 설정(Pytorch의 CELoss에서는 -100은 손실 계산에서 제외함)
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

            # question 토큰을 Text encoder에 입력하여 인코딩 -> 인코딩된 question 토큰과 이미지 임베딩과 cross-attention을 통해 상호작용
            # 결과가 딕셔너리 형태로 반환됨(이미지 정보와 질문의 연관성을 내포한 임베딩 벡터)
            question_output = self.text_encoder(question.input_ids,  # question tokens
                                                attention_mask = question.attention_mask, # question mask 
                                                encoder_hidden_states = image_embeds, # image embeddings
                                                encoder_attention_mask = image_atts, # image mask         
                                                return_dict = True)    

            question_states = []                
            question_atts = []  
            # decoder에 답변 후보와 질문을 줄 때, 한 질문에 대해 n개 답변 후보가 있을 때, 한 번 계산된 질문 인코딩을 n번 복제해 “질문–답변” 쌍끼리 매핑

            for b, n in enumerate(k): # Batch 내 각 질문의 개수(b)와 각 질문에 맞는 답변의 개수(n) -> EarthVQA는 1질문-1정답임
                question_states += [question_output.last_hidden_state[b]]*n # b(1)개 질문에 대해 각 질문에 대한 인코딩 결과를 답변 개수 n(1개)만큼 복제해서 리스트에 추가 
                question_atts += [question.attention_mask[b]]*n # 각 질문에 대한 답변 개수 만큼 어텐션 마스크 리스트에 복제              
            question_states = torch.stack(question_states,0) # 리스트에 있는 모든 텐서를 하나의 텐서로 합친다
            question_atts = torch.stack(question_atts,0)   # 리스트에 있는 모든 텐서를 하나의 텐서로 합친다

            # Decoder에 정답 시퀀스와 해당 attention mask(실제 단어와 패딩 구분), 질문에 대한 인코딩 결과를 넣어 모델이 정답을 예측하도록 함
            # 주어진 답변 후보 시퀀스를 기반으로, 실제 answer와의 loss를 계산하는 역할 
            answer_output = self.text_decoder(answer.input_ids, 
                                              attention_mask = answer.attention_mask, # 실제 정답의 어텐션 마스크
                                              encoder_hidden_states = question_states, # encoder의 output(이미지 정보와 질문의 연관성을 내포한 임베딩 벡터)을 후보 개수만큼 담은 list
                                              encoder_attention_mask = question_atts, # encoder의 output의 어텐션 마스크                
                                              labels = answer_targets, # 실제 정답을 전달하여 예측한 정답과의 loss를 계산하도록 함
                                              return_dict = True,   
                                              reduction = 'none',
                                             ) 
                 
            # 후보 정답 개수별로 encoder를 통해 출력된 질문 임베딩을 decoder에 넣어 디코딩을 통해 예측 정답에 대한 로짓을 계산
            # 각 질문 임베딩별 출력된 예측 로짓과 후보 정답 값을 CE Loss를 통해 각각 loss를 구한다.

            #텍스트 디코더는 “질문+이미지” 인코딩 정보를 기반으로 답변 후보를 예측하고, Label을 이용하여 정확하게 예측되었는지 평가, loss계산


            loss = weights * answer_output.loss # 계산된 각 정답 후보별 loss에 가중치를 곱하여 더 중요한 후보의 loss를 크게 반영되도록 함
            loss = loss.sum()/image.size(0) # batch내 모든 loss 값을 합산하여 batch로 나누어 batch당 평균 손실을 구함

            return loss
            

        else: # Train 모드가 아닐 때(추론 모드)
            # question 토큰을 Text encoder에 입력하여 인코딩 -> 인코딩된 question 토큰과 이미지 임베딩과 cross-attention을 통해 상호작용
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True) 
            
            # if inference=='generate': # 디코더를 통해 답변을 생성
            #     num_beams = 3  # Beam search에서 3가지 후보를 고려
            #     #Beam search : 순차적인 데이터 생성할 때 사용하는 탐색 알고리즘 
            #     # 생성할 다음 후보 단어들에 대해 1가지만 선택하는 것이 아니라 3개의 후보에 대해 모두 고려하여 탐색하는 방법

            #     # encoder를 통해 각 batch 내 질문 시퀀스들에 대한 답변 후보 임베딩들을 num_beams만큼 반복하여 저장
            #     # 각 질문마다 여러 후보를 동시에 평가하기 위함
            #     question_states = question_output.last_hidden_state.repeat_interleave(num_beams, dim=0)
                
            #     # Batch 내 토큰에 대해 모두 1로 마스킹하여 모두 유효한 정보임을 표시함
            #     question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
                
            #     # Decoder에 추가적으로 전달할 인자들을 딕셔너리 형태로 저장함
            #     model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
                
            #     # Batch 크기만큼, 각 샘플마다 시작 토큰을 정의함
            #     bos_ids = torch.full((image.size(0),1),fill_value=self.tokenizer.bos_token_id,device=image.device)
                
            #     # Decoder generate 함수 호출
            #     outputs = self.text_decoder.generate(input_ids=bos_ids,
            #                                          max_length=10,
            #                                          min_length=1,
            #                                          num_beams=num_beams,
            #                                          eos_token_id=self.tokenizer.sep_token_id,
            #                                          pad_token_id=self.tokenizer.pad_token_id, 
            #                                          **model_kwargs)
                
            #     answers = []    
            #     for output in outputs: # 생성된 각 답변 토큰 시퀀스를 순회하며 특수 토큰을 제외하여 텍스트 문자열로 반환한다.
            #         answer = self.tokenizer.decode(output, skip_special_tokens=True)    
            #         answers.append(answer)
            #     return answers
            if inference == 'generate':
                num_beams = 3

                # (1) 질문 임베딩 (원본 크기)
                question_states = question_output.last_hidden_state      # (batch, seq_len, dim)
                question_atts   = torch.ones(
                    question_states.size()[:2],                          # (batch, seq_len)
                    dtype=torch.long,
                    device=question_states.device
                )

                # (2) decoder.generate() 에 넘길 키워드를 원래대로
                model_kwargs = {
                    "encoder_hidden_states": question_states,
                    "encoder_attention_mask":   question_atts
                }

                # (3) 시작 토큰
                bos_ids = torch.full(
                    (image.size(0), 1),
                    fill_value=self.tokenizer.bos_token_id,
                    device=image.device
                )

                # (4) generate 호출
                outputs = self.text_decoder.generate(
                    input_ids=bos_ids,
                    max_length=10,
                    min_length=1,
                    num_beams=num_beams,
                    eos_token_id=self.tokenizer.sep_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **model_kwargs
                )

                # (5) 디코딩
                answers = [
                    self.tokenizer.decode(o, skip_special_tokens=True)
                    for o in outputs
                ]
                return answers


            elif inference=='rank': # 미리 주어진 후보 답변들 중에서 질문과 가장 잘 맞는 답변을 평가하고 선택하는 과정
                max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask, 
                                           answer.input_ids, answer.attention_mask, k_test) # 상위 k_test개만큼의 답변에 대해 평가
                return max_ids # 가장 높은 점수를 받은 답변 정보를 반환한다.
 
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0) # Batch 내 질문 개수
        start_ids = answer_ids[0,0].repeat(num_ques,1) # 전처리한 answer의 bos token을 가져와 각 question에 대해 bos token을 복제하여 텐서를 만듦
        # 후보 정답 집합에서 첫 토큰(BOS 토큰)을 각 질문마다 반복해서 준비(모든 질문에 대해 ‘시작’ 버튼을 누른다고 생각)
        
        # bos token을 입력으로 디코더를 실행하여, 각 질문에 대한 output 출력
        # 질문마다 bos token(시작 버튼)을 넣어 각 질문에 대한 예측 결과를 얻는다
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # 각 질문에 대한 답변으로 후보 답변들의 첫 토큰이 나올 예측 점수를 추출
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1] # 각 후보 답변들의 두 번째 토큰(bos token다음에 오늘 실제 첫 단어)를 가져온다.

        # decoder가 각 질문에 대해 후보 답변들 중 어떤 단어가 가장 가능성이 높은지 확률로 계산한다.
        # 후보 답변들이 첫 토큰(bos_token)에서 얼마나 유망한지를 평가할 수 있게 된다.
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) #각 질문에 대해 상위 k개의 후보를 선택하고, 이 후보들의 인덱스를 기록
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        #각 질문마다 상위 k개로 선택된 토큰들의 전체 토큰 시퀀스를 찾아내어 하나의 텐서로 결합한다.
        for b, topk_id in enumerate(topk_ids): # 각 질문별로 상위 k개의 후보 답변의 인덱스를 담은 텐서
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        # 후보 답변 시퀀스에서, 패딩 토큰은 loss 계산에서 무시되도록 마스킹한다.
        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)


        # repeat encoder's output for top-k answers
        #각 질문에 대해 선택된 후보 답변과 질문의 encoding 결과를 일대일로 대응시킴
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        # 각 답변 후보 전체 시퀀스와 인코딩된 질문을 decoder에 넣어 해당 답변이 질문과 얼마나 잘 맞는지 평가한다.
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')   
        
        # 각 질문에 대한 후보 답변과 정답에 대해 계산된 CELoss로, 손실이 낮을 수록 후보 답변의 예측이 좋다
        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques,k)

        max_topk_ids = log_probs_sum.argmax(dim=1) 
        max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]

        return max_ids # 각 질문에 대해 가장 높은 점수를 받은 후보 답변의 인덱스를 반환
    
    
    def load_from_pruned_pretrained(self, pretraining_weights, mask, config, is_eval=False):
        print("[Debug] blip_vqa.py : load_from_pruned_pretrained() 함수 호출 -> pruning mask 적용")
        self.load_pretrained(pretraining_weights, config)

        print(f"Loading from mask at: {mask}")
        mask = torch.load(mask, map_location="cpu")
        mask = inherit_encoder_decoder_masks(mask)
        msg = self.load_state_dict(mask, strict=False) # pruning mask를 가중치에 적용함
        relevant_missing_keys = [k for k in msg.missing_keys if "bias" not in k and "layernorm" not in k.lower() and "pruning_mask" in k]
        if len(relevant_missing_keys) > 0:
            print(f"missing keys: {relevant_missing_keys}")
        
        keys_to_exclude = ["bias", "layernorm", "pruning_mask", "text_encoder_m", "visual_encoder_m"]
        relevant_unexpected_keys = [k for k in msg.unexpected_keys if not any([x in k.lower() for x in keys_to_exclude])]
        if len(relevant_unexpected_keys) > 0:
            print(f"unexpected keys: {relevant_unexpected_keys}")


    def load_pretrained(self, weights_ckpt, config, is_eval=False):
        print("[Debug] blip_vqa.py : load_pretrained() 함수 호출 -> pre-trained된 가중치 정보 출력")
        print("Loaded params from: ", weights_ckpt)
        _, msg = load_checkpoint(self, weights_ckpt)
        relevant_missing_keys = [k for k in msg.missing_keys if "pruning_mask" not in k]
        if len(relevant_missing_keys) > 0:
            print(f"missing keys: {relevant_missing_keys}")

        # the checkpoint also contains the weights of the momentum encoders, which are not to be loaded 
        keys_to_exclude = ["visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m", "text_decoder"]
        relevant_unexpected_keys = [k for k in msg.unexpected_keys if not any([x in k for x in keys_to_exclude])]
        if len(relevant_unexpected_keys) > 0:
            print(f"unexpected keys: {relevant_unexpected_keys}")
    
    
def blip_vqa(pretrained='',**kwargs):
    model = BLIPVQA(**kwargs)
    if pretrained:
        model, _ = load_checkpoint(model,pretrained)
#         assert(len(msg.missing_keys)==0)
    return model  


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
        
        