import os
import traceback
import sys
import json
from random import randint
from random import random as rand

from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets.utils import pre_question

from torchvision.transforms.functional import hflip
from transformers import BertTokenizer, RobertaTokenizer


# def vqa_collate_fn(batch): # 데이터셋의 batch를 만드는 함수로, 각 샘플의 데이터를 모델에 전달할 수 있게 준비함
#     print("[Debug] vqa_dataset.py : vqa_collate_fn() 함수 호출")
#     # 각 항목들을 별도의 list에 모아 batch 단위로 결합할 수 있게 하기 위함
#     image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
#     for image, question, answer, weights in batch: # batch에 들어있는 각 샘플을 순회하면서 데이터를 분리하여 각 list에 추가함
#         image_list.append(image)
#         question_list.append(question)
#         weight_list += weights       
#         answer_list += answer
#         n.append(len(answer))
#     # batch 반환 -> image는 stack으로 weight는 tensor로, question, answer는 list 형태 그대로 반환됨
#     return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n

def vqa_collate_fn(batch): # 입력된 train dataset의 batch의 개별 샘플을 하나의 batch로 묶어주는 함수
    image_list, question_list, answer_list, weight_list, n_list = [], [], [], [], []

    for (image, question, answers, weights) in batch:
        # answers: 예) ["Yes"] (단일 정답), 또는 [["Yes"], ["No"], ...] 일 수도 있음
        # 만약 answers 자체가 2차원이라면, 각 sublist를 평탄화
        # 예: [["No"], ["0"]] -> ["No","0"]
        flattened_answers = []                                                                                     
        for ans in answers:
            if isinstance(ans, list) and len(ans) == 1:
                flattened_answers.append(ans[0])  # 안에 있는 문자열만 추출
            elif isinstance(ans, list):
                # 혹시 여러개가 들어있는 경우 (희박하지만) 전부 이어 붙이거나 처리
                flattened_answers.extend(ans)
            else:
                flattened_answers.append(ans)

        image_list.append(image) # 입력된 batch내 image를 하나로 묶음
        question_list.append(question)
        # flattened_answers는 문자열들의 리스트가 됨 -> answer_list에 합치기
        answer_list.extend(flattened_answers)

        weight_list += weights
        n_list.append(len(flattened_answers)) # batch별로 정답 개수

    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n_list


#  vqa_root, vg_root -> earthvqa_root로 변경
class VQADataset(Dataset):
    def __init__(self, ann_file, transform, earthvqa_root, split="train", max_ques_words=30, answer_list='',
                 text_encoder='', use_roberta=False):
        print("[Debug] vqa_dataset.py : VQADataset 클래스 init()함수 호출 -> dataset 설정")
        self.careful_hflip = True

        self.split = split
        self.ann = []
        # 각 json 파일 순회
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.transform = transform
        self.earthvqa_root = earthvqa_root
        # self.vqa_root = vqa_root
        # self.vg_root = vg_root
        self.max_ques_words = max_ques_words

        try:
            # hugging face에서 pre-trained BertTokenizer 가져옴
            tokenizer = RobertaTokenizer.from_pretrained(text_encoder) if use_roberta else BertTokenizer.from_pretrained(text_encoder)
            self.pad_token_id = tokenizer.pad_token_id # 패딩 값으로 사용할 숫자
            self.eos_token = '</s>' if use_roberta else '[SEP]' # 문장 종료로 표시할 토큰
            print("[Debug] vqa_dataset.py : hugging face에서 pre-trained BertTokenizer 가져옴")
        except:
            print("[Debug] vqa_dataset_py : except 발생, Hugging face 라이브러리 미설치/text encoder 경로 오류로 BLIP 내장 tokenizer 사용")
            from models.blip.blip_captioning import init_tokenizer
            tokenizer = init_tokenizer()
            self.pad_token_id = tokenizer.pad_token_id
            self.eos_token = '[SEP]'

        if split == 'test': # test 전용 설정
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, 'r'))
        
    def __len__(self):
        return len(self.ann)

    def left_or_right_in(self, question, answer):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(question):
            return True

        if isinstance(answer, list):
            for ans in answer:
                if _func(ans):
                    return True
        else:
            if _func(answer):
                return True

        return False

    def __private_getitem__(self, index):

        ann = self.ann[index]

        if 'dataset' in ann.keys(): #vq, vqa => EarthVQA로 통합
            # if ann['dataset'] == 'vqa':
            #     image_path = os.path.join(self.vqa_root, ann['image'])
            if ann['dataset'] == 'EarthVQA': # earthvqa인 경우 디렉토리 루트 + 파일명으로 전체 이미지 경로 생성
                image_path = os.path.join(self.earthvqa_root, ann['image'])
            # elif ann['dataset'] == 'vg':
            #     image_path = os.path.join(self.vg_root, os.path.basename(ann['image']))
            elif ann['dataset'] == 'gqa':
                image_path = ann['image']
            else:
                raise NotImplementedError

        else:
            image_path = os.path.join(self.vqa_root, ann['image'])

        image = Image.open(image_path).convert('RGB')

        if (self.split != 'test') and rand() < 0.5: # 테스트셋이 아니면 50%확률로 이미지에 수평 반전을 적용하여 증강함
            if self.careful_hflip and self.left_or_right_in(ann['question'], ann['answer']):
                pass
            else:
                image = hflip(image)

        image = self.transform(image) # 이미지를 모델에 맞게 전처리

        if self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['question_id']
            return image, question, question_id # 

        elif self.split == 'train': # EarthVQA로 변경
            question = pre_question(ann['question'], self.max_ques_words)

            # if ('dataset' in ann.keys()) and (ann['dataset'] == 'vg'):
            #     answers = [ann['answer']]
            #     weights = [0.5]
            
            if ('dataset' in ann.keys()) and (ann['dataset'] == 'EarthVQA'):
                answers = [ann['answer']]
                weights = [0.5] #가중치가 무엇을 의미?

            # else:
            #     answer_weight = {}
            #     for answer in ann['answer']:
            #         if answer in answer_weight.keys():
            #             answer_weight[answer] += 1 / len(ann['answer'])
            #         else:
            #             answer_weight[answer] = 1 / len(ann['answer'])

            #     answers = list(answer_weight.keys())
            #     weights = list(answer_weight.values())

            # answers = [answer + self.eos_token for answer in answers]  # fix bug

            return image, question, answers, weights

        else:
            raise NotImplementedError
        
    def __getitem__(self, index):
        # call the __private_getitem__ method, but make it
        # more robust to errors with try-catching, retries and logging
        for _ in range(10):
            try:
                return self.__private_getitem__(index)
            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                print('-'*20)
                sys.stdout.flush()
        
        ann = self.ann[index]
        if 'dataset' in ann.keys():
            # if ann['dataset'] == 'vqa':
            #     error_path = os.path.join(self.vqa_root, ann['image'])
            if ann['dataset'] == 'EarthVQA':
                error_path = os.path.join(self.earthvqa_root, ann['image'])
            # elif ann['dataset'] == 'vg':
            #     error_path = os.path.join(self.vg_root, os.path.basename(ann['image']))
            elif ann['dataset'] == 'gqa':
                error_path = ann['image']
            else:
                error_path = os.path.join(self.vqa_root, ann['image'])
        else:
            error_path = os.path.join(self.vqa_root, ann['image'])
        raise RuntimeError(f"Failed to load data after 10 retries: {error_path}")