import os
import sys
from random import randint
import json

import traceback
import torch
from torch.utils.data import Dataset

from PIL import Image
from PIL import Image
from collections import OrderedDict
from datasets.utils import pre_caption


def coco_karpathy_train_collate_fn(batch, tokenizer, max_tokens): # 한 번에 묶여들어온 Batch를 모델에 넣기 편한 형태로 변환하는 함수
    # unravel the batch
    images, captions, ids = [], [], []
    for img, cap, img_id in batch: # batch에 들어온 각 샘플을 순회하면서 데이터를 분리하여 각 list에 추가함
        images.append(img) # 이미지 텐서 리스트 [tensor(C,H,W), tensor(C,H,W), …]
        captions.append(cap) # 캡션 리스트 [str, str, …]
        ids.append(img_id) # 이미지 ID 리스트 [int, int, …]

    # generate the individual tensors
    images = torch.stack(images) # image 텐서 리스트를 하나의 4차원 텐서로 합침
    captions = tokenizer( # 배치 내 가장 긴 문장에 맞춰 패딩 후 Tokenizer를 통해 토큰화
        captions,
        padding='longest',
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt"
    )
    ids = torch.tensor(ids) # 이미지 ID 리스트를 텐서로 변환
    return images, captions, ids


class CocoCaptionTrainDataset(Dataset):
    def __init__(self, ann_file_list, transform, image_root='images/rsicd', max_words=30, prompt=''):

        # load all the json files containing the annotations
        self.annotation = []
        for f in ann_file_list:
            self.annotation += json.load(open(f, 'r'))

        # setup useful class attributes
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        # create a dictionary where the keys are the image ids,
        # and the values are ordered integers replacing the ids
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __private_getitem__(self, index):
        # load the annotation data
        ann = self.annotation[index]

        # load the image and apply transformations on top
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # preprocess the caption and prepend the prompt
        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        return image, caption, self.img_ids[ann['image_id']]

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

        # if we get here, it means the 10 retries failed
        error_path = os.path.join(
            self.image_root, self.annotation[index]['image'])
        raise RuntimeError(
            f"Failed to load data after 10 retries: {error_path}")


# class CocoCaptionEvalDataset(Dataset):
#     def __init__(self, ann_file, transform, image_root='images/rsicd'):
#         self.annotation = json.load(open(ann_file, 'r'))
#         self.transform = transform
#         self.image_root = image_root

#     def __len__(self):
#         return len(self.annotation)

#     def grab_image_id_from_image_path(self, image_path):
#         # equivalent to the following commented code, but keeping the old
#         # one for consistency
#         # >>> img_basename = os.path.basename(image_path)
#         # >>> img_name, ext = os.path.splitext(img_basename)
#         # >>> id_as_a_string = img_name.split('_')[-1]
#         # >>> return id_as_a_string
#         return image_path.split('/')[-1].strip('.jpg').split('_')[-1]

#     def __private_getitem__(self, index):

#         ann = self.annotation[index]

#         image_path = os.path.join(self.image_root, ann['image'])
#         image = Image.open(image_path).convert('RGB')
#         image = self.transform(image)

#         img_id = self.grab_image_id_from_image_path(image_path=ann['image'])
#         return image, int(img_id)

#     def __getitem__(self, index):
#         # call the __private_getitem__ method, but make it
#         # more robust to errors with try-catching, retries and logging
#         for _ in range(10):
#             try:
#                 return self.__private_getitem__(index)
#             except Exception as e:
#                 print(traceback.format_exc())
#                 print('encounter broken data: %s' % e)
#                 print('-'*20)
#                 sys.stdout.flush()

#         error_path = os.path.join(
#             self.image_root, self.annotation[index]['image'])
#         raise RuntimeError(
#             f"Failed to load data after 10 retries: {error_path}")


class CocoCaptionEvalDataset(Dataset):
    def __init__(self, ann_file, transform, image_root='images/rsicd', gt_coco_file=None):
        """
        ann_file       : rsicd_val.json 같은 [{ "image": "...", "caption": [...] }, ...]
        gt_coco_file   : test_gt.json / val_gt.json 처럼 COCO 포맷으로 된 GT 파일
        """
        self.annotation = json.load(open(ann_file, 'r'))
        self.transform  = transform
        self.image_root = image_root

        # 1) GT coco 파일에서 이미지별 정수 ID 매핑 로드
        assert gt_coco_file is not None, "gt_coco_file 경로를 넘겨주세요"
        coco_gt = json.load(open(gt_coco_file, 'r'))
        # coco_gt["images"] = [ {"file_name": "...", "id": X}, ... ]
        self.fname2id = {
            img["file_name"]: img["id"]
            for img in coco_gt["images"]
        }

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann     = self.annotation[index]
        relpath = ann['image']                     # e.g. "val_images/airport_61.jpg"
        full    = os.path.join(self.image_root, relpath)

        image = Image.open(full).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 2) self.fname2id 매핑에서 GT 정수 ID 가져오기
        img_id = self.fname2id[relpath]

        return image, img_id