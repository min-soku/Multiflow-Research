"""Evaluation functions with a shared signature for VQA, used in vqa.py"""
# 이 파일은 VQA(Vision Question Answering) 모델 평가를 위한 유틸리티를 제공합니다.
# blip_evaluation: BLIP 기반 VQA 평가
# xvlm_evaluation: XVLM 기반 VQA 평가
import os
import random
import json
import torch
import lightning as L
import re
from functools import partial
from utils.misc import mprint as print

# 질문 ID에 따라 무작위 답변을 생성하는 디버그용 페이크 함수
# ques_id: 질문 식별자, answer_list: 가능한 답변 목록
def vqa_fake_result(ques_id, answer_list):
    return {"question_id": ques_id, "answer": random.choice(answer_list)}

def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r'\s*%\s*', '%', ans)
    ans = re.sub(r'\s*-\s*', '-', ans)
    ans = re.sub(r'\s+', ' ', ans)
    return ans

@torch.no_grad()
# BLIP 모델을 이용하여 VQA 평가를 수행하는 함수
# output_dir: 결과를 저장할 디렉토리 경로
# model: 평가할 VQA 모델
# data_loader: 이미지∙질문∙질문ID 배치를 제공하는 데이터 로더
# tokenizer: 질문을 토크나이즈할 토크나이저
# fabric: Lightning Fabric 객체 (분산 환경 설정)
# config: 평가 설정을 담은 딕셔너리 (inference 방식, 출력 빈도, k_test 등)
# split: 검증/테스트 데이터 구분 ("val" 또는 "test")
# debug_mode: True일 경우 일부 샘플에 페이크 결과 사용
def blip_evaluation(output_dir, model, data_loader, tokenizer, fabric, config, split="val", debug_mode=False):
    model.eval()  # 모델을 평가 모드로 전환하여 드롭아웃 등 비활성화
    device = fabric.device  # 연산에 사용할 디바이스(CPU/GPU)
    rank = fabric.global_rank  # 분산 환경에서의 프로세스 랭크

    result = []  # 평가 결과를 저장할 리스트
    computed = 0  # 처리한 배치 수 카운터

    # 디버그 모드: 페이크 결과 생성을 위한 partial 함수 준비
    if debug_mode:
        vqa_fake_result_partial = partial(vqa_fake_result, answer_list=data_loader.dataset.answer_list)
    
    # 랭크 기반 추론시: 모든 답변 후보를 토크나이즈하여 미리 준비
    if config['inference'] == 'rank':   
        answer_list = data_loader.dataset.answer_list  # 가능한 답변 목록
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
        # 모든 후보 입력의 첫 토큰을 BOS로 설정
        answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id
        
    # 배치별 평가 수행
    for batch_idx, (image, question, question_id) in enumerate(data_loader):       
        # 일정 주기마다 진행 상황 출력
        if batch_idx % config['print_freq'] == 0:
            print(f"[Evaluation]\tBatch {batch_idx}/{len(data_loader)}") 
        
        # 디버그 모드에서 일부 이후 샘플은 페이크 결과로 대체
        if debug_mode and computed > 100:
            result_for_this_batch = [vqa_fake_result_partial(ques_id.item()) for ques_id in question_id]
            result += result_for_this_batch

        # 생성 기반 추론: 모델이 직접 답변 시퀀스를 생성
        elif config['inference'] == 'generate':
            print("[Debug] Generation inference 수행")
            question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
            answers = model(image, question_input, train=False, inference='generate')
            # for answer, ques_id in zip(answers, question_id):
            #     ques_id = int(ques_id.item())
            #     result.append({"question_id": ques_id, "answer": answer})
            for answer, ques_id in zip(answers, question_id):
               # ques_id 가 tensor 면 .item(), 아니면 바로 int()
               if hasattr(ques_id, "item"):
                   qid = int(ques_id.item())
               else:
                   qid = int(ques_id)
               # 정규화 적용
               norm_answer = normalize_answer(answer)    
               result.append({"question_id": qid, "answer": norm_answer})
            computed += 1
        
        # 랭크 기반 추론: 미리 준비한 후보 중 상위 k_test 답안 선택
        elif config['inference'] == 'rank':
            print("[Debug] Rank inference 수행")
            question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
            answer_ids = model(image, question_input, answer_candidates, train=False, inference='rank', k_test=config['k_test'])
            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id": int(ques_id), "answer": answer_list[answer_id]})
            computed += 1

    # 평가 결과를 JSON 파일로 저장
    outpath = os.path.join(output_dir, f"vqa_{split}_{rank}.json")
    print(f"Dumping eval file at {outpath}")
    with open(outpath, "w") as f:
        json.dump(result, f)
    
    return result  # 최종 평가 결과 반환

@torch.no_grad()
# XVLM 모델을 이용하여 VQA 평가를 수행하는 함수
# output_dir: 결과를 저장할 디렉토리 경로
# model: 평가할 VQA 모델
# data_loader: 이미지∙질문∙질문ID 배치를 제공하는 데이터 로더
# tokenizer: 질문을 토크나이즈할 토크나이저
# fabric: Lightning Fabric 객체 (분산 환경 설정)
# config: 평가 설정을 담은 딕셔너리 (inference 방식, 출력 빈도, k_test 등)
# split: 검증/테스트 데이터 구분 ("val" 또는 "test")
# debug_mode: True일 경우 일부 샘플에 페이크 결과 사용
def xvlm_evaluation(output_dir, model, data_loader, tokenizer, fabric: L.Fabric, config, split="val", debug_mode=False):
    assert split in ("val", "test")  # split 값이 "val" 또는 "test"인지 확인
    model.eval()  # 모델을 평가 모드로 전환하여 드롭아웃 등 비활성화
    device = fabric.device  # 연산에 사용할 디바이스(CPU/GPU)
    rank = fabric.global_rank  # 분산 환경에서의 프로세스 랭크
    
    # 가능한 답변 목록에 EOS 토큰 추가 후 토크나이즈
    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

    result = []  # 평가 결과를 저장할 리스트
    computed = 0  # 처리한 배치 수 카운터

    # 디버그 모드: 페이크 결과 생성을 위한 partial 함수 준비
    if debug_mode:
        vqa_fake_result_partial = partial(vqa_fake_result, answer_list=data_loader.dataset.answer_list)
    
    # 배치별 평가 수행
    for batch_idx, (image, question, question_id) in enumerate(data_loader):    
        # 일정 주기마다 진행 상황 출력
        if batch_idx % config['print_freq'] == 0:
            print(f"[Evaluation]\tBatch {batch_idx}/{len(data_loader)}")
        
        # 디버그 모드에서 일부 이후 샘플은 페이크 결과로 대체
        if debug_mode and computed > 100:
            result_for_this_batch = [vqa_fake_result_partial(ques_id.item()) for ques_id in question_id]
            result += result_for_this_batch
        else:
            # 질문을 토크나이즈하여 모델에 입력
            question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
            # 모델이 상위 k_test 답변 후보를 선택
            topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])
            
            for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
                ques_id = int(ques_id.item())
                _, pred = topk_prob.max(dim=0)  # 가장 높은 확률의 답변 선택
                ans = data_loader.dataset.answer_list[topk_id[pred]]
                result.append({"question_id": ques_id, "answer": ans})
                
            computed += 1

    # 평가 결과를 JSON 파일로 저장
    outpath = os.path.join(output_dir, f"vqa_{split}_{rank}.json")
    print(f"Dumping eval file at {outpath}")
    with open(outpath, "w") as f:
        json.dump(result, f)

    return result  # 최종 평가 결과 반환