"""Evaluation Functions with a shared signature for Image-Text Retrieval, used in retrieval.py"""
import torch
import time
import datetime
import utils
import torch.nn.functional as F

@torch.no_grad()
def blip_evaluation(model, data_loader, tokenizer, fabric, config, debug_mode=False):
    # Debugging 메시지 출력
    print("[Debug] evaltools/itr_utils.py -> evaluation()함수 실행")
    
    # 시작 시간 기록
    start_time = time.time()  
    
    # 모델을 평가 모드로 설정
    model.eval() 
    
    # GPU 캐시 비우기
    torch.cuda.empty_cache()
    
    # Fabric에서 사용 중인 디바이스 가져오기
    device = fabric.device

    # 분산 학습 환경에서 작업 수와 순위 가져오기
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    # 데이터셋에서 텍스트와 이미지 데이터 가져오기
    texts = data_loader.dataset.text   # 텍스트 데이터
    num_text = len(texts)  # 텍스트 데이터 개수
    num_images = len(data_loader.dataset.image)  # 이미지 데이터 개수

    # 테스트 시 텍스트 배치 크기 설정
    text_bs = config['batch_size_test_text']  # 256
    
    print('Computing features for evaluation...')

    # I/T (Image-to-Text) 및 T/I (Text-to-Image) 점수를 저장할 행렬 초기화
    score_matrix_i2t = torch.full((num_images, num_text), -100.0).to(device)
    score_matrix_t2i = torch.full((num_text, num_images), -100.0).to(device)
    
    # 텍스트 데이터에서 특징 추출
    text_ids_list, text_embeds_list, text_atts_list = [], [], []
    for i in range(0, num_text, text_bs):
        # 현재 처리 중인 텍스트 인덱스 계산
        current_idx = min(num_text, i + text_bs)
        print(f"Processing text {current_idx} / {num_text}", end='\r' if i + text_bs < num_text else "\n")
        
        # 텍스트 배치 가져오기
        text = texts[i: min(num_text, i+text_bs)]
        
        # 토크나이저를 사용하여 텍스트를 입력 형식으로 변환
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=35, 
                               return_tensors="pt").to(device) 
        
        # 텍스트 인코더를 사용하여 텍스트 특징 추출
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        
        # 텍스트 특징을 정규화
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        
        # 특징 및 입력 데이터를 리스트에 추가
        text_embeds_list.append(text_embed.cpu())   
        text_ids_list.append(text_input.input_ids.cpu())
        text_atts_list.append(text_input.attention_mask.cpu())
        
        # 메모리 해제
        del text_input, text_output, text_embed

    # 리스트를 텐서로 병합하고 GPU로 이동
    text_embeds = torch.cat(text_embeds_list, dim=0).to(device); del text_embeds_list
    text_ids = torch.cat(text_ids_list,dim=0).to(device); del text_ids_list
    text_atts = torch.cat(text_atts_list, dim=0).to(device); del text_atts_list

    # 토크나이저의 시작 토큰 ID 설정
    text_ids[:, 0] = tokenizer.enc_token_id
    
    # extracting image data
    image_feats_list, image_embeds_list = [], []
    for i, (image, img_id) in enumerate(data_loader): 
        print(f"Processing image batch {i} / {len(data_loader)}", end='\r' if i != len(data_loader)-1 else "\n")
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats_list.append(image_feat.cpu())
        image_embeds_list.append(image_embed.cpu())
        del image, image_feat, image_embed
     
    torch.cuda.empty_cache()
    image_feats = torch.cat(image_feats_list,dim=0).to(device); del image_feats_list
    image_embeds = torch.cat(image_embeds_list, dim=0).to(device); del image_embeds_list
    
    # 이미지 특징과 텍스트 특징 간의 유사도 행렬 계산
    sims_matrix = image_embeds @ text_embeds.t()

    # 분산 학습 환경에서 각 GPU가 처리할 이미지 범위 설정
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    # 이미지-텍스트 유사도 계산 및 점수 행렬 업데이트
    for i, sims in enumerate(sims_matrix[start:end]): 
        if debug_mode and i == 300: break  # 디버그 모드에서 300개까지만 처리
        
        # 텍스트 중 상위 k개의 유사도와 인덱스 가져오기
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        # 텍스트 인코더를 사용하여 점수 계산
        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score + topk_sim
        del output; del encoder_att; del encoder_output
        
        # 진행 상황 로그 출력
        if (i+1) % 50 == 0 or (i+1) == end-start:
            print(f"[Evaluation] Processing image {i+1}/{end-start} ({(i+1)/(end-start)*100:.2f}%)")
        
    # GPU 메모리 캐시 비우기
    torch.cuda.empty_cache()
    
    # 유사도 행렬 전치
    sims_matrix = sims_matrix.t()
    
    # 분산 학습 환경에서 각 GPU가 처리할 텍스트 범위 설정
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0), start+step)    
    
    # 텍스트-이미지 유사도 계산 및 점수 행렬 업데이트
    for i, sims in enumerate(sims_matrix[start:end]): 
        if debug_mode and i == 300: break  # 디버그 모드에서 300개까지만 처리
        
        # 이미지 중 상위 k개의 유사도와 인덱스 가져오기
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score + topk_sim
        del output; del encoder_att; del encoder_output

        # 진행 상황 로그 출력
        if (i+1) % 50 == 0 or (i+1) == end-start:
            print(f"[Evaluation] Processing text {i+1}/{end-start} ({(i+1)/(end-start)*100:.2f}%)")

    # 모든 GPU에서 점수 행렬을 합산
    fabric.barrier()
    score_matrix_i2t = fabric.all_reduce(score_matrix_i2t, reduce_op="sum") 
    score_matrix_t2i = fabric.all_reduce(score_matrix_t2i, reduce_op="sum")        
        
    # 총 평가 시간 계산 및 출력
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    # 최종 점수 행렬 반환
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def xvlm_evaluation(model, data_loader, tokenizer, fabric, config, debug_mode=False):
    
    start_time = time.time()  
    model.eval()
    torch.cuda.empty_cache()
    device = fabric.device

    # some configs...
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    texts = data_loader.dataset.text   
    num_text = len(texts)
    num_images = len(data_loader.dataset.image)
    text_bs = config['batch_size_test_text']  # 256

    # some logging stuff...
    print('Computing features for evaluation...')

    # define matrices to store I/T and T/I scores
    score_matrix_i2t = torch.full((num_images, num_text), -100.0).to(device)
    score_matrix_t2i = torch.full((num_text, num_images), -100.0).to(device)

    # extracting text data from all the texts in the dataset
    text_feats_list, text_embeds_list, text_atts_list = [], [], []
    for i in range(0, num_text, text_bs):
        current_idx = min(num_text, i + text_bs)
        print(f"Processing text {current_idx} / {num_text}", end='\r' if i + text_bs < num_text else "\n")
        
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))

        text_embeds_list.append(text_embed.cpu())
        text_feats_list.append(text_feat.cpu())
        text_atts_list.append(text_input.attention_mask.cpu())
        del text_embed; del text_feat; del text_input; del text

    text_embeds = torch.cat(text_embeds_list, dim=0).to(device, non_blocking=True); del text_embeds_list
    text_feats = torch.cat(text_feats_list, dim=0).to(device, non_blocking=True); del text_feats_list
    text_atts = torch.cat(text_atts_list, dim=0).to(device, non_blocking=True); del text_atts_list

    # extracting image data
    image_feats_list, image_embeds_list = [], []
    
    for i, (image, img_id) in enumerate(data_loader):
        print(f"Processing image batch {i+1} / {len(data_loader)}", end='\r' if i != len(data_loader)-1 else "\n")
        
        image_feat = model.vision_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats_list.append(image_feat.cpu())
        image_embeds_list.append(image_embed.cpu())
        del image_embed; del image_feat; del image

    torch.cuda.empty_cache()
    image_feats = torch.cat(image_feats_list, dim=0).to(device, non_blocking=True); del image_feats_list
    image_embeds = torch.cat(image_embeds_list, dim=0); del image_embeds_list
    
    # compute the similarity matrix (pairwise dot product)
    sims_matrix = image_embeds.to(device) @ text_embeds.to(device).t()

    # configure the images to be processed on this gpu
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    
    for i, sims in enumerate(sims_matrix[start:end]):
        if debug_mode and i == 300: break
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(
            encoder_embeds=text_feats[topk_idx],
            attention_mask=text_atts[topk_idx],
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode='fusion'
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score
        del output; del encoder_att; del encoder_output
        
        # log progress
        if (i+1) % 50 == 0 or (i+1) == end-start:
            print(f"[Evaluation] Processing image {i+1}/{end-start} ({(i+1)/(end-start)*100:.2f}%)")


    torch.cuda.empty_cache()
    sims_matrix = sims_matrix.t()
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0), start + step)
    
    for i, sims in enumerate(sims_matrix[start:end]):
        if debug_mode and i == 300: break
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(
            encoder_embeds=text_feats[start + i].repeat(config['k_test'], 1, 1),
            attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode='fusion'
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score
        del output; del encoder_att; del encoder_output

        # log progress
        if (i+1) % 50 == 0 or (i+1) == end-start:
            print(f"[Evaluation] Processing text {i+1}/{end-start} ({(i+1)/(end-start)*100:.2f}%)")

    fabric.barrier()
    score_matrix_i2t = fabric.all_reduce(score_matrix_i2t, reduce_op="sum") 
    score_matrix_t2i = fabric.all_reduce(score_matrix_t2i, reduce_op="sum")        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

