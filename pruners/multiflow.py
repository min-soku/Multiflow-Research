import time
import datetime
import torch
from functools import partial

from lightning.pytorch.utilities.combined_loader import CombinedLoader
from pruners.base import Pruner
from pruners.accumulators import forward_output, region_forward_output
from utils.prune_utils import (
    make_prunable, check_blip_state_dict, recursive_getattr
)
from utils.functions import detect_modality_fn



class MultiFlow(Pruner):
    #prune.py 125번째 줄에서 호출
    def __init__(self, model, *args, **kwargs): # Multuflow Pruner의 초기 설정(model정보 + pruning에 필요한 정보를 인자로 넘겨받음)
        print("<Multiflow.py -> Init() 함수>")
        #모델의 layer에 mask를 적용할 수 있는 구조로 설정
        #mask_dtype=torch.bool: 마스크 데이터 타입을 불리언으로 설정
        #pattern_lock=True: Pruning 패턴을 고정하여 일관된 마스크 패턴을 유지
        #mask_on_the_fly=True: 런타임 중 마스크 적용 가능
        #store_input=True: Layer의 입력 데이터를 저장하여 나중에 활성화 값을 기반으로 프루닝에 활용
        make_prunable(model, mask_dtype=torch.bool, pattern_lock=True, mask_on_the_fly=True, store_input=True)
        
        # Multiflow는 Pruner클래스를 상속받았기 때문에 부모 클래스의 초기화 작업을 호출하여 pruning알고리즘의 기본 설정 상속
        super(MultiFlow, self).__init__(model, *args, **kwargs) 
        
        # 3 functions the user must implement to use this pruner
        # partial 함수 : 특정 함수의 일부 인자를 fix하는 역할 -> 코드 재사용성 높임

        # 모달리티 감지 함수(모델의 이름에 따라 각 layer가 text, vision중 어디에 속하는지 감지)
        self.detect_modality_fn = partial(detect_modality_fn, self.model.name) 
        #모델의 출력값을 가져오는 함수 -> Pruning 과정에서 필요한 활성화값 저장 가능능
        self.forward_output = partial(forward_output, self.model.name)
        #영역 기반의 데이터가 포함된 경우(객체 검출), 출력값을 forward로 가져오는 함수 -> 선택적으로 사용용
        self.region_forward_output = partial(region_forward_output, self.model.name) # optional, only for combined loaders
        
        # default variables 
        # Pruning 전략 설정정
        self.scores_computed = False #Saliency 점수가 계산되었는지 여부 저장(초기값 false)
        self.is_one_shot = True #Pruning이 한 번에 이루어지는지 설정(True일 경우, 단일 단계 Pruning 후 종료)
        self.modifies_weights = False #Pruning 과정에서 가중치 수정 여부(multiflow는 가중치를 직접 수정하지 않으므로 false)
        self.name = 'multiflow' # Pruner 이름은 multiflow
        
        # variables for the algorithm (note that pruning is always done in fp32) Pruning 알고리즘 변수 초기화
        #활성화 값의 Norm을 저장하는 딕셔너리(초기에는 모든 값이 1, 활성화 값의 크기를 기반으로 Saliency 점수를 계산할 때 사용)
        self.actn_norms = {id(p): torch.ones(p.size()[1], dtype=torch.float32) for _, _, p in self.named_masked_parameters}
        #각 layer에서 수집한 샘플 수를 저장(초기 값은 0으로 설정)
        self.nsamples = {id(p): 0 for _, _, p in self.named_masked_parameters}
        #초기 가중치를 저장하여 나중에 모델을 초기 상태로 복원 가능(가중치를 clone하고 detach한 후, CPU로 이동하여 저장)
        #가중치를 수정하는 프루닝 기법이 아니라, 마스크를 적용하는 방식으로 프루닝을 수행 -> 가중치가 직접 수정되지 않고, 필요 시 초기 가중치로 복원할 수 있는 점이 특징
        self.init_weights = {id(p): torch.clone(p.data).detach().cpu() for _, _, p in self.named_masked_parameters}


    def _compute_abs_scores(self): # 모델 Parameter의 절댓값을 기반으로 Saliency 점수를 계산
        print("<Multiflow.py -> _compute_abs_scores 함수 호출 : 파라미터 절댓값 계산>")
        for _, _, param in self.named_masked_parameters: # Pruning이 가능한 이름, mask, parameter 튜플 리스트의 파라미터에 대해 반복 수행
            #param.data : 파라미터의 실제 데이터(가중치)
            #데이터 clone 후, 그래프와 연결을 끊어 역전파 시 영향을 받지 않도록 설정한 후, 절댓값 계산
            self.scores[id(param)] = torch.clone(param.data).detach().abs_() #파라미터의 고유 Id를 key로 하여 점수 저장하는 딕셔너리(tensor)

    def _modal_mask(self, target_sparsity): # 모달리티 별로 중요도 점수를 기준으로 target sparsity에 따라 중요도가 낮은 파라미터를 0으로 만들어 mask 업데이트
        print("<Multiflow.py -> modal_mask() 함수 호출 : masking>")
        #파라미터 n이 속한 모달리티를 탐지한다(Vision, Text, Fusion)
        different_modalities = set([self.detect_modality_fn(n) for n, _, _ in self.named_masked_parameters]) #set을 사용하여 모델에 존재하는 서로 다른 모달리티들을 중복 제거하여 수집
        for modality in different_modalities: # 모달리티 별 점수 수집

            # get the scores of the tensors of this modality
            #현재 모달리티에 속하는 파라미터들만 필터링하여 각 파라미터의 중요도 점수를 딕셔너리로 저장한다.
            scores_for_this_modality = {
                id(p): self.scores[id(p)] for n, _, p in self.named_masked_parameters 
                if self.detect_modality_fn(n) == modality #Init함수에서 선언한 모달리티 감지 함수
            }
            #현재 반복중인 모달리티와 일치하는는 모달리티의 각 파라미터 중요도 점수를 1차원 텐서로 병합한다.
            modal_scores = torch.cat([torch.flatten(v) for v in scores_for_this_modality.values()])

            # get the modality threshold
            k = int(modal_scores.numel() * target_sparsity) # 모달리티 점수 총 개수 * 목표 희소성 = Pruning할 파라미터 개수를 계산한다.
            threshold, _ = torch.kthvalue(modal_scores, k=k) # k번째로 작은 값을 기준으로 임계값 설정(임계값 이하 점수를 가진 파라미터는 pruning 대상)
            
            # compute the mask for the parameters of this modality
            for name, mask, param in self.named_masked_parameters:
                if self.detect_modality_fn(name) != modality: continue # 현재 반복 중인 파라미터가 현재 모달리티에 속하는지 확인
                score = self.scores[id(param)]# 파라미터 점수
                zero = torch.tensor([0], dtype=torch.bool).to(mask.device)#0
                one = torch.tensor([1], dtype=torch.bool).to(mask.device)#1
                #계산된 score값이 임계값보다 작으면 0, 그렇지 않으면 1로 업데이트 후, 계산된 마스크 값을 파라미터에 직접 복사하여 pruning 적용
                mask.copy_(torch.where(score.to(mask.device) <= threshold, zero, one))

    #모달리티 별로 target sparsity에 따른 pruning 분포를 계산한다.
    def multimodal_distribution(self, target_sparsity):
        print("<Multiflow.py -> multimodal_distribution() 함수 호출 : 모달리티별 pruning 분포 계산>")
        self._compute_abs_scores()#파라미터의 절댓값 기반 Saliency 점수 계산(self.scores에 저장됨)
        self._modal_mask(target_sparsity)#모달리티 별로 중요도 점수를 기준으로 pruning함

        distribution = {}
        # grab the sparsity distribution for each param, rewind the masks and the scores
        for _, mask, param in self.named_masked_parameters:
            #mask.sum().item() : 현재 파라미터에서 Pruning되지 않은 요소의 개수
            #mask.numel(): 총 파라미터 개수
            sparsity = 1 - mask.sum().item() / mask.numel()#파라미터 100개중 70개가 pruning되었으면 1 - 30% = 희소성은 70%

            distribution[id(param)] = sparsity #각 파라미터의 ID를 key로 하여 희소성값 저장
            #파라미터별로 희소성을 저장하는 이유는 파라미터(가중치)마다 크기나 중요도가 다를 수 있기 때문이다.
            mask.fill_(1)#모든 마스크를 1로 초기화 
            self.scores[id(param)] = torch.zeros_like(self.scores[id(param)])#self.scores를 0으로 초기화하여 새로운 pruning 작업에 대비
        
        return distribution #각 파라미터의 희소성 분포를 반환하여 pruning 후 분석에 사용(파라미터별로 sparsity저장한 딕셔너리)

    #각 파라미터에 대해 input과 output 뉴런의 saliency를 계산하여 Pruning 점수를 생성
    def score(self):
        print("<Multiflow.py -> score 함수 호출 : Saliency 계산>")
        for _, _, param in self.named_masked_parameters:
            # compute the importance of each input and output neuron
            #각 파라미터의 활성화 값의 norm을 저장하는 텐서(forward pass중에 수집된 입력 활성화 값의 크기를 반영)
            actn_norm = torch.sqrt(self.actn_norms[id(param)]).to(param.device)
            #각 출력 뉴런에 연결된 가중치의 절댓값에 활성화 norm을 곱한 후, 출력 뉴런 기준 평균 계산
            importance_per_output = (param.abs() * actn_norm).mean(dim=1)
            #각 입력 뉴런에 연결된 가중치의 절댓값에 활성화 norm을 곱한 후, 입력 뉴런 기준 평균 계산
            importance_per_input = (param.abs() * actn_norm).mean(dim=0)

            # make a cross product of the two importance vectors
            #입력과 출력 뉴런의 중요도의 외적 계산 => 가중치w_ij에 대해 입력 뉴런 i와 출력 뉴런 j(가중치 요소)의 중요도를 결합한 행렬이 생성된다
            score = torch.outer(importance_per_output, importance_per_input)
            
            # final score |Imp(input)| * |w_ij| * |Imp(output)|
            score = score * param.abs() #가중치 w_ij의 요소 간 외적 결과에 가중치의 절댓값을 곱하여 최종 중요도 점수를 계산한다.
            self.scores[id(param)] = torch.clone(score).detach().cpu()# 계산된 결과를 self.score에 clone하여 그래프 연결을 끊고, cpu로 이동시킨 후 딕셔너리에 저장
                        
    #입력 활성화 값을 관리하고 정규화(norm)을 계산하는 역할(모달리티 별로 처리)
    def _offload_actns(self, text_atts_history): #text_atts_history -> 리스트트
        print("<Multiflow.py -> _offload_actns 함수 호출 : 정규화 계산>")
        for name, _, param in self.named_masked_parameters:
            mname = ".".join(name.split(".")[:-1])#현재 파라미터가 속한 파라미터 이름에서 모듈 이름만 추출
            module = recursive_getattr(self.model, mname)#모델의 해달 모듈을 동적으로 참조한다
            modality = self.detect_modality_fn(name)#현재 파라미터가 어떤 모달리티에 속하는지 확인
            
            # if the current layer is a textual one, we must make sure not to include the [PAD] tokens 
            # in the computation of the score
            num_samples_to_add = 0
            if modality in ("text", "fusion"): # 현재 처리중인 layer가 text인지 fusion인지 확인하는 조건
                # 각 batch의 어텐션 마스크 리스트(의미 있는 토큰 : 1/ 패딩 값 : 0)와 모델이 입력으로 받은 토큰 임베딩을 저장한한 리스트의 길이가 같아야 한다
                # ex)text_atts_history=[어텐션1, 어텐션2, 어텐션3], module.input_history=[입력1, 입력2, 입력3]
                assert len(text_atts_history) == len(module.input_history), \
                    "The number of text attentions and the number of input histories must be the same." \
                    "Instead got {} attentions and {} input histories.".format(len(text_atts_history), len(module.input_history))
                
                #어텐션 마크스와 임력 임베딩을 한 쌍으로 묶어서 하나씩 처리한다
                for index_in_history, (text_att, input_sample) in enumerate(zip(text_atts_history, module.input_history)):
                    # text_att is a tensor of size [B, L] telling which tokens are relevant for each sample of the current batch
                    # input_sample is a tensor of size [B, L, embed_size] containing the embedding of each token of each batch
                    # our goal is to remove from the second dimension of :input_sample: the tokens that are not relevant
                    #어텐션 마스크의 Batch크기와 입력 임베딩의 Batch 크기가 같아야 한다
                    #ex)text_att=[[1,1,0],[1,0,0]] => 배치 크기 : 2 , 시퀀스 길이 : 3
                    #   input_sample=[ => 배치 크기 : 2, 시퀀스 길이 : 3
                    #                   [[0.1,0.3],[0.2,0.3],[0.0,0.0]],
                    #                   [[0.2,0.4],[0.0,0.0],[0.0,0.0]]]
                    assert text_att.size()[0] == input_sample.size()[0], \
                        f"The text attentions and the input history must have the same batch size. Instead got {text_att.size()[0]} and {input_sample.size()[0]}." \
                        f"Please check your implementation."
                    
                    num_samples_to_add += input_sample.size()[0]
                    
                    # in cross attention layers inputs will have a shape defined by the number of image patches, so we must
                    # make sure to avoid filtering out those
                    L_att = text_att.size()[-1] # 어텐션 마스크 시퀀스 길이
                    L_seq = input_sample.size()[1] # 임력 임베딩 시퀀스 길이
                    if L_att == L_seq:#시퀀스 길이가 일치할 때만 패딩값 제거
                        #text_att == 1인 토큰만 남기고, 0인 토큰은 제거거
                        input_sample = input_sample[text_att.to(input_sample.device).squeeze() == 1, :] 

                    # modify the input sample s.t. it has shape (L, embed_size), where L is now the number of relevant tokens after filtering
                    #기존 3차원의 input_sample텐서를 패딩 제거 후 남은 (유효한 토큰의 개수)와 (임베딩 크기)로 이루어진 2차원의 텐서로 변환한다
                    module.input_history[index_in_history] = input_sample.view(-1, input_sample.size()[-1])
                
            # if the current layer is a vision one, there is no concept of image attention and we simply
            # reshape each input sample to have shape (L, embed_size), where L is the number image patches
            elif modality == "vision":# 현재 파라미터가 VISION 모달리티일 때
                for index_in_history, input_sample in enumerate(module.input_history):
                    # each vision input sample will have shape [B, P, embed_size], where P is the number of patches
                    #B : 배치 크기(이미지 개수), P : 패치 수(이미지 1개를 몇 개의 패치로 분할했는지), embed_size : 임베딩 크기(각 패치의 몇 차원 벡터로 표현됐는지)
                    num_samples_to_add += input_sample.size()[0]
                    #input_sample의 모양을 BxP, embed_size형태로 변환 -> 이미지 패치를 일려로 나열된 벡터로 처리
                    #ex) (2,4,3): 2개 이미지를 각 4개의 패치로 분할함, 각 패치는 3차원 임베딩 -> (8,3) : 2개 이미지 x 4개의 패치, 각 패치는 3차원 임베딩딩
                    module.input_history[index_in_history] = input_sample.view(-1, input_sample.size()[-1])
            else:
                raise NotImplementedError("Modality {} not supported.".format(modality))

            # rescale the offloaded activations and update the number of samples to include in the reduction
            #활성화 norm 재조정 - 기존 샘플 수에 맞게 재조정한다다
            #self.actn_norms[id(param)] : 현재 활성화 norm / self.nsamples[id(param)] : 지금까지 처리한 샘플의 수 / num_samples_to_add : 이번에 새로 추가된 샘플의 수
            self.actn_norms[id(param)] *= self.nsamples[id(param)] / (self.nsamples[id(param)]+num_samples_to_add)
            #누적 샘플 수 업데이트 - 기존 샘플 수에 새로 처리한 샘플 수를 더한다
            self.nsamples[id(param)] += num_samples_to_add

            # update the running norm vector
            #여러 입력 데이터가 저장된 리스트를 하나의 큰 텐서로 병합하여 활성화 norm을 계산할 수 있게 한다
            #활성화 Norm이란 활성화 값의의 크기를 나타내는 개념(절댓값, 제곱합..등)
            X = torch.cat(module.input_history, dim=0).type(torch.float32)
            #torch.norm(X, p=2, dim=0) : L2 Norm을 사용하여 입력 데이터 X의 크기를 계산한다
            #/ self.nsamples[id(param)] : 총 샘플 수로 나누어 평균 활성화 norm을 계산한다
            #최종적으로 이전 활성화norm에 새로 계산된 norm을 더하여 최종 활성화 norm을 업데이트한다
            self.actn_norms[id(param)] = self.actn_norms[id(param)].to(X.device) + torch.norm(X, p=2, dim=0) ** 2  / self.nsamples[id(param)]

            # when activation norms are computed, release the input history
            del module.input_history; del X
            module.input_history = []


    def _flag_norms(self):
        print("<Multiflow.py -> _flag_norms 함수 호출>")
        # turn-off the flag for storing inputs for the layers that don't have to be pruned
        for module in self.model.modules():
            if hasattr(module, "store_input_flag"):
                module.store_input_flag = False

        for name, _, _ in self.named_masked_parameters:
            module_name = ".".join(name.split(".")[:-1])
            module = recursive_getattr(self.model, module_name)
            if hasattr(module, "store_input_flag"):
                module.store_input_flag = True


    @torch.no_grad()
    def prune(self, target_sparsity, model, dataloader, device, fabric, num_batches_per_step, **kwargs):
        # NOTE: fabric is kept here to share the signature with other pruners, but since there is no backward pass it is not used
        time_in = time.time()
        print("<Multiflow.py -> Prune() : pruning 시작 부분>")
        # detect if we also have a region-level dataloader
        is_combined = 'region_loader' in kwargs and kwargs['region_loader'] is not None
        if is_combined:# 입력데이터가 region_loader를 포함하면 dataloader와 묶어서 tuple형태로 변환됨
            dataloader = CombinedLoader((dataloader, kwargs['region_loader']), mode="min_size")

        # precompute the layer-wise pruning ratios (eq.7 of the paper)
        #각 레이어 내 모달리티 별 pruning 비율 계산
        distribution = self.multimodal_distribution(target_sparsity)
        
        # initialize a buffer for the binary indices of the tokens to attend (will be used to discard [PAD] tokens from the input history of each module) 
        text_atts_history = []

        # be sure to avoid storing inputs for the layers that don't have to be pruned (would consume useless memory)
        #Pruning 대상이 아닌 레이어 제외
        self._flag_norms()
        
        # **debugging
        print(f"DataLoader 크기: {len(dataloader)}")

        if len(dataloader) == 0:
            print("DataLoader가 비어 있음")
            exit()

        if is_combined and len(kwargs['region_loader']) == 0:
            print("region_loader가 비어 있음")
            exit()

        print("<multiflow.py : DataLoader가 정상적으로 로드>")
        #debugging

        # start iterating over the data
        #배치 단위로 처리
        print("<Dataloader를 배치 단위로 처리>")
        for batch_idx, batch in enumerate(dataloader):
            print(f"[{self.name.upper()}] Processing batch {batch_idx%num_batches_per_step+1}/{num_batches_per_step}", end='\r', flush=True)

            # unpack "out-of-domain" and "in-domain" batches (refer to the "4M pretraining set" of VLMs for these terms)
            if is_combined:
                general_batch, region_batch = batch
            else:
                general_batch = batch

            # when processing text, some of the tokens are padded in order to properly
            # collate the batch; therefore, in order not to bias the criterion towards [PAD] tokens,
            # we must keep track of their position here, and we will get rid of them before aggregating batches
            # in the 'score' function
            if hasattr(model, "is_vlm") and model.is_vlm:
                text_attentions = general_batch['attention_mask'] if 'attention_mask' in general_batch else general_batch[2].clone() 
                text_atts_history.append(text_attentions)
            
            # forward the model on the ood/general data, with no backward pass
            _ = self.forward_output(model, general_batch, device, modality="fusion")

            # also forward the model on the id data
            if is_combined:
                if hasattr(model, "is_vlm") and model.is_vlm:
                    text_attentions = region_batch['attention_mask'] if 'attention_mask' in region_batch else region_batch[2].clone() 
                    text_atts_history.append(text_attentions)
                _ = self.region_forward_output(model, region_batch, device)
            
            # compute the running aggregration of the norms of the activations and release the associated memory
            self._offload_actns(text_atts_history)
            del text_atts_history; text_atts_history = []

            # once all activations are computed, compute the scores and inject the pruning ratios in "compute_mask"
            is_end_step = (batch_idx+1) % num_batches_per_step == 0 or batch_idx == len(dataloader) - 1
            if is_end_step:
                # compute the information flow score (eq.6 of the paper)
                self.score()
                # compute the final mask using the information flow score and the layer-wise pruning ratios (eq.8 of the paper)
                self.compute_mask(distribution, scope="local")
                break
                
        self.scoring_time = int(time.time() - time_in)
        print(f"Total pruning time (hh:mm:ss) = {datetime.timedelta(seconds=self.scoring_time)}")


    def hard_reset(self):
        self.reset()

    def reset(self):
        print("<Reset 시작>")
        for _, mask, param in self.named_masked_parameters:
            mask.fill_(1)
            if mask.grad is not None:
                mask.grad.data.zero_()
            if param.grad is not None:
                param.grad.data.zero_()
            self.scores[id(param)] = torch.zeros(param.size(), dtype=torch.float32)
            self.nsamples[id(param)] = 0
            self.actn_norms[id(param)] = torch.zeros(param.size()[1], dtype=torch.float32)
        
        # re-tie the weights in the case of blip models
        if self.model.name == 'blip':
            check_blip_state_dict(self.model.state_dict())
