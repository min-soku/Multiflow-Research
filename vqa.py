import argparse
import os
import math
import ruamel.yaml as yaml
import time
import datetime
import json

import torch
import torch.backends.cudnn as cudnn


from models import BLIPVQA, XVLMVQA, BertTokenizerForXVLM

import utils
from utils.loggers import init_wandb_logger
from utils.optim import create_optimizer, create_scheduler
from utils.prune_utils import make_prunable, named_masked_parameters, stats
from utils.misc import millions
from utils.functions import get_unprunable_parameters

from datasets import create_dataset, create_sampler, create_loader
from datasets.vqa_dataset import vqa_collate_fn

from evaltools import blip_vqa_evaluation, xvlm_vqa_evaluation

import lightning as L



def train(model, data_loader, optimizer, tokenizer, epoch, scheduler, fabric: L.Fabric, config, debug_mode=False):
    print("[Debug] vqa.py : train() 함수 호출")
    model.train() # 모델을 train모드로 변경
    optimizer.zero_grad() # 가중치 초기화
    
    losses, steps = [], 0 # step별 loss 저장할 list, train step 카운터 선언
    for i, (image, question, answer, weights, n) in enumerate(data_loader): # dataloader가 Batch별로 넘겨주며 반복문 실행(vqa_collate_fn()실행)
        if debug_mode:
            print(f"[Debug] Batch {i}/{len(data_loader)}")
            print("[Debug] image size:", image.size())  # e.g. [batch_size, C, H, W]
            print("[Debug] question (first 2):", question[:2])
            print("[Debug] answer (first 2):", answer[:2])
            print("[Debug] weights (first 10):", weights[:10])
            print("[Debug] n (first 10):", n[:10])

        if debug_mode and steps == 10: break
            

        # define the gradient accumulator context manager
        # grad_acc_steps의 배수가 아니거나 마지막 batch가 아니면 동기화 X(True), grad_acc_steps이거나 마지막 batch인 경우 동기화 진행(False)
        is_accumulating = not (((i + 1) % config['grad_acc_steps'] == 0) or ((i + 1) == len(data_loader)))
        with fabric.no_backward_sync(model, enabled=is_accumulating): # is_accumulating이 True인 경우 동기화 X -> forwarding을 통해 loss만 구함
            # tokenize both question and answer
            question_input = tokenizer( # 질문 리스트 Tokenizing, longest한 길이에 맞춰 padding 추가하고, max_length보다 길면 자름
                question, padding='longest', truncation=True, max_length=config['max_tokens'], return_tensors="pt"
            ).to(fabric.device)

            if debug_mode:
                # question는 어차피 list[str] (?)
                print("[Debug] question sample:", question[:2])
                question_input_debug = tokenizer(question[:2], padding='longest', truncation=True, max_length=config['max_tokens'], return_tensors="pt")
                print("[Debug] question_input_debug['input_ids'] shape:", question_input_debug["input_ids"].shape)
                print("[Debug] question_input_debug tokens:", question_input_debug["input_ids"])
                # 실제 batch 토큰 shape
                print("[Debug] question_input['input_ids'].shape:", question_input["input_ids"].shape)
                print("[Debug] question_input['attention_mask'].shape:", question_input["attention_mask"].shape)

            answer_input = tokenizer( # 정답 리스트 Tokenizing
                answer, padding='longest', return_tensors="pt"
            ).to(fabric.device)

            if debug_mode:
                print("[Debug] answer (first 2):", answer[:2])
                print("[Debug] answer_input['input_ids'].shape:", answer_input["input_ids"].shape)
                print("[Debug] answer_input['attention_mask'].shape:", answer_input["attention_mask"].shape)
            # forward the model and scale the loss according to grad acc steps
            # Tokenizing되어 전처리된 질문, 정답 입력값은 model의 forward연산에 사용됨
            loss = model(image, question_input, answer_input, train=True, k=n, weights=weights) # blip_vqa.py의 forward()함수 호출
            loss = loss / config['grad_acc_steps']
            fabric.backward(loss)
        
        # perform the update 
        if not is_accumulating: # is_accumulating이 False인 경우, 모든 프로세스의 gradient를 동기화하고 optimize함
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
            # some cool stuff
            steps += 1
            loss_scalar = loss.item()
            losses.append(loss_scalar)
            print(f"[Epoch {epoch+1}]\tBatch = {i+1}/{len(data_loader)}\t loss = {loss_scalar:.4f}")

            # some more cool stuff
            fabric.log_dict({"loss": loss_scalar, "lr": optimizer.param_groups[0]["lr"], "epoch": epoch})
    
    return {"loss_epoch_level": torch.tensor(losses).mean().item()}


def merge_eval_from_ranks(output_dir, epoch, world_size, split):
    assert split in ("val", "test")
    
    result = []
    for r in range(world_size):
        path_for_this_rank = os.path.join(output_dir, f"vqa_{split}_{r}.json")
        with open(path_for_this_rank, "r") as f:
            r_result = json.load(f)
            result += r_result
    
    result_path = os.path.join(output_dir, f"{split}_vqa_result_epoch{epoch}.json")
    with open(result_path, "w") as f:
        json.dump(result, f)
    return result_path


def cleanup_eval(output_dir, world_size, split):
    for r in range(world_size):
        path_for_this_rank = os.path.join(output_dir, f"vqa_{split}_{r}.json")
        os.remove(path_for_this_rank)



def main(args, config):
    
    if "32" in args.precision:
        torch.set_float32_matmul_precision(precision="high")
    elif "16" in args.precision:
        torch.set_float32_matmul_precision(precision="medium")

    # setup the experiment logger (wandb)
    loggers = []
    if args.wandb:
        loggers.append(init_wandb_logger(config))
    
    # initialize distributed mode with fabric
    # 분산 학습 설정
    fabric = L.Fabric(
        accelerator="cuda",
        devices=args.devices,
        strategy="ddp",
        precision=args.precision,
        loggers=loggers
    )
    fabric.launch()
    utils.setup_for_distributed(is_master=fabric.is_global_zero)

    world_size = fabric.world_size
    rank = fabric.global_rank

    # define gradient accumulation steps
    # step에 따라 처리할 batch 크기 설정
    E = config['batch_size_target']
    B = config['batch_size_train']
    config['grad_acc_steps'] = (E // world_size) // B # 몇 번의 batch마다 optimize할 지

    # reproducibility settings
    L.seed_everything(args.seed)
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    # create the datasets
    # 데이터셋 불러옴
    print("Combining VQA2.0 [Train and Val] and VisualGenome QA Datasets.")
    print("[Debug] vqa.py : Dataset 불러옴 -> vqa_dataset.py의 init()함수 호출")
    train_dataset, val_dataset, vqa_test_dataset = create_dataset('vqa', config)
    
    # create the dataloaders, samplers and set them up
    # train/val/test 데이터셋에 대해 샘플링할 수 있도록 sampler를 생성함(batch를 만들기 위해 어떤 순서로 만들지 전략을 세우는 것)
    (train_sampler, val_sampler, test_sampler) = create_sampler(
        (train_dataset, val_dataset, vqa_test_dataset), 
        shuffles=[True, False, False], 
        num_replicas=world_size, 
        global_rank=rank, 
        is_eval=[False, True, True]
    )
    
    # Sampler와 dataset을 이용하여 Dataloader를 생성함(batch를 실제로 만들어 반환함)
    train_loader, val_loader, test_loader = create_loader(
        (train_dataset, val_dataset, vqa_test_dataset), 
        (train_sampler, val_sampler, test_sampler),
        batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
        num_workers=[8, 8, 8], 
        is_trains=[True, False, False],
        collate_fns=[vqa_collate_fn, None, None] #train loader는 vqa_collate_fn 함수 사용 -> vqa_dataset.py에 있음
    )
    # 생성된 Dataloader들을 분산 학습 환경에 맞게 설정함
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader, use_distributed_sampler=False)

    # variable initialization according to the model
    if args.model == 'xvlm':
        # model
        model = XVLMVQA(config=config)
        setattr(model, "name", "xvlm")
        
        # tokenizer
        tokenizer = BertTokenizerForXVLM.from_pretrained(config['text_encoder'])
        
        # evaluation fn
        evaluation = xvlm_vqa_evaluation
    
    elif args.model == 'blip':
        # model
        print("[Debug] vqa.py : BLIP 모델 객체 생성 -> blip_vqa.py로 이동")
        model = BLIPVQA( 
            image_size=config['image_res'], 
            vit=config['vit'], 
            vit_grad_ckpt=config['vit_grad_ckpt'], 
            vit_ckpt_layer=config['vit_ckpt_layer']
        )
        setattr(model, "name", "blip")

        # tokenizer
        # BLIP Tokenizer 설정 
        tokenizer = model.tokenizer

        # evaluation fn
        #training 후 evaluation할 함수 지정 -> evaltools/itr_utils.py의 blip_evaluation 함수
        evaluation = blip_vqa_evaluation
    else:
        raise NotImplementedError(f"Model {args.model} not supported.")

    # grab the configs for the special tokens from the dataset and pass them to the model
    # Dataset에 저장된 pad, eos 토큰 정보를 모델 config에 반영 -> 모델이 패딩 처리에 필요한 토큰 정보를 알 수 있게 하기 위함
    print(f"PAD token ID = {vqa_test_dataset.pad_token_id}; EOS token = {vqa_test_dataset.eos_token}")
    config['pad_token_id'] = vqa_test_dataset.pad_token_id
    config['eos'] = vqa_test_dataset.eos_token
    
    # apply the pruning mask to the model
    if not args.dense: # pruned한 모델인 경우
        print("[Debug] vqa.py : pruned model")
        #utils.prune_utils의 make_prunable 함수 호출 : 모델 파라미터에 pruning mask를 적용할 수 있도록 함
        make_prunable(model, pattern_lock=True, mask_on_the_fly=True) 
        # blip_vqa.py : load_from_pruned_pretrained() 함수 호출 : pruning mask를 파라미터에 적용
        model.load_from_pruned_pretrained(args.pretraining_weights, args.mask, config)

        # default behaviour is, since the answer decoder inherits weights from the multimodal decoder, 
        # also inherit the mask; this if-block allows to override this behaviour
        if args.model == 'xvlm':
            if not args.inherit_mask:
                print("Not inheriting mask from pretrained model.")
                dense_masks = {}
                for name, param in model.named_buffers():
                    if "pruning_mask" in name and name.startswith("text_decoder"):
                        dense_masks[name] = torch.ones_like(param)
                model.load_state_dict(dense_masks, strict=False)
            else:
                print("Inheriting mask from pretrained model.")
    else:
        print("[Debug] vqa.py : dense model")
        # blip_vqa.py : load_pretrained() 함수 호출
        model.load_pretrained(args.pretraining_weights, config, is_eval=False)
        #utils.prune_utils의 make_prunable 함수 호출
        make_prunable(model, pattern_lock=False, mask_on_the_fly=False)
    
    # optimizer initialization
    print("[Debug] vqa.py : optimizer 생성 -> vqa.yaml에서 optimizer 설정 가져옴")
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model) # utils/optim.py의 create_optimizer()함수 호출

    # setup everything with fabric
    # model, optimizer 분산학습 설정
    model, optimizer = fabric.setup(model, optimizer)

    # training epoch, 시작 시간, 스케줄러 초기화
    start_epoch = 0
    start_time = time.time()
    sched_state_dict = None
    
    # resume from a snapshot if existing
    if os.path.exists(args.snapshot):
        print("[Debug] vqa.py : snapshot 복원 -> 마지막 지점의 training 정보 가져옴 ")
        restored_state = fabric.load(args.snapshot)
        # NOTE: the model state will contain both the weights and the masks
        model.load_state_dict(restored_state['model'])
        optimizer.load_state_dict(restored_state['optimizer'])
        sched_state_dict = restored_state['scheduler']
        start_epoch = restored_state['last_epoch'] + 1
        start_time = time.time() - restored_state['elapsed_time']
        print(f"Loaded state, resuming from epoch = {start_epoch}")
        print(
            "IMPORTANT: You are resuming training from a snapshot.\n"
            "As per the README.md, note that while the code for resuming is given, it has not been tested.\n"
            "The authors are not responsible for any issues that may arise from resuming training.\n\n"
        )

    # learning rate scheduler initialization
    print("[Debug] vqa.py : scheduler 생성 -> vqa.yaml에서 scheduler 설정 가져옴")
    arg_sche = utils.AttrDict(config['scheduler'])
    steps_per_epoch = math.ceil(len(train_dataset) / config['batch_size_target']) # epoch 당 step수 계산
    num_training_steps = steps_per_epoch * arg_sche['epochs'] # 전체 학습 step수 계산 (전체 epoch x epoch당 step수)
    num_warmup_steps = int(num_training_steps * arg_sche['num_warmup_steps']) # 워밍업 step수 계산
    lr_scheduler = create_scheduler( # 스케줄러 생성 utils/optim.py의 create_scheduler() 함수 호출
        mode=arg_sche['sched'], 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        total_steps=num_training_steps, 
        last_epoch=-1 if sched_state_dict is None else sched_state_dict['last_epoch']-1 
    )
    if sched_state_dict is not None: # snapshot 존재할 경우 스케줄러 상태 복원
        lr_scheduler.load_state_dict(sched_state_dict)

    # also log some stats regarding the pruned parameters of the answer decoder
    remaining_params, total_params = stats(named_masked_parameters(model, exclude=get_unprunable_parameters(model.name)))
    print(f"Remaining Params (M) = {millions(remaining_params, decimals=2)} / {millions(total_params, decimals=2)}", end=" ")
    print(f"({100*remaining_params/total_params:.2f}%)")
    
    print("Start training")    
    fabric.barrier() # 분산 학습에서 모든 프로세스가 이 지점까지 작업을 마칠 때까지 대기 -> Training 동기화를 위해
    max_epoch = config['scheduler']['epochs']
    for epoch in range(start_epoch, max_epoch):

        # go with a training epoch!
        train_loader.sampler.set_epoch(epoch) # 분산 학습에서 여러 프로세스가 한 데이터셋을 Batch로 나눠서 뽑아갈 때, Batch 내 데이터르르 중복 없이 고르게 분배하도록 함
        train(model, train_loader, optimizer, tokenizer, epoch, lr_scheduler, fabric, config, debug_mode=args.debug) # Training 시작 train()함수 호출

        # NOTE: in order to meet the same training scheme as the dense model, I have not performed any form of best model
        # selection or early stopping on the validation data, since these procedures are not employed in the original code of BLIP and XVLM;
        # The results of the paper are computed on the test-dev partition after 10 epochs of training.

        # checkpoint the model after each epoch (fabric dumps only on rank 0 by default)
        # 체크포인트 저장
        state = {"model": model, "optimizer": optimizer, "scheduler": lr_scheduler.state_dict(), 
                "last_epoch": epoch, "elapsed_time": time.time() - start_time}
        fabric.save(path=args.snapshot, state=state)
        print(f"[Epoch {epoch+1}/{max_epoch}] Dumped snapshot at {args.snapshot}")
        
        # wait for all models to sync before starting the next epoch
        fabric.barrier() # Epoch 종료마다 모든 프로세스를 다시 동기화시킴

    # after training, evaluate on the test-dev partition; the dumped file will be tested on the official evalai server
    print("Finished training. Starting evaluation on the test-dev partition.")
    # Training이 모두 끝난 뒤 모델의 성능을 평가함 -> evaltools/itr_utils.py의 blip_evaluation 함수
    evaluation(args.result_dir, model, test_loader, tokenizer, fabric, config, split="test", debug_mode=args.debug)
    fabric.barrier() 

    # when each device has dumped its answer file, merge them
    # to avoid strange behaviors, only do this on the main rank and sync at the end
    # 각 프로세스가 만든 결과를 병합함
    if fabric.is_global_zero: # 오직 하나의 프로세스만 병합하도록 수행(충돌 방지)
        result_filepath = merge_eval_from_ranks(args.result_dir, epoch="_last", world_size=world_size, split="test")
        print(f"Dumped test result file at {result_filepath}")

        # ... and avoid leaving unnecessary files on disk
        # 병합 후, 각 프로세스가 갖고 있는 결과 지움(하나의 병합된 결과만 있으면 됨)
        cleanup_eval(args.result_dir, world_size, split="test")

    # dump total training time when everything finished
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total Time for VQA Finetuning {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['xvlm', 'blip'])
    parser.add_argument('-pre', '--pretraining_weights', required=True)
    parser.add_argument('-m', '--mask', type=str, required=False, 
                        help="Path to the pruning mask. If not provided, ensure to pass the --dense flag.")
    parser.add_argument('--dense', action="store_true", default=False, 
                        help="Set this flag to train the dense model. Overrides mask settings.")
    parser.add_argument('--inherit_mask', action="store_true", default=False, 
                        help="XVLM Only. Set this flag to inherit the pruning mask from the pretrained encoder. "
                        "Please see the XVLM Paper, Section 4.3 for details. In a nutshell, a 6-layer transformer encoder "
                        "is introduced to answer questions, and the basic setup within the paper is to initialize it with the weights of the "
                        "fusion encoder. This flag also allows to inherit the pruning mask of the fusion encoder to the answer encoder.")
    parser.add_argument('--config', required=True, 
                        help="Path to the .yaml configuration file for this script. For convenience, you can use"
                        " configs/xvlm/vqa.yaml and configs/blip/vqa.yaml.")
    parser.add_argument('--snapshot', type=str, required=False, default="vqa_snapshot.pt", 
                        help="Path to the snapshot to load/save. If not provided, the default is vqa_snapshot.pt. "
                        "If provided, the script will resume from the snapshot. If not, the script will start from scratch. "
                        "Please note that the code for resuming is given, but has not been tested, nor used for the experiments in the paper.")
    parser.add_argument('--output_dir', default='output/vqa', 
                        help="Path to the output directory of the script. This includes the config file, as well as the generated answer files at each epoch.")
    parser.add_argument('-d', '--devices', type=int, default=1, 
                        help="Number of devices (i.e, gpus) to use with Lightning Fabric and DDP. Default is 1.")
    parser.add_argument('-p', '--precision', type=str, default='bf16-mixed', choices=['32-true', '16-mixed', 'bf16-mixed'],
                        help="Precision strategy for VQA finetuning. Default is bf16-mixed.")
    parser.add_argument('--seed', default=42, type=int, 
                        help="Seed for reproducibility. Default is 42.")
    parser.add_argument('-wdb', '--wandb', action="store_true", default=False, 
                        help="Set this flag to log data on Weights & Biases. Please remember to login with `wandb login` before running the script.")
    parser.add_argument('-exp', '--experiment_name', default=None, required=False, 
                        help="Experiment name for W&B. If passed, will override the experiment name set in the config.")
    parser.add_argument('--wdb_offline', action='store_true', default=False, required=False, 
                        help='Whether or not to log data on wandb. Remember to run wandb sync at the end of the training if this flag is active.')
    parser.add_argument('--debug', action="store_true", default=False, help="Set this flag to run in debug mode. \
                        Only processes a few train batches per epoch. Remember to cleanup later.")

    args = parser.parse_args()

    # load the config (default is: configs/vqa.yaml)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # merge the config and the args
    config.update(vars(args))

    # generate needed folders if necessary and dump needed files
    args.result_dir = os.path.join(args.output_dir, 'result')
    os.makedirs(args.result_dir, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)

