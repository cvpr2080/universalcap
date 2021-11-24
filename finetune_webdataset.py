import random
from data import TextField, WebPairedDataset, WebDictionaryDataset, WebDictionaryText
from models import clip
from evaluation import PTBTokenizer, Cider
from train_webdataset import init_distributed, init_logging, Trainer, evaluate_metrics
from torch.nn import NLLLoss
from torch.nn import functional as F
from torch.optim import Adam
import webdataset as wds
import evaluation
from models import Captioner
import torch
from utils import exclusive
import argparse, os
import numpy as np
import logging
import wandb
import deepspeed
import multiprocessing
from itertools import chain

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
_logger = logging.getLogger('train')
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


def train_scst(model_engine: deepspeed.DeepSpeedEngine, dataloader, text_field: TextField, trainer: Trainer, cider, args):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool(4)
    running_losses = []
    running_rewards = []
    beam_size = 5
    model_engine.train()

    for ds_idxs, _, images, captions, tags in dataloader:
        if args.input_tags:
            tags = text_field.process_pad_tags(tags, args, with_bos=True)
            tags = tags.to(device)
        else:
            tags = None

        ds_idxs = torch.tensor([text_field._tokenizer.encoder[str(x)+'</w>'] for x in ds_idxs.tolist()])
        ds_idxs, images = ds_idxs.to(device), images.to(device)
        with torch.no_grad():
            images = image_model.intermediate_features(images)
            if not model_engine.fp16_enabled():
                images = images.float()

        sot_tokens = ds_idxs if args.use_ds_idx else None
        outs, log_probs = model_engine(images, beam_size=beam_size, out_size=beam_size, tags=tags, sot_tokens=sot_tokens)
        outs = outs.view(-1, outs.shape[-1])

        # Rewards
        caps_gen = text_field.decode(outs)

        caps_gt = list(chain(*([c, ] * beam_size for c in captions)))
        caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
        reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
        reward = torch.from_numpy(reward).to(device).view(images.shape[0], beam_size)
        reward_baseline = torch.mean(reward, -1, keepdim=True)
        loss = -(torch.sum(log_probs, -1) / torch.sum(log_probs != 0, -1)) * (reward - reward_baseline)
        loss = loss.mean()

        model_engine.backward(loss)
        model_engine.step()

        # Reduce across workers
        reduced_loss = loss.data.clone()
        torch.distributed.all_reduce(reduced_loss)
        running_losses.append(reduced_loss.item() / args.world_size)
        running_loss = sum(running_losses[-10:]) / len(running_losses[-10:])

        reduced_reward = reward.mean().data.clone()
        torch.distributed.all_reduce(reduced_reward)
        running_rewards.append(reduced_reward.item() / args.world_size)
        running_reward = sum(running_rewards[-10:]) / len(running_rewards[-10:])

        if trainer.step % 10 == 0 and args.rank == 0:
            _logger.info('Loss: %.04f / Reward: %.04f' % (running_loss, running_reward))

        if trainer.step > 0 and trainer.step % args.save_interval == 0:
            trainer.save_state()

        if trainer.step > 0 and trainer.step % args.eval_interval == 0:
            evaluate_metrics(model_engine.module, text_field, trainer.step, model_engine, image_model, transform, trainer, args)

        if args.logging and args.rank == 0:
            wandb.log({"loss": reduced_loss.item() / args.world_size, "reward": reduced_reward.item() / args.world_size,
                       "lr": model_engine.get_lr()[0], "step": trainer.step})

        trainer.step += 1

    tokenizer_pool.close()
    return None


def train_xe(model_engine: deepspeed.DeepSpeedEngine, dataloader, text_field: TextField, trainer: Trainer, args):
    # Training with cross-entropy
    running_losses = []
    loss_fn = NLLLoss(ignore_index=0)
    model_engine.train()

    for ds_ids, images, captions, tags in dataloader:
        if not args.use_ds_idx:
            captions = text_field.process(captions)
        else:
            captions = text_field.process(captions, sot_tokens=ds_ids)

        tags = text_field.process_pad_tags(tags, args, with_bos=True)            
        images, captions, tags = images.to(device), captions.to(device), tags.to(device)
        with torch.no_grad():
            images = image_model.intermediate_features(images).detach()
            if not model_engine.fp16_enabled():
                images = images.float()

        if args.input_tags:
            model_input = torch.cat([tags, captions], 1)
            offset = tags.shape[1]
            skip_idx = None
        else:
            model_input = captions
            offset = 0
            skip_idx = None

        logits = model_engine(images, model_input)

        # XE loss
        gt_output = model_input[:, offset+1:]
        pred_output = logits[:, offset:-1]
        if skip_idx is not None:
            gt_output = torch.cat([gt_output[:, :skip_idx-1], gt_output[:, skip_idx:]], dim=1)
            pred_output = torch.cat([pred_output[:, :skip_idx-1], pred_output[:, skip_idx:]], dim=1)
        gt_output = gt_output[:, :pred_output.shape[1]].contiguous()
        pred_output = pred_output.contiguous()
        loss = loss_fn(F.log_softmax(pred_output, dim=-1).view(-1, text_field._tokenizer.vocab_size), gt_output.view(-1))

        model_engine.backward(loss)
        model_engine.step()

        # Reduce across workers
        reduced_loss = loss.data.clone()
        torch.distributed.all_reduce(reduced_loss)
        running_losses.append(reduced_loss.item() / args.world_size)
        running_loss = sum(running_losses[-10:]) / len(running_losses[-10:])

        if trainer.step % 10 == 0 and args.rank == 0:
            _logger.info('Loss: %.04f' % running_loss)

        if trainer.step > 0 and trainer.step % args.save_interval == 0:
            trainer.save_state()

        if trainer.step > 0 and trainer.step % args.eval_interval == 0:
            evaluate_metrics(model_engine.module, text_field, trainer.step, model_engine, image_model, transform, trainer, args)

        if args.logging and args.rank == 0:
            wandb.log({"loss": reduced_loss.item() / args.world_size, "lr": model_engine.get_lr()[0], "step": trainer.step})

        trainer.step += 1

    return None


if __name__ == '__main__':
    _logger.info('Universal Captioner Finetuning')

    # Argument parsing
    parser = argparse.ArgumentParser(description='Universal Captioner')
    parser.add_argument('exp_name', type=str)
    parser.add_argument('exp_name_resume', type=str)
    parser.add_argument('checkpoint_id', type=int)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--image_model', type=str, default='clip_RN50x16')
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--input_tags', action='store_true')
    parser.add_argument('--use_ds_idx', action='store_true')
    parser.add_argument('--n_tags', type=int, default=5)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--with_pe', action='store_true')
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--shards_file', type=str, default='coco-training-dict.shards')
    parser.add_argument('--scst', action='store_true')
    parser.add_argument('--use_gpt_init', action='store_true')
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    _logger.info(args)

    init_distributed(args)
    init_logging(args)

    args.webdataset_path = '<path_to_webdatasets>'
    args.checkpoint_dir = '<path_to_checkpoints>'
        
    # Pipeline for text
    text_field = TextField()

    # Models
    clip_model, transform = clip.load(args.image_model[5:], jit=False)
    image_model = clip_model.visual
    image_model.to(device)
    image_model.eval()
    args.image_dim = image_model.embed_dim
    model = Captioner(args, text_field).to(device)
    if args.scst:
        model.forward = model.beam_search

    optimizer = Adam(lr=args.lr, params=[p for p in model.parameters() if p.requires_grad])
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                        model=model,
                                                        optimizer=optimizer)

    if args.logging and args.rank == 0:
        wandb.watch(model, log=None, log_freq=1)
    trainer = Trainer(args, model_engine)
    if not args.resume_last:
        trainer.load_module(args.checkpoint_id)

    # Create the dataset and samplers
    shards = open(args.shards_file).readlines()
    shards = [args.webdataset_path + '/' + s.strip() for s in shards]
    random.shuffle(shards)

    if args.scst:
        dataset = WebDictionaryDataset(shards, transform, args).shuffle(1000).repeat().batched(args.batch_size)
        ref_caps_train = PTBTokenizer.tokenize(WebDictionaryText(shards))
        cider_train = Cider(ref_caps_train)
    else:
        dataset = WebPairedDataset(shards, transform, args).shuffle(1000).repeat().batched(args.batch_size)
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=args.workers, pin_memory=True)

    # Main loop
    _logger.info("Training starts")

    if args.scst:
        train_scst(model_engine, dataloader, text_field, trainer, cider_train, args)
    else:
        train_xe(model_engine, dataloader, text_field, trainer, args)
