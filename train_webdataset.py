import random
from data import TextField, WebPairedDataset, WebDictionaryDataset
from models import clip
from torch.optim.lr_scheduler import LambdaLR
import webdataset as wds
import evaluation
from models import Captioner
import torch
from torch.nn import NLLLoss
from torch.nn import functional as F
from torchvision.transforms import RandomCrop
import argparse, os
import numpy as np
import logging
import wandb
import collections
import deepspeed
from itertools import cycle, islice
from collections import defaultdict

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
_logger = logging.getLogger('train')
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


class Trainer(object):
    def __init__(self, args, model_engine):
        self.args = args
        self.model_engine = model_engine
        self.start_epoch = 0
        self.best_cider = defaultdict(float)
        self.patience = 0
        self.step = 0
        if args.resume_last:
            self.resume_last()

    def load_state(self, ckpt_id):
        _logger.info("Resuming checkpoint %d" % ckpt_id)
        checkpoint_dir = self.args.checkpoint_dir + '/' + self.args.exp_name
        self.model_engine.load_checkpoint(checkpoint_dir, ckpt_id, load_module_strict=False)

    def load_module(self, ckpt_id):
        _logger.info("Resuming module weights from checkpoint %d" % ckpt_id)
        checkpoint_dir = self.args.checkpoint_dir + '/' + self.args.exp_name_resume
        self.model_engine.load_checkpoint(checkpoint_dir, ckpt_id, load_module_strict=False, load_optimizer_states=False, load_lr_scheduler_states=False)

    def resume_last(self):
        checkpoint_dir = self.args.checkpoint_dir + '/' + self.args.exp_name
        ckpt_id = max([int(c) for c in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, c))])
        self.load_state(ckpt_id)
        self.step = ckpt_id 
        random.seed(ckpt_id)
        torch.manual_seed(ckpt_id)
        np.random.seed(ckpt_id)

    def save_state(self):
        checkpoint_dir = self.args.checkpoint_dir + '/' + self.args.exp_name
        ckpt_id = self.step
        self.model_engine.save_checkpoint(checkpoint_dir, ckpt_id)


def evaluate_metrics(model: Captioner, text_field: TextField, step: int, model_engine, image_model, transform, trainer, args, batch_size=None):
    _logger.info("Evaluating metrics")
    if batch_size is None:
        batch_size = 5
    batch_size = max(1, batch_size)
    all_shards = {
        'coco-validation': ['coco-384-validation-dict-625-%03d.tar' % i for i in range(8)],
        'coco-test': ['coco-384-test-dict-625-%03d.tar' % i for i in range(8)]
    }
    cider = dict()

    for ds_name, shards in all_shards.items():
        gen = {}
        gts = {}
        shards = [args.webdataset_path + '/' + s for s in shards]
        dataset = WebDictionaryDataset(shards, transform, args, shuffle=False, split_by_worker=False).batched(batch_size)
        previous_model_state = model.training
        model.eval()

        with torch.no_grad():
            for it, (ds_idxs, _, images, caps_gt, tags) in enumerate(dataset):
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
                text, _ = model.beam_search(images, beam_size=5, out_size=1, tags=tags, sot_tokens=sot_tokens)

                caps_gen = text_field.decode(text)
                for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                    gen['%d_%d_%d' % (args.rank, it, i)] = [gen_i, ]
                    gts['%d_%d_%d' % (args.rank, it, i)] = gts_i
            
        model.train(previous_model_state)
        
        # Collect data from all nodes
        gts_all = [None for _ in range(args.world_size)]
        gen_all = [None for _ in range(args.world_size)]
        torch.distributed.barrier()
        torch.distributed.all_gather_object(gts_all, gts)
        torch.distributed.all_gather_object(gen_all, gen)
        gts = dict(collections.ChainMap(*gts_all))
        gen = dict(collections.ChainMap(*gen_all))        
        
        if args.rank == 0:
            # Run evaluation
            gts = evaluation.PTBTokenizer.tokenize(gts)
            gen = evaluation.PTBTokenizer.tokenize(gen)
            scores, _ = evaluation.compute_scores(gts, gen)

            _logger.info("Evaluation scores on %s: %s" % (ds_name, scores))
            cider[ds_name] = scores['CIDEr']
            if args.logging:
                wandb.log({"%s_cider" % ds_name: scores['CIDEr'], "%s_bleu1" % ds_name: scores['BLEU'][0], "%s_bleu4" % ds_name: scores['BLEU'][3],
                    "%s_meteor" % ds_name: scores['METEOR'], "%s_rouge" % ds_name: scores['ROUGE'], "step": step})
            
    if args.rank == 0:
        if cider['coco-validation'] > trainer.best_cider['coco-validation']:
            trainer.best_cider['coco-validation'] = cider['coco-validation']
            trainer.best_cider['coco-test'] = cider['coco-validation']
            if args.logging:
                wandb.run.summary["best_coco_val_cider"] = cider['coco-validation']
                wandb.run.summary["best_coco_test_cider"] = cider['coco-test']
            trainer.patience = 0
        else:
            trainer.patience += 1


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


def init_distributed(args):
    # Distributed initialization
    _logger.info("HOST: %s, MASTER_ADDR: %s, MASTER_PORT: %s, RANK: %d, LOCAL_RANK: %d, WORLD_SIZE: %d" % (os.uname()[1], os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'], int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])))
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        torch.cuda.set_device('cuda:0')
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
        _logger.info('Process %d, total %d, running on node %s, CUDA_VISIBLE_DEVICES=%s.'
                     % (args.rank, args.world_size, os.uname()[1], os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0


def init_logging(args):
    # Wandb logging
    if args.rank == 0 and args.logging:
        args.wandb_id = wandb.util.generate_id()
        wandb.init(entity='yourlab', project='universalcap', config=args, id=args.wandb_id)
        wandb.run.name = args.exp_name


if __name__ == '__main__':
    _logger.info('Universal Captioner Training')

    # Argument parsing
    parser = argparse.ArgumentParser(description='Universal Captioner')
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--image_model', type=str, default='clip_RN50x16')
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--random_crop', action='store_true')
    parser.add_argument('--input_tags', action='store_true')
    parser.add_argument('--use_ds_idx', action='store_true')
    parser.add_argument('--n_tags', type=int, default=5)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--with_pe', action='store_true')
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--lr_multiplier', type=float, default=1)
    parser.add_argument('--shards_file', type=str, default='coco-cc3m-cc12m-yfcc-wit.shards')
    parser.add_argument('--coco_balancing_factor', type=float, default=-1)
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
    if args.random_crop:
        transform.transforms[1] = RandomCrop(transform.transforms[1].size)
    image_model = clip_model.visual
    image_model.to(device)
    image_model.eval()
    args.image_dim = image_model.embed_dim
    model = Captioner(args, text_field).to(device)

    def lr_scheduler(optim):
        def lambda_lr(s):
            warm_up = args.warmup
            s += 1
            return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5) * args.lr_multiplier
        return LambdaLR(optim, lambda_lr)

    model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                        model=model,
                                                        model_parameters=[p for p in model.parameters() if p.requires_grad],
                                                        lr_scheduler=lr_scheduler)

    if args.logging and args.rank == 0:
        wandb.watch(model, log=None, log_freq=1)
    trainer = Trainer(args, model_engine)

    # Create the dataset and samplers
    shards = open(args.shards_file).readlines()
    shards = [args.webdataset_path + '/' + s.strip() for s in shards]
    if args.coco_balancing_factor >= 0:
        coco_shards = [s for s in shards if 'coco' in s]
        not_coco_shards = [s for s in shards if 'coco' not in s]
        desired_len = int(len(shards) * args.coco_balancing_factor)
        print("Adjusting (eventually repeating) COCO to have %d samples, so to reach a balancing factor of %.02f" % (desired_len, args.coco_balancing_factor))
        coco_shards = list(islice(cycle(coco_shards), desired_len))
        shards = coco_shards + not_coco_shards

    random.shuffle(shards)
    dataset = WebPairedDataset(shards, transform, args).shuffle(1000).repeat().batched(args.batch_size)
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=args.workers, pin_memory=True)

    # Main loop
    _logger.info("Training starts")

    train_xe(model_engine, dataloader, text_field, trainer, args)
