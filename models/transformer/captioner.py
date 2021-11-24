from data.field import TextField
from . import Encoder, Decoder, ScaledDotProductAttentionMemory
import torch
from torch import Tensor
from torch import nn
import math
from utils import TensorOrSequence
from models.containers import Module
from models.beam_search import *


class Captioner(Module):
    def __init__(self, args, text_field: TextField):
        super(Captioner, self).__init__()
        if args.input_tags:
            max_encoding_length = 80
            self.max_generation_length = 30
        else:
            max_encoding_length = 40
            self.max_generation_length = 30

        self.encoder = Encoder(args.N_enc, 500, args.image_dim, d_model=args.d_model, d_ff=args.d_ff, h=args.head,
                                        attention_module=ScaledDotProductAttentionMemory,
                                        attention_module_kwargs={'m': args.m},
                                        with_pe=args.with_pe)

        self.decoder = Decoder(text_field._tokenizer.vocab_size, max_encoding_length, args.N_dec, d_model=args.d_model, d_ff=args.d_ff, h=args.head, args=args)

        self.bos_idx = text_field._tokenizer.bos_idx
        self.eos_idx = text_field._tokenizer.eos_idx
        self.vocab_size = text_field._tokenizer.vocab_size
        self.args = args
        
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        self.decoder.train(mode)

    def init_weights(self):
        def init(n_layers):
            def fn(module):

                if self.args.use_gpt_init:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        nn.init.xavier_uniform_(module.weight)
                        if module.padding_idx is not None:
                            module.weight.data[module.padding_idx].zero_()
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)

                    # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
                    #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
                    #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
                    #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
                    #
                    # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
                    for name, p in module.named_parameters():
                        if "fc_o" in name and "weight" in name:
                            # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                            nn.init.xavier_uniform_(p, gain=1 / math.sqrt(2 * n_layers))
                else:
                    for name, p in module.named_parameters():
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p)
            return fn

        blocks = [self.encoder, self.decoder]
        for block in blocks:
            block.apply(init(len(block.layers)))


    def forward(self, images, seq, tags=None):
        enc_output, mask_enc = self.encoder(images)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t: int, prev_output: Tensor, visual: Tensor, **kwargs) -> Tensor:
        input = None
        if t == 0:
            self.enc_output, self.mask_enc = self.encoder(visual)

            if self.args.use_ds_idx:
                input = kwargs['sot_tokens'].reshape(-1, 1)
            else:
                input = visual.data.new_full((visual.shape[0], 1), self.bos_idx, dtype=torch.long)
            if self.args.input_tags:
                input = torch.cat([kwargs['tags'], input], 1)
        else:
            input = prev_output

        logits = self.decoder(input, self.enc_output, self.mask_enc)
        return logits[:, -1:]

    def beam_search(self, visual: TensorOrSequence, beam_size: int, out_size=1,
                    return_logits=False, **kwargs):
        bs = BeamSearch(self, self.max_generation_length, self.eos_idx, beam_size)
        return bs.apply(visual, out_size, return_logits, **kwargs)

    def grid_beam_search(self, visual: TensorOrSequence, constraints: torch.Tensor, 
                    num_constraints: int, beam_size: int, out_size=1, **kwargs):
        bs = GridBeamSearch(self, self.max_generation_length, self.eos_idx, beam_size, num_constraints)
        return bs.apply(visual, constraints, out_size, **kwargs)
