# coding: utf8
from torch.utils.data.dataloader import default_collate
from itertools import takewhile
import torch
import random

from .tokenizer.simple_tokenizer import SimpleTokenizer as _Tokenizer


class RawField(object):
    """ Defines a general datatype.

    Every dataset consists of one or more types of data. For instance,
    a machine translation dataset contains paired examples of text, while
    an image captioning dataset contains images and texts.
    Each of these types of data is represented by a RawField object.
    An RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before creating an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object
            Default: None.
    """

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        """ Preprocess an example if the `preprocessing` Pipeline is provided. """
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        """ Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class TextField(RawField):
    def __init__(self):
        self._tokenizer = _Tokenizer()
        super(TextField, self).__init__()

    def preprocess(self, x):
        if x is None:
            return ''
        return x

    def process(self, texts, sot_tokens=None):
        if isinstance(texts, str):
            texts = [texts]

        if sot_tokens is None:
            sot_tokens = [self._tokenizer.bos_idx for _ in range(len(texts))]
        else:
            sot_tokens = [self._tokenizer.encoder[str(x)+'</w>'] for x in sot_tokens.tolist()]
        eot_token = self._tokenizer.eos_idx
        all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for sot_token, text in zip(sot_tokens, texts)]
        result = torch.zeros(len(all_tokens), max(len(s) for s in all_tokens), dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def process_tags(self, tags, with_bos=False):
        if isinstance(tags[0], str):
            tags = [tags]

        sot_token = self._tokenizer.bos_idx
        tags = [' '.join(t) for t in tags]

        if with_bos:
            all_tokens = [[sot_token] + self._tokenizer.encode(text) for text in tags]
        else:
            all_tokens = [self._tokenizer.encode(text) for text in tags]
        result = torch.zeros(len(all_tokens), max(len(s) for s in all_tokens), dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def process_pad_tags(self, tags, args, with_bos=False):
        tags = [t[:args.n_tags] for t in tags]
        [random.shuffle(t) for t in tags]
        tags = self.process_tags(tags, with_bos=with_bos)
        
        max_len = 2*args.n_tags
        if tags.shape[1] > max_len:
            tags = tags[:, :max_len]
        else:
            tags = torch.cat([tags, torch.zeros((tags.shape[0], max_len-tags.shape[1]), device=tags.device, dtype=tags.dtype)], dim=1)
        
        return tags

    def decode(self, word_idxs, exclude_before_bos=False):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ])[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ])[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0))[0]

        captions = []
        for wis in word_idxs:
            wis = wis.tolist()
            if exclude_before_bos and self._tokenizer.bos_idx in wis:
                wis = wis[wis.index(self._tokenizer.bos_idx)+1:]
            wis = list(takewhile(lambda tok: tok != self._tokenizer.eos_idx, wis))
            caption = self._tokenizer.decode(wis)
            captions.append(caption)
        return captions
