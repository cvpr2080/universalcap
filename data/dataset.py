import json
import webdataset as wds


def _identity(x):
    return x


def _dataset_id(x):
    if 'coco' in x.lower():
        return 1
    else:
        return 0


def WebPairedDataset(urls, image_transform, args, **kwargs) -> wds.WebDataset:
    ds = wds.WebDataset(urls, **kwargs).decode('pil')
    ds = ds.to_tuple('__key__', 'jpg', 'txt', 'tags.json')
    ds = ds.map_tuple(_dataset_id, image_transform, _identity, _identity)
    return ds


def WebDictionaryDataset(urls, image_transform, args, **kwargs) -> wds.WebDataset:
    ds = wds.WebDataset(wds.PytorchShardList(urls, **kwargs)).decode('pil')
    ds = ds.to_tuple('__key__', 'jpg', 'jpg', 'json', 'tags.json')
    ds = ds.map_tuple(_dataset_id, _identity, image_transform, _identity, _identity)
    return ds


def WebDictionaryText(urls, **kwargs) -> list:
    return [json.loads(x[0]) for x in wds.WebDataset(urls, **kwargs).to_tuple('json')]
