# Universal Captioner

This repository contains the source code of CVPR 2022 submission #2080, entitled *Universal Captioner: Long-Tail Vision-and-Language Model Training through Content-Style Separation*.

The code is intended for reviewing purposes only, and will be cleaned and simplified before the final release. We also commit to release the full set of URL-captions pairs used during training.

## Environment setup
Clone this repository and create the `universalcap` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate universalcap
```

## Data preparation
This codebase employs the [webdataset](https://github.com/webdataset/webdataset) file format to store input images, tags and annotations. 

Each training sample used during XE training should be represented with a webdataset item with three keys: 

- `jpg`, storing the input image;
- `txt`, storing the target caption in plain text;
- `tags.json`, storing tags as a JSON-encoded list of strings.

Loading a training item should, therefore, return the following:
```
>>> import webdataset as wds
>>> ds = wds.WebDataset('conceptual-captions-384-training-000.tar').decode('jpg')
>>> next(iter(ds))
{'__key__': 'cc3m_0000000', '__url__': 'conceptual-captions-384-training-000.tar', '__worker__': 'None', '__rank__': 'None', '__nodeinfo__': "('login02', 3947160)", 'jpg': <PIL.Image.Image image mode=RGB size=575x384 at 0x7FFF69118B50>, 'tags.json': ['buses', 'mumbai', 'busses', 'india', 'incidents', 'taxis', 'halls', 'vehicles', 'stationary', 'parking'], 'txt': 'a very typical bus station'}
```

Each image used during SCST finetuning or evaluation, instead, should be represented with a webdataset item with three keys:

- `jpg`, storing the input image;
- `json`, storing target captions as a JSON-encoded list of strings
- `tags.json`, storing tags as a JSON-encoded list of strings.


Loading an item should, in this case, return the following:
```
{'__key__': 'COCO_val2014_000000184613', '__url__': 'coco-384-validation-dict-625-000.tar', '__worker__': 'None', '__rank__': 'None', '__nodeinfo__': "('login02', 3964837)", 'jpg': <PIL.Image.Image image mode=RGB size=571x384 at 0x7FFF5E248BE0>, 'json': ['A child holding a flowered umbrella and petting a yak.', 'A young man holding an umbrella next to a herd of cattle.', 'a young boy barefoot holding an umbrella touching the horn of a cow', 'A young boy with an umbrella who is touching the horn of a cow.', 'A boy holding an umbrella while standing next to livestock.'], 'tags.json': ['oxen', 'bangladesh', 'philippines', 'indonesia', 'india', 'ox', 'cattle', 'pradesh', 'vietnamese', 'agriculture']}
```

When creating webdataset files, we employ sharding through the [`webdataset.ShardWriter` class](https://webdataset.github.io/webdataset/creating/). This results in a list of `.tar` files for each training/evaluation dataset, each containing a chunk of images with their corresponding annotations. For reference, `coco-cc3m-cc12m-yfcc-wit.shards` contains the list of shards used during the first training stage with XE, and `coco-training-dict.shards` the list of shards used during SCST fine-tuning. 


## Training configuration
Training is done in a distributed fashion, employing PyTorch and the [DeepSpeed](https://www.deepspeed.ai/) library.

### DeepSpeed
Before starting the training, adjust the DeepSpeed configuration file (`config.json` of this repository) according to your needs. In particular, you might want to adjust the total batch size (`train_batch_size`), the batch size of each GPU (`train_micro_batch_size_per_gpu`) and the number of accumulation steps (`gradient_accumulation_steps`). 

### Distributed configuration
Each node should also set the following environment variables: `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`. If you are not familiar with their meaning, see the [PyTorch documentation](https://pytorch.org/docs/stable/distributed.html#initialization) on initializing a distributed training using environment variables. 

Each process participating to the distributed environment should be assigned to only one GPU on its node. Therefore, `LOCAL_RANK` is always set to 0 and `CUDA_VISIBLE_DEVICES` or cgroups should be adopted to make only one GPU visible to each process.


In a SLURM cluster, this can be achieved via the following job submission script:

```
#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH --output=<out_log_fie>
#SBATCH --error=<err_log_file>
#SBATCH --ntasks=<world_size>
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1

. <path_to_conda>/etc/profile.d/conda.sh
conda activate universalcap

cd universalcap

IFS=, read MASTER_ADDR <<< `python expand_nodelist.py $SLURM_NODELIST`
export MASTER_ADDR
export MASTER_PORT=`comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export WORLD_SIZE=<world_size>
export LOCAL_RANK=0

# Launching the master process
export RANK=0
srun -N1 -n1 -w $MASTER_ADDR --gpus=1 --exclusive python -u train_webdataset.py <arguments> &
sleep 5

# Launching the remaining processes
for i in {1..<world_size - 1>}; do
          export RANK=$i
          srun -N1 -n1 --gpus=1 --exclusive python -u train_webdataset.py <arguments> &
done
wait

```
where `expand_nodelist.py` is a Python script that takes the `SLURM_NODELIST` variable and return the first hostname. For instance, if `SLURM_NODELIST` is `gpu206n[02-07,09-10,12-14,20]`, `expand_nodelist.py` should return `gpu206n02`.

### Filesystem paths
Modify `train_webdataset.py` and `finetune_webdataset.py` by setting:

- `args.webdataset_path`, to the path in your filesystem in which you are storing webdatasets
- `args.checkpoint_dir`, to the path in your filesystem in which you want to store checkpoints.

## Training commands
To run the first XE training stage, use the following arguments to `train_webdataset.py`:

```
python -u train_webdataset.py --N_enc 3 --N_dec 6 --d_ff 2048 --d_model 512 --head 8 --deepspeed_config config.json --batch_size 108 --workers 4 --logging small_xe --warmup 6000 --lr_multiplier 5 --eval_interval 1000 --coco_balancing_factor 0.1 --use_gpt_init --shards_file coco-cc3m-cc12m-yfcc-wit.shards
```

Transformer hyperparameters (e.g. `N_dec`, `d_ff`, `d_model`, `head`) can be adjusted depending on the model configuration which needs to be trained.

To run the SCST training stage, use the following arguments to `finetune_webdataset.py`:

```
python -u finetune_webdataset.py --N_enc 3 --N_dec 6 --d_ff 2048 --d_model 512 --head 8 --deepspeed_config config.json --batch_size 8 --workers 1 --logging small_scst small_xe <iter> --eval_interval 500 --input_tags --shards_file coco-training-dict.shards --scst --lr 5e-6 --save_interval 500
```

where `<iter>` indicates the iteration at which XE pre-training weights should be loaded.

In case you need to resume training, the `--resume_last` flag can be employed in both training and finetuning scripts.

Evaluation, done employing the [speaksee library](https://github.com/aimagelab/speaksee), is conducted and printed to standard output every `--eval_interval` steps.
