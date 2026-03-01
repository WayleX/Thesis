# DeepFake Dataset Generation

### 1. Download the dataset

```bash
bash download_dataset.sh
```
If script is not working, you can do it manually



### 2. Model setup

Basically setup.sh scripts

As of now it clones repos and installs requirements into venv + some fixes

```bash
# Wan2.2
bash Wan2/setup.sh

# HunyuanVideo 1.5
bash Hunyuan/setup.sh
```

Here it is just simple setupping, works on cuda under 13.0, 13.0 didn't test properly

#### Hunyuan


For Hunyuan it clones repo, creates venv, takes default requirements + few overrides (flash-attn, diffusers) in bash script

Also it needs to swap diffusers file as current implementation doesnt correctly utilize flash-attn

#### Wan

For Wan, it clones repo, creates venv, installs requirements, install flash-attn that worked


### 3. Generate videos

Both models support two modes via `--mode`:
- `i2v` (default) — **image-to-video**: uses the first frame from the dataset as a conditioning image
- `t2v` — **text-to-video**: generates purely from the text prompt, no image input

For wan we have 5b model and 14b model, you need to choose one

```bash
source Wan2/Wan2.2/venv/bin/activate

source Hunyuan/HunyuanVideo-1.5/venv/bin/activate

```

#### Wan2.2

generate.sh is basically parsing input, activating venv and launching python file

You can actually launch python directly if you want, or use generate_multiple

```bash
# I2V (default) — 14B model
bash Wan2/generate.sh --mode t2v

bash Wan2/generate.sh --mode i2v

#small model
bash Wan2/generate.sh --mode t2v --model-size 5b

bash Wan2/generate.sh --mode i2v --model-size 5b

bash Wan2/generate_multiple.sh --mode t2v --model-size 5b --gpus 0,1



# I2V — 5B model with custom paths (only absoulte paths probably, didnt check relative sorry)
bash Wan2/generate.sh --model-size 5b --dataset-dir /data/dataset --output-dir /data/output

```


#### HunyuanVideo 1.5

generate.sh is basically parsing input, activating venv and launching python file

You can actually launch python directly if you want, or use generate_multiple


```bash
bash Hunyuan/generate.sh

bash Hunyuan/generate.sh --mode t2v

bash Hunyuan/generate.sh --mode t2v --output-dir /data/t2v_output

bash Hunyuan/generate_multiple.sh --mode t2v --gpus 0,1

```
