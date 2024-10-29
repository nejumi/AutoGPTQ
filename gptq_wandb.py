import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch
import argparse
import os
import datetime
from datasets import load_dataset
import wandb
import gc

# 環境変数の設定
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

# コマンドライン引数の設定
parser = argparse.ArgumentParser()
parser.add_argument('--bits', type=int, default=8, help='量子化のビット数')
parser.add_argument('--group_size', type=int, default=128, help='量子化のグループ数')
parser.add_argument('--hf_model_repo', type=str, required=True, help='Hugging Faceのモデルリポジトリ')
parser.add_argument('--percdamp', type=float, default=0.01, help='percdampの値')
parser.add_argument('--n_data', type=int, default=1024, help='使用するcalibration dataの行数')
parser.add_argument('--model_seqlen', type=int, default=2048, help='model_seqlenの値')
parser.add_argument('--desc_act', type=bool, default=True, help='desc_actの有無')
parser.add_argument('--dataset', type=str, default="nejumi/wikipedia-ja-20230720-4k", help='calibration dataset')
args = parser.parse_args()

if args.n_data < 1024:
    n = str(round(args.n_data / 1024, 1))
else:
    n = str(args.n_data // 1024)

# データセットの準備（例としてwikitext-2を使用）
from datasets import load_dataset
dataset = load_dataset(args.dataset, split='train', streaming=True)
cal_data = [item['text'] for item in dataset.take(args.n_data)]

# モデルとトークナイザーの設定
hf_model_repo = args.hf_model_repo
tokenizer = AutoTokenizer.from_pretrained(hf_model_repo)

# 量子化の設定
bits = args.bits
quantization_config = GPTQConfig(
     bits=bits,
     group_size=args.group_size,
     dataset=cal_data,
     desc_act=args.desc_act,
     tokenizer=tokenizer,
     use_exllama=False, 
     cache_examples_on_gpu=False,
     damp_percent=args.percdamp,
     use_cuda_fp16=True,
     model_seqlen=args.model_seqlen,
     cache_examples=0
)

# WandBの初期化
project_name = os.getenv('WANDB_PROJECT')
if not project_name:
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    project_name = f'AutoGPTQ_{current_time}'

wandb.init(
    project=project_name,
    config=quantization_config,
)

# 量子化計算用の0を除いた状態での配置を決める
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# GPU情報の自動検出
def get_gpu_memory_config(gpu_memory_disabled=False):
    gpu_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if gpu_memory_disabled:
                gpu_memory[i] = 0
            else:
                total_memory = torch.cuda.get_device_properties(i).total_memory
                usable_memory = int(total_memory * 0.9 / 1024 / 1024 / 1024)
                gpu_memory[i] = f"{usable_memory}GiB"
    
    import psutil
    total_cpu_memory = psutil.virtual_memory().total
    usable_cpu_memory = int(total_cpu_memory * 0.8 / 1024 / 1024 / 1024)
    gpu_memory['cpu'] = f"{usable_cpu_memory}GiB"
    
    return gpu_memory

# 自動検出したメモリ設定を使用
memory_config = get_gpu_memory_config(gpu_memory_disabled=False)

temp_model = AutoModelForCausalLM.from_pretrained(hf_model_repo, torch_dtype=torch.float16, device_map='auto',max_memory=memory_config,)
device_map = temp_model.hf_device_map
#device_map = {k: v + 1 if isinstance(v, int) else v for k, v in device_map.items()}
device_map = {k: 0 if k=='model.embed_tokens' else 'cpu' for k, v in device_map.items()}

del temp_model

print(device_map)

# GPUキャッシュをクリア
torch.cuda.empty_cache()

# 念のため、ガベージコレクションを実行
import gc
gc.collect()

# モデルの読み込みと量子化
quant_memory_config = get_gpu_memory_config(gpu_memory_disabled=True)
quant_model = AutoModelForCausalLM.from_pretrained(hf_model_repo, torch_dtype=torch.float16, 
                                                   quantization_config=quantization_config,
                                                   device_map=device_map, 
                                                   #device_map='auto', 
                                                   max_memory=quant_memory_config,
                                                   )

# モデル名のみを使用して保存先ディレクトリとモデルIDを設定
def get_model_name(hf_model_repo):
    """
    hf_model_repoからモデル名のみを抽出する関数
    例: 'cyberagent/calm3-22b-chat' -> 'calm3-22b-chat'
    """
    return os.path.basename(hf_model_repo)


model_name = get_model_name(hf_model_repo)
pretrained_model_dir = f"{model_name}-GPTQ-Int{bits}-calib-ja-{n}k"
quantized_model_id = f"{model_name}-GPTQ-Int{bits}-calib-ja-{n}k"

quant_model.to("cpu")
quant_model.save_pretrained(pretrained_model_dir, safe_serialization=True)
tokenizer.save_pretrained(pretrained_model_dir)

# Hugging Faceにプッシュする場合はコメントを解除
#quant_model.push_to_hub(quantized_model_id)
#tokenizer.push_to_hub(quantized_model_id)

print(f"Quantized moaved to: {pretrained_model_dir}")
print(f"Quantized model ID: {quantized_model_id}")