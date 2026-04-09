# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES=0,1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../python:${PYTHONPATH}"
# export HF_DATASETS_OFFLINE=1

parallel_decoding='threshold' # 
length=2048 # generate length
block_length=32 # block length
model_path='/path/to/local_model' # your model path  
threshold=0.5 # threshold for parallel decoding
cache='prefix' # or 'prefix' for prefix cache; or '' if you don't want to use cache
warmup_times=0 # warmup times for cache
prefix_look=0
after_look=0
cont_weight=0 # cont weight
use_compile=True # use compile
tp_size=2 # tensor parallel size
gpus='0;1' # gpus for tensor parallel inference
parallel='tp' # 'tp' for tensor parallel or 'dp' for data parallel
output_dir='outputs' # your customer output path
model_type='llada2' # llada2 (for llada2-mini) 
use_bd=True # use block diffusion
master_port="23456"
save_samples=True # save samples
batch_size=1

# tasks:  gsm8k_llada_mini   minerva_math_algebra  minerva_math500  asdiv_llada_mini 
if [ "$parallel" = "tp" ]; then
  for task in gsm8k_llada_mini; do
    output_path="${output_dir}/${task}"
    python eval_dinfer_sglang.py --tasks "${task}" \
      --confirm_run_unsafe_code --model dInfer_eval \
      --model_args model_path="${model_path}",gen_length="${length}",block_length="${block_length}",threshold="${threshold}",low_threshold="${low_threshold}",show_speed=True,save_dir="${output_path}",parallel_decoding="${parallel_decoding}",cache="${cache}",warmup_times="${warmup_times}",use_compile="${use_compile}",tp_size="${tp_size}",parallel="${parallel}",cont_weight="${cont_weight}",use_credit="${use_credit}",prefix_look="${prefix_look}",after_look="${after_look}",gpus="${gpus}",model_type="${model_type}",use_bd="${use_bd}",master_port="${master_port}",save_samples="${save_samples}" \
      --output_path "${output_path}" --include_path tasks --apply_chat_template
  done
else
  echo "parallel must be tp"
fi


