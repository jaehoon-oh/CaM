export HF_TOKEN="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"

task=openbookqa #mathqa,boolq,copa,winogrande...
shots=0

GPU=0
model=meta-llama/Llama-2-7b-hf
model_arch=llama   # llama / gpt-neox / opt
CUDA_VISIBLE_DEVICES=${GPU} python -u run_lm_eval_harness.py \
  --input-path data/${task}-${shots}shot.jsonl \
  --output-path results/${task}-${shots}-${model_arch}-full.jsonl \
  --model-name ${model} \
  --model-type ${model_arch}


method=h2o #{streamingllm, cam, h2o}
CUDA_VISIBLE_DEVICES=${GPU} python -u run_lm_eval_harness.py \
  --input-path data/${task}-${shots}shot.jsonl \
  --output-path results/${task}-${shots}-${model_arch}-${method}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch} \
  --start_ratio 0.1 \
  --recent_ratio 0.1 \
  --enable_small_cache \
  --method ${method}