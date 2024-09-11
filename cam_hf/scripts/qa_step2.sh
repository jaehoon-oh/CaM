export HF_TOKEN="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"

task=openbookqa #mathqa,boolq,copa,winogrande...
shots=0

GPU=0
model=meta-llama/Llama-2-7b-hf
model_arch=llama   # llama / gpt-neox / opt
# CUDA_VISIBLE_DEVICES=${GPU} python -u run_lm_eval_harness.py \
#   --input-path data/${task}-${shots}shot.jsonl \
#   --output-path results/${task}-${shots}shot-${model_arch}-full.jsonl \
#   --model-name ${model} \
#   --model-type ${model_arch}

method_array=("streamingllm" "h2o" "cam")
for method in "${method_array[@]}"
do
  CUDA_VISIBLE_DEVICES=${GPU} python -u run_lm_eval_harness.py \
    --input-path data/${task}-${shots}shot.jsonl \
    --output-path results/${task}-${shots}shot-${model_arch}-${method}.jsonl \
    --model-name ${model} \
    --model-type ${model_arch} \
    --start_ratio 0.1 \
    --recent_ratio 0.1 \
    --enable_small_cache \
    --method ${method}
done