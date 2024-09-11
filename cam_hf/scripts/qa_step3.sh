export HF_TOKEN="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"

task=openbookqa #mathqa,boolq,copa,winogrande...
shots=0

GPU=0
model=meta-llama/Llama-2-7b-hf
model_arch=llama   # llama / gpt-neox / opt
method_array=("full" "streamingllm" "h2o" "cam")

for method in "${method_array[@]}"
do
  python -u evaluate_task_result.py \
    --result-file results/${task}-${shots}shot-${model_arch}-${method}.jsonl \
    --task-name ${task} \
    --num-fewshot ${shots} \
    --model-type ${model_arch} \
    --start-ratio 0.1 \
    --recent-ratio 0.1 \
    --ret-path results/${task}-${shots}shot-${model_arch}-${method}.txt
done