export HF_TOKEN="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"

task=openbookqa #mathqa,boolq,copa,winogrande...
shots=0 
python -u generate_task_data.py \
  --output-file data/${task}-${shots}shot.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots}