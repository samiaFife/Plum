echo "running task"
TASKS_BBH=("disambiguation_qa")
TASKS=("mnli")
TASKS_NAT_INSTR=("task021")
for TASK in "${TASKS[@]}"; do
  echo "Running task: $TASK"
# mkdir ./output/hs/test/$TASK/checkpoints
  python ./main.py \
    --task-name $TASK \
    --data-dir "./data/" \
    --algorithm "tabu" \
    --mode "Instruction Only" \
    --train-seed 42 \
    --num-compose 1 \
    --num-candidates 10 \
    --backbone "tlite" \
    --num-iter 10 \
    --patience 5 \
    --write-preds \
    --meta-dir "./logs/ts/test/$TASK/" \
    --meta-name "$TASK-1.txt" \
    --meta-test-dir "./logs_test/" \
    --meta-test-name "test_2" \
    --print-orig \
    --key-id 0 \
    --batch-size 16 \
    --tournament-selection 3 \
    --project-name 'ts-prompt' \
    --checkpoint-freq 10 \
    --output-dir "./output/ts/test/$TASK/"
  echo "Finished task: $TASK"
done
for TASK in "${TASKS_NAT_INSTR[@]}"; do
  echo "Running task: $TASK"
  python ./main.py \
    --task-name $TASK \
    --bench-name "natural_instructions" \
    --data-dir "./data/" \
    --algorithm "tabu" \
    --mode "Instruction Only" \
    --train-seed 42 \
    --num-compose 1 \
    --num-candidates 10 \
    --backbone "tlite" \
    --num-iter 10 \
    --patience 5 \
    --write-preds \
    --meta-dir "./logs/ts/test/$TASK/" \
    --meta-name "$TASK-1.txt" \
    --meta-test-dir "./logs_test/" \
    --meta-test-name "test_2" \
    --print-orig \
    --key-id 0 \
    --batch-size 16 \
    --tournament-selection 3 \
    --project-name 'ts-prompt' \
    --checkpoint-freq 10 \
    --output-dir "./output/ts/test/$TASK/"
  echo "Finished task: $TASK"
done
for TASK in "${TASKS_BBH[@]}"; do
  echo "Running task: $TASK"
  echo "Running task: $TASK"
  python ./main.py \
    --task-name $TASK \
    --bench-name "bbh" \
    --data-dir "./data/" \
    --algorithm "tabu" \
    --mode "Instruction Only" \
    --train-seed 42 \
    --num-compose 1 \
    --num-candidates 10 \
    --backbone "tlite" \
    --num-iter 10 \
    --patience 5 \
    --write-preds \
    --meta-dir "./logs/ts/test/$TASK/" \
    --meta-name "$TASK-1.txt" \
    --meta-test-dir "./logs_test/" \
    --meta-test-name "test_2" \
    --print-orig \
    --key-id 0 \
    --batch-size 16 \
    --tournament-selection 3 \
    --project-name 'ts-prompt' \
    --checkpoint-freq 10 \
    --output-dir "./output/ts/test/$TASK/"
  echo "Finished task: $TASK"
done