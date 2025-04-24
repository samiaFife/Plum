echo "running task"
TASKS_BBH=("boolean_expressions" "hyperbaton" "temporal_sequences" "object_counting" "disambiguation_qa" "logical_deduction_three_objects" "logical_deduction_five_objects" "logical_deduction_seven_objects" "causal_judgement" "date_understanding" "ruin_names" "word_sorting" "geometric_shapes" "movie_recommendation" "salient_translation_error_detection" "formal_fallacies" "penguins_in_a_table" "dyck_languages" "multistep_arithmetic_two" "navigate" "reasoning_about_colored_objects" "tracking_shuffled_objects_three_objects" "tracking_shuffled_objects_five_objects" "tracking_shuffled_objects_seven_objects" "sports_understanding" "snarks" "web_of_lies")
TASKS=("gsm8k" "math" "medqa" "mnli" "mr" "openbookqa" "qnli" "samsum" "sst-2" "trec" "yahoo")
TASKS_NAT_INSTR=("task021" "task050" "task069")
META_DIR="./logs/full_ts/test/$TASK/"
$META_DIR_TEST="test_3"
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
    --meta-dir $META_DIR \
    --meta-name "$TASK-1.txt" \
    --meta-test-dir "./logs_test/" \
    --meta-test-name $META_DIR_TEST \
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
    --meta-dir $META_DIR \
    --meta-name "$TASK-1.txt" \
    --meta-test-dir "./logs_test/" \
    --meta-test-name $META_DIR_TEST \
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
    --meta-dir $META_DIR \
    --meta-name "$TASK-1.txt" \
    --meta-test-dir "./logs_test/" \
    --meta-test-name $META_DIR_TEST \
    --print-orig \
    --key-id 0 \
    --batch-size 16 \
    --tournament-selection 3 \
    --project-name 'ts-prompt' \
    --checkpoint-freq 10 \
    --output-dir "./output/ts/test/$TASK/"
  echo "Finished task: $TASK"
done