#!/usr/bin/env bash

set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [current_training_pid|-] [gpu_visible_list]"
  echo "Example: $0 1234567 0"
  echo "Example: $0 - 0,1"
  exit 1
fi

CURRENT_PID="${1:-}"

# GPU visible list for this ablation run. Override with the second CLI arg, e.g.:
#   bash run_edtformer_tmp_ablation.sh - 2
#   bash run_edtformer_tmp_ablation.sh - 2,3
GPU_VISIBLE_LIST="${GPU_VISIBLE_LIST:-2}"
GPU_DEVICES="${2:-${GPU_VISIBLE_LIST}}"

PROJECT_ROOT="/media/data/zhangjingyi/EDTformer"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="$(date +%m%d_%H%M%S)"
NOHUP_LOG="${PROJECT_ROOT}/nohup_run_edtformer_tmp_ablation_${TIMESTAMP}.log"

if [[ "${RUN_UNDER_NOHUP:-1}" == "1" && "${EDTFORMER_ABLATION_NOHUP_LAUNCHED:-0}" != "1" ]]; then
  echo "[$(date '+%F %T')] launching under nohup..."
  echo "[$(date '+%F %T')] log: ${NOHUP_LOG}"
  EDTFORMER_ABLATION_NOHUP_LAUNCHED=1 nohup bash "$0" "$@" > "${NOHUP_LOG}" 2>&1 &
  echo "[$(date '+%F %T')] background pid: $!"
  exit 0
fi

VARIANT_ORDER=(
  "msls:exp4_EDTformer_default_msls"
  "pitts:exp4_EDTformer_default_pitts"
  "all:exp4_EDTformer_default_all"
)

COMMON_ARGS=(
  -u train.py
  --eval_datasets_folder /media/data1/chenshunpeng1/datasets
  --eval_dataset_names pitts30k Msls_740
  --final_eval_dataset_names sped amstertime tokyo nordland svox
  --train_dataset_path /data_nvme/zhangjingyi/Gsv_reflect/mixgsv
  --training_subsets default tmp
  --foundation_model_path /media/data1/zhangjingyi/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth
  --resume_author /media/data/zhangjingyi/ImAge/module/EDTformer.pth
  --epochs_num 20
  --train_batch_size 72
  --infer_batch_size 16
  --resize 322 322
  --lr 5e-6
  --patience 5
  --stage1_epochs 0
  --synthetic_ratio 0.125
)

wait_for_current_run() {
  if [[ -z "${CURRENT_PID}" || "${CURRENT_PID}" == "-" ]]; then
    echo "[$(date '+%F %T')] no current training pid provided; starting immediately."
    return
  fi

  echo "[$(date '+%F %T')] waiting for current training pid ${CURRENT_PID} to finish..."
  while kill -0 "${CURRENT_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date '+%F %T')] current training pid ${CURRENT_PID} has finished."
}

run_variant() {
  local tmp_group="$1"
  local save_dir="$2"
  local log_path="${PROJECT_ROOT}/log_${save_dir}_${TIMESTAMP}.txt"

  echo "[$(date '+%F %T')] starting EDTformer variant tmp_group=${tmp_group}, save_dir=${save_dir}, CUDA_VISIBLE_DEVICES=${GPU_DEVICES}"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" "${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
      --tmp_group "${tmp_group}" \
      --save_dir "${save_dir}"
  ) 2>&1 | tee "${log_path}"
  local exit_code="${PIPESTATUS[0]}"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "[$(date '+%F %T')] variant ${save_dir} failed with exit code ${exit_code}. log: ${log_path}"
    exit "${exit_code}"
  fi
  echo "[$(date '+%F %T')] variant ${save_dir} completed. log: ${log_path}"
}

run_all_variants() {
  local variant_spec tmp_group save_dir

  for variant_spec in "${VARIANT_ORDER[@]}"; do
    tmp_group="${variant_spec%%:*}"
    save_dir="${variant_spec#*:}"
    run_variant "${tmp_group}" "${save_dir}"
  done
}

wait_for_current_run
run_all_variants
