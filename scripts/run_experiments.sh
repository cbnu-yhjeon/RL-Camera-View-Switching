#!/usr/bin/env bash
# 전체 실험 (README '실험 설계'). 실행을 작업 큐로 만들어 PARALLEL개씩 동시에 돈다.
#   RL-0/RL-1  DQN·Bandit × λ_sw 스윕 × seed
#   RL-2       A2C·PPO (λ_sw=0.1)
#   RL-N       노이즈 입력 학습 DQN (λ_sw=0.1)
#   ablation   단일 행동 타깃 DQN (λ_sw=0.1)
# 이어서 비교 정책 평가 → 강건성 재평가 → 보고서.
#
#   bash scripts/run_experiments.sh            # 전부
#   SEEDS="0" PARALLEL=2 bash scripts/run_experiments.sh   # 축소
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
LSWS=${LSWS:-"0 0.05 0.1 0.2 0.3 0.5"}
PARALLEL=${PARALLEL:-4}
LOGS=results/logs
mkdir -p "$LOGS"

jobs=()
for s in $SEEDS; do                    # 오래 걸리는 on-policy(1M 스텝)를 먼저 넣어 꼬리를 줄인다
  jobs+=("configs/a2c.yaml $s 0.1" "configs/ppo.yaml $s 0.1")
done
for s in $SEEDS; do
  jobs+=("configs/dqn_noisy.yaml $s 0.1" "configs/dqn_vanilla.yaml $s 0.1" "configs/dqn_maneuver.yaml $s 0.1")
  for l in $LSWS; do
    jobs+=("configs/dqn.yaml $s $l")
    jobs+=("configs/bandit.yaml $s $l")
  done
done

run_one() {
  cfg=$1; seed=$2; lsw=$3
  name=$(basename "$cfg" .yaml)_lsw${lsw}_seed${seed}
  if [ -f "results/runs/$(basename "$cfg" .yaml)_lsw${lsw}/seed${seed}/test_gt.json" ]; then
    echo "skip $name"; return 0
  fi
  $PY scripts/train.py --config "$cfg" --seed "$seed" --tag "lsw${lsw}" \
      --set reward.lambda_sw="$lsw" > "$LOGS/$name.log" 2>&1 && echo "done $name" || echo "FAIL $name"
}
export -f run_one
export PY LOGS
printf '%s\n' "${jobs[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'run_one {}'

$PY scripts/evaluate.py baselines --lsw $LSWS --conds robustness
$PY scripts/evaluate.py baselines --split val --lsw $LSWS --conds gt noisy-medium
$PY scripts/evaluate.py runs results/runs/dqn_lsw0.1/seed* results/runs/dqn_noisy_lsw0.1/seed* \
    results/runs/dqn_maneuver_lsw0.1/seed* --conds robustness
$PY scripts/make_report.py
