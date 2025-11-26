IMAGE="registry2.d.pjlab.org.cn/ccr-chenchiyu/910c:xtuner_cann8.2RC2_ipc_20251120_1"

clusterx run \
    -J ${1:-rl-qwen3-30ba3b-gsm8k-n1-graph} \
    -N 1 \
    --gpus-per-task 16 \
    --cpus-per-task 576 \
    --memory-per-task 1600 \
    --no-env \
    --image ${IMAGE} \
    bash $(realpath ${2:-run_30BA3B_rl.sh})
