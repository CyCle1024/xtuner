export QWEN3_MODEL_PATH='/mnt/chenchiyu/model/Qwen3-30B-A3B'
export ROLLOUT_DATA_PATH='/mnt/chenchiyu/data/gsm8k/train.jsonl'
export ROLLOUT_TEST_DATA_PATH='/mnt/chenchiyu/data/gsm8k/test.jsonl'
export XTUNER_DIR=/mnt/chenchiyu/workspace/rl_test_ref/xtuner_graph_mode
export LMDEPLOY_DIR=/mnt/chenchiyu/workspace/rl_test_ref/lmdeploy
export PYTHONPATH=$XTUNER_DIR:$LMDEPLOY_DIR:$PYTHONPATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export CPU_AFFINITY_CONF=0
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True,segment_size_mb:128"
export MULTI_STREAM_MEMORY_REUSE=2
# export TORCH_HCCL_ZERO_COPY=1
export ASCEND_PROCESS_LOG_PATH=/mnt/chenchiyu/ascend_plog/${JOB_NAME}/${RANK}

#export ASDOPS_LOG_LEVEL=INFO
#export ATB_LOG_LEVEL=INFO

cd $XTUNER_DIR
ray stop
bash $XTUNER_DIR/examples/v1/scripts/run_rl.sh $XTUNER_DIR/examples/v1/config/rl_qwen3_30BA3B_grpo.py "lmdeploy" $QWEN3_MODEL_PATH $ROLLOUT_DATA_PATH $ROLLOUT_TEST_DATA_PATH
