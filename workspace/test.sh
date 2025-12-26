CUR_DIR=$(cd "$(dirname "$0")"; pwd)
export PYTHONPATH=$CUR_DIR/..:$PYTHONPATH
torchrun --nproc_per_node=2 test_infer.py # 2>&1 | tee test_infer.log
