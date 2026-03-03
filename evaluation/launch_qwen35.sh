export PYTHONPATH=/home/qichu/qwen35_sgl/sglang/python

python3 -m sglang.launch_server \
        --port 9000 \
        --model-path /mnt/raid0/models/Qwen3.5-397B-A17B/ \
        --tp-size 8 \
        --attention-backend triton \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-multimodal \
        --mem-fraction-static 0.85 \
        # --speculative-algorithm NEXTN \
        # --speculative-num-steps 3 \
        # --speculative-eagle-topk 1 \
        # --speculative-num-draft-tokens 4
