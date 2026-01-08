curl -sS -X POST "http://localhost:30000/v1/images/edits" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -F "url=https://primus-biz-data.oss-cn-wulanchabu.aliyuncs.com/laap%2Fcomfyui%2Fmodels%2Floras%2FKK%2F%E8%BE%93%E5%85%A5%E5%9B%BE%E5%83%8F%2F%E4%B8%8A%E4%BC%A0oss%2F%E6%97%A0%E6%A0%87%E9%A2%98%E8%A1%A8%E6%A0%BC%2F2026-01-06%2Frow_5_col_2.png?OSSAccessKeyId=LTAI5t7WDZ1dYGN4JULN8KX9&Expires=1783244176&Signature=gIw2QdkpXHxvjpDxh8xdHUlVHr0%3D" \
  -F "prompt=为图中婴儿添加汉朝风格的头饰，保持其传统造型与细节，使其与整体服饰协调。" \
  -F "neg_prompt= " \
  -F "guidance_scale=4.0" \
  -F "embedded_guidance_scale=6.0" \
  -F "seed=0" \
  -F "response_format=b64_json"

