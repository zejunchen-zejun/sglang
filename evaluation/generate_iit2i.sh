curl -sS -X POST "http://localhost:30000/v1/images/edits" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -F "url=https://primus-biz-data.oss-cn-wulanchabu.aliyuncs.com/laap%2Fcomfyui%2Fmodels%2Floras%2FKK%2F%E8%BE%93%E5%85%A5%E5%9B%BE%E5%83%8F%2F%E4%B8%8A%E4%BC%A0oss%2F%E5%8F%8C%E5%9B%BE%2F2026-01-06%2Frow_3_col_2.png?OSSAccessKeyId=LTAI5t7WDZ1dYGN4JULN8KX9&Expires=1783245266&Signature=ECmLp430FN9qAK3iivG4n7I%2BTo8%3D" \
  -F "url=https://primus-biz-data.oss-cn-wulanchabu.aliyuncs.com/laap%2Fcomfyui%2Fmodels%2Floras%2FKK%2F%E8%BE%93%E5%85%A5%E5%9B%BE%E5%83%8F%2F%E4%B8%8A%E4%BC%A0oss%2F%E5%8F%8C%E5%9B%BE%2F2026-01-06%2Frow_3_col_3.png?OSSAccessKeyId=LTAI5t7WDZ1dYGN4JULN8KX9&Expires=1783245242&Signature=l7bLiGwW9IBqbwCgBx6W5QdXFzQ%3D" \
  -F "prompt=将两个人合成一张合影，让他们并肩站立，面带微笑，背景为山景，保持自然的光线
和透视关系。" \
  -F "neg_prompt= " \
  -F "guidance_scale=4.0" \
  -F "embedded_guidance_scale=6.0" \
  -F "seed=0" \
  -F "num_inference_steps=8" \
  -F "response_format=b64_json"

