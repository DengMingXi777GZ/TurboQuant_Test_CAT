import requests

# vLLM 默认启动地址
url = "http://localhost:8000/v1/completions"

# 测试请求（你可以随便改 prompt）
data = {
    "model": "Qwen35_2b",
    "prompt": "你好，请简单介绍一下自己",
    "max_tokens": 128,
    "temperature": 0.7
}

try:
    print("正在测试 vLLM 8000 端口...")
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print("✅ 端口通了！服务正常运行！")
        print("返回结果：")
        print(response.json()["choices"][0]["text"])
    else:
        print(f"❌ 服务异常，状态码：{response.status_code}")
        print(response.text)

except Exception as e:
    print("❌ 无法连接 vLLM 服务！")
    print("错误原因：", e)
