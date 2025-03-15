import requests
import json
import time

# 配置参数
API_KEY = "app-YgM4Zw0JF8liojB5Rvb8njXo"
USER_ID = "taokehao"


# 发起会话函数
def send_message():
    url = "http://localhost/v1/chat-messages"

    payload = {
        "inputs": {},
        "query": '''Your are a professor in material science. You are going to help with optimizing a material with your experience in your field.
I have a material categorized as LLTO or its variants, along with its ionic migration barrier value. The ionic migration barrier is the minimum energy required for a lithium ion to move through the material. A lower barrier generally indicates higher ionic conductivity, which is critical for applications like batteries, fuel cells, and other electrochemical devices.
Below, I will provide the material's structure and its ionic migration barrier value. The structure will be enclosed between ">>>" and "<<<", with each line representing an atom. The format is: Element Symbol | x, y, z where the coordinates are in Cartesian format.
Material Structure:
>>>
Li | 0.000,1.938,6.689
Li | 1.938,0.000,6.689
Li | 0.000,1.938,20.068
Li | 1.938,0.000,20.068
La | 0.000,0.000,11.207
Ti | 0.000,0.000,4.220
O | 0.000,0.000,1.943
O | 0.000,0.000,6.008
La | 1.938,1.938,24.586
Ti | 1.938,1.938,17.599
O | 1.938,1.938,15.322
O | 1.938,1.938,19.387
La | 0.000,0.000,15.551
Ti | 0.000,0.000,22.537
O | 0.000,0.000,24.814
O | 0.000,0.000,20.749
La | 1.938,1.938,2.172
Ti | 1.938,1.938,9.159
O | 1.938,1.938,11.435
O | 1.938,1.938,7.370
Ti | 0.000,0.000,0.000
Ti | 1.938,1.938,13.379
O | 0.000,1.938,3.814
O | 1.938,0.000,17.192
O | 1.938,0.000,3.814
O | 0.000,1.938,17.192
O | 0.000,1.938,22.944
O | 1.938,0.000,9.565
O | 1.938,0.000,22.944
O | 0.000,1.938,9.565
O | 0.000,1.938,0.000
O | 1.938,0.000,13.379
O | 1.938,0.000,0.000
O | 0.000,1.938,13.379
Li | 3.877,1.938,6.689
Li | 5.815,0.000,6.689
Li | 3.877,1.938,20.068
Li | 5.815,0.000,20.068
La | 3.877,0.000,11.207
Ti | 3.877,0.000,4.220
O | 3.877,0.000,1.943
O | 3.877,0.000,6.008
La | 5.815,1.938,24.586
Ti | 5.815,1.938,17.599
O | 5.815,1.938,15.322
O | 5.815,1.938,19.387
La | 3.877,0.000,15.551
Ti | 3.877,0.000,22.537
O | 3.877,0.000,24.814
O | 3.877,0.000,20.749
La | 5.815,1.938,2.172
Ti | 5.815,1.938,9.159
O | 5.815,1.938,11.435
O | 5.815,1.938,7.370
Ti | 3.877,0.000,0.000
Ti | 5.815,1.938,13.379
O | 3.877,1.938,3.814
O | 5.815,0.000,17.192
O | 5.815,0.000,3.814
O | 3.877,1.938,17.192
O | 3.877,1.938,22.944
O | 5.815,0.000,9.565
O | 5.815,0.000,22.944
O | 3.877,1.938,9.565
O | 3.877,1.938,0.000
O | 5.815,0.000,13.379
O | 5.815,0.000,0.000
O | 3.877,1.938,13.379
Li | 0.000,5.815,6.689
Li | 1.938,3.877,6.689
Li | 0.000,5.815,20.068
Li | 1.938,3.877,20.068
La | 0.000,3.877,11.207
Ti | 0.000,3.877,4.220
O | 0.000,3.877,1.943
O | 0.000,3.877,6.008
La | 1.938,5.815,24.586
Ti | 1.938,5.815,17.599
O | 1.938,5.815,15.322
O | 1.938,5.815,19.387
La | 0.000,3.877,15.551
Ti | 0.000,3.877,22.537
O | 0.000,3.877,24.814
O | 0.000,3.877,20.749
La | 1.938,5.815,2.172
Ti | 1.938,5.815,9.159
O | 1.938,5.815,11.435
O | 1.938,5.815,7.370
Ti | 0.000,3.877,0.000
Ti | 1.938,5.815,13.379
O | 0.000,5.815,3.814
O | 1.938,3.877,17.192
O | 1.938,3.877,3.814
O | 0.000,5.815,17.192
O | 0.000,5.815,22.944
O | 1.938,3.877,9.565
O | 1.938,3.877,22.944
O | 0.000,5.815,9.565
O | 0.000,5.815,0.000
O | 1.938,3.877,13.379
O | 1.938,3.877,0.000
O | 0.000,5.815,13.379
Li | 3.877,5.815,6.689
Li | 5.815,3.877,6.689
Li | 3.877,5.815,20.068
Li | 5.815,3.877,20.068
La | 3.877,3.877,11.207
Ti | 3.877,3.877,4.220
O | 3.877,3.877,1.943
O | 3.877,3.877,6.008
La | 5.815,5.815,24.586
Ti | 5.815,5.815,17.599
O | 5.815,5.815,15.322
O | 5.815,5.815,19.387
La | 3.877,3.877,15.551
Ti | 3.877,3.877,22.537
O | 3.877,3.877,24.814
O | 3.877,3.877,20.749
La | 5.815,5.815,2.172
Ti | 5.815,5.815,9.159
O | 5.815,5.815,11.435
O | 5.815,5.815,7.370
Ti | 3.877,3.877,0.000
Ti | 5.815,5.815,13.379
O | 3.877,5.815,3.814
O | 5.815,3.877,17.192
O | 5.815,3.877,3.814
O | 3.877,5.815,17.192
O | 3.877,5.815,22.944
O | 5.815,3.877,9.565
O | 5.815,3.877,22.944
O | 3.877,5.815,9.565
O | 3.877,5.815,0.000
O | 5.815,3.877,13.379
O | 5.815,3.877,0.000
O | 3.877,5.815,13.379
<<<
Ionic Migration Barrier: 0.93 eV
Propose a modification to achieve a migration barrier of 0.50 eV. Use one of the following modification types:
1. exchange: Swap the element types of two atoms while keeping their positions.
2. substitute: Replace the element type of an atom with another while keeping its position.
Output the result as a Python dictionary in the format:
{
  "Reason": $REASON,
  "Modification": [
    {"type": $TYPE, "action": ($ARG1, $ARG2)},
    ...
  ]
}
Requirements:
1. Reason: Explain the analysis behind the chosen modifications.
2. Modification: A list of up to 15, with each entry as described above.
3. $TYPE: Must be either "exchange" or "substitute".
4. $ARG1 and $ARG2:
    - For "exchange": $ARG1 and $ARG2 are the indices of the atoms to swap (indices correspond to line numbers, starting at 0).
    - For "substitute": $ARG1 is the atom index and $ARG2 is the new element symbol.
5. Do not modify Lithium, Titanium, or nonmetals. Substitutions should only use transition metals.
6. Ensure correct Python syntax, escaping quotes as needed.''',
        "response_mode": "blocking",  # 可改为 blocking 测试不同模式
        "conversation_id": "",  # 留空创建新会话
        "user": "taokehao",
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            # data=json.dumps(payload),
            stream=payload["response_mode"] == "streaming"
        )
        response.raise_for_status()

        conversation_id = ""

        # 处理流式响应
        if payload["response_mode"] == "streaming":
            for line in response.iter_lines():
                if line:
                    try:
                        event = json.loads(line.decode('utf-8'))
                        print(f"[流式响应] {event.get('event', '')}")

                        # 捕获会话ID
                        if 'conversation_id' in event:
                            conversation_id = event['conversation_id']

                        # 实时显示回答片段
                        if event.get('event') == 'message' and 'answer' in event:
                            print(f"AI: {event['answer'][:50]}...")

                    except json.JSONDecodeError:
                        print(f"解析失败: {line}")

        # 处理非流式响应
        else:
            data = response.json()
            conversation_id = data['conversation_id']
            print(f"完整回答: {data['answer']}...")

        return conversation_id

    except Exception as e:
        print(f"发送消息失败: {str(e)}")
        return None


# 查询历史函数
def get_history(conversation_id=""):
    url = "http://localhost/v1/messages"

    params = {
        "user": USER_ID,
        "conversation_id": conversation_id
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        print(f"\n历史记录查询结果 ({len(data['data'])} 条):")
        for msg in data['data']:
            print(f"[{msg['created_at']}] Q: {msg['query']}")
            print(f"   A: {msg['answer'][:50]}...\n")

    except Exception as e:
        print(f"查询历史失败: {str(e)}")


# 主流程
if __name__ == "__main__":
    print("正在发起新会话...")
    new_conversation_id = send_message()

    if new_conversation_id:
        print(f"\n获取到会话ID: {new_conversation_id}")

        # 等待1秒确保数据写入
        time.sleep(1)

        print("\n正在查询最新历史记录...")
        get_history(new_conversation_id)