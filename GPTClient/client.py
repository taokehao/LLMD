from pyexpat.errors import messages

import requests
import json
import warnings
import time
from .context import HistoryStack




class Client:
    def __init__(
        self, api_key,
        endpoint='http://localhost/v1/chat-messages',
        mask=None, model='gpt-4o', history_thresh=0,
        retry=3, wait_time=5,
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.mask = mask
        self.model = model
        self.history = HistoryStack(capacity=history_thresh, mask=mask)
        self.retry = retry
        self.wait_time = wait_time
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        # self.history_thresh = history_thresh


    @property
    def headers(self) -> dict:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def __send_request__(self, messages) -> requests.Response:
        # data = {
        #     'model': self.model,
        #     'temperature': self.temperature,
        #     'top_p': self.top_p,
        #     'frequency_penalty': self.frequency_penalty,
        #     'presence_penalty': self.presence_penalty,
        #     'messages': messages
        # }
        # return requests.post(self.endpoint, headers=self.headers, data=json.dumps(data))
        payload = {
            "inputs": {},
            "query": messages,
            "response_mode": "blocking",  # 可改为 blocking 测试不同模式
            "conversation_id": "",  # 留空创建新会话
            "user": "taokehao",
        }
        return requests.post(self.endpoint, headers=self.headers, json=payload)

    @staticmethod
    def __decode_request__(response:requests.Response) -> str:
        return response.json()['answer']

    def ask(self, query, save_history=False) -> str :
        # messages = self.history.history() + [self.history.one_message(query, type='user')]
        messages = self.history.history() + query
        # print(messages)
        response = self.__send_request__(
            messages=messages
        )
        valid_flag = True
        if response.status_code == 200:
            ...
        else:
            warnings.warn(f"Request failed with status code {response.status_code}")
            print(response.json())
            valid_flag = False
            for i in range(self.retry):
                time.sleep(self.wait_time)
                response = self.__send_request__(
                    messages = self.history.history() + query
                )
                if response.status_code == 200:
                    valid_flag = True
                    break
                else:
                    warnings.warn(f"Request retry attempt {i+1} failed.")
                    
        if not valid_flag:
            warnings.warn(f"Unable to acquire response from server after {self.retry+1} times attempt.")
            return None
        answer = self.__decode_request__(response)
        if save_history:
            self.history.append(query=query, answer=answer)
        return answer





if __name__ == '__main__':
    ...


