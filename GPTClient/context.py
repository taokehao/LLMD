

class HistoryStack:
    def __init__(self, capacity=20, mask=None):
        self.capacity = capacity
        # self.save_all = save_all
        self.content = []
        self.outdated = []
        if mask:
            self.init_prompt = mask
        else:
            self.init_prompt = []

    @staticmethod
    def one_message(text:str, type:str) -> dict:
        if type == 'user':
            return {'role': 'user', 'content': text}
        elif type == 'assistant':
            return {'role': 'assistant', 'content': text}
        else:
            raise ValueError('type must be user or assistant')

    def encode_history_cell(self, user_message:str, assistant_message:str) -> dict:
        return {
            'user': self.one_message(user_message, 'user'),
            'assistant': self.one_message(assistant_message, 'assistant')
        }

    def append(self, query: str, answer: str) -> None:
        cell = self.encode_history_cell(query, answer)
        self.content.append(cell)
        if len(self.content) > self.capacity:
            self.outdated.append(self.content.pop(0))

    def clear(self):
        self.content = []
        self.outdated = []

    def valid_history(self) -> list:
        ret = []
        for cell in self.content:
            ret.append(cell['user'])
            ret.append(cell['assistant'])
        return ret

    def outdated_history(self) -> list:
        ret = []
        for cell in self.outdated:
            ret.append(cell['user'])
            ret.append(cell['assistant'])
        return ret

    def all_history(self) -> list:
        return self.init_prompt + self.outdated_history() + self.valid_history()

    def history(self) -> list:
        # return self.init_prompt + self.valid_history()
        return self.init_prompt


