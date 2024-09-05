import numpy as np
import jieba

class TextProcessor:
    def __init__(self, path):
        self.path = path
        self.words_list = []
        self.book = {}
        self.book2 = {}
        self.ibook = {}
        self.ibook2 = {}
        self.load_text()
        self.process_text()

    def load_text(self):
        # 讀取文本檔案
        with open(self.path, 'r', encoding="utf-8") as f:
            input_text = f.read()
        # 使用 jieba 進行分詞
        words_sp = jieba.cut(input_text)
        # 過濾掉空白字符
        self.words_list = [word for word in words_sp if word.strip()]

    def process_text(self):
        # 處理文本，建立各種字典
        for i in range(len(self.words_list) - 2):
            word = self.words_list[i]
            nextword = self.words_list[i + 1]
            nextnextword = self.words_list[i + 2]

            # 更新每個字下一個字的字典
            self.update_dict(self.book, word, nextword)
            # 更新每個字下兩個字的字典
            self.update_dict(self.book2, word, nextnextword, is_second=True)
            # 更新每個字上一個字的字典
            self.update_dict(self.ibook, nextword, word, is_reverse=True)
            # 更新每個字上兩個字的字典
            self.update_dict(self.ibook2, nextnextword, word, is_reverse=True, is_second=True)

    def update_dict(self, target_dict, key, value, is_reverse=False, is_second=False):
        # 更新字典的方法
        if key not in target_dict:
            target_dict[key] = {}
        if value not in target_dict[key]:
            target_dict[key][value] = 1
        else:
            target_dict[key][value] += 1

class TextGenerator:
    def __init__(self, text_processor, start='同學', stop='。', stop_count=3):
        self.text_processor = text_processor
        self.start = start
        self.stop = stop
        self.stop_count = stop_count
        self.ans = start
        self.ptr = start

    def generate_text(self, max_steps=100000):
        # 生成文本的主方法
        for i in range(max_steps):
            if i < 1:
                # 第一步：生成下一個字
                self.step_one()
            else:
                # 第二步：根據機率生成下一個字
                self.step_two()
            if self.stop_count == 0:
                break
            else:
                self.stop_count -= 1
        # 輸出生成的文本
        print(self.ans)

    def step_one(self):
        # 根據當前字生成下一個字
        val = list(self.text_processor.book[self.ptr].values())
        X = np.cumsum(val)
        pick = np.random.randint(X[0], X[-1] + 1, size=10)[0]
        choose = np.where(pick <= X)[0][0]
        self.ans += list(self.text_processor.book[self.ptr].keys())[choose]
        self.ptr = list(self.text_processor.book[self.ptr].keys())[choose]

    def step_two(self):
        # 根據機率計算生成下一個字
        a = list(self.text_processor.book[self.ptr].keys())
        pa = []
        pda1 = []
        pda2 = []
        for j in range(len(a)):
            # 計算 P(A)
            pa.append(np.sum(list(self.text_processor.book[a[j]].values())) / len(self.text_processor.words_list))
            # 計算 P(d1|A)
            ikey = list(self.text_processor.ibook[a[j]].keys())
            ival = list(self.text_processor.ibook[a[j]].values())
            pda1.append(ival[np.argwhere(np.array(ikey) == self.ptr)[0][0]] / np.sum(list(ival)))
            # 計算 P(d2|A)
            ikey2 = list(self.text_processor.ibook2[a[j]].keys())
            ival2 = list(self.text_processor.ibook2[a[j]].values())
            try:
                pda2.append(ival2[np.argwhere(np.array(ikey2) == self.start)[0][0]] / np.sum(list(ival2)))
            except:
                pda2.append(0)

        pa = np.array(pa)
        pda1 = np.array(pda1)
        pda2 = np.array(pda2)
        pad = pa * pda1 * pda2
        index_pda = np.argmax(pad)
        self.ans += a[index_pda]
        self.ptr = a[index_pda]

# 範例使用
if __name__ == "__main__":
    path = 'data_set.txt'
    text_processor = TextProcessor(path)
    text_generator = TextGenerator(text_processor)
    text_generator.generate_text()

