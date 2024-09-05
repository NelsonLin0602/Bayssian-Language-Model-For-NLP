import numpy as np
import jieba

# 文字處理器，建立好文字頻率資料庫
class TextProcessor:
    def __init__(self, path):
        self.path = path
        self.words_list = []
        self.book = {}  # 記錄每個字及其下一個字的關聯
        self.book2 = {}  # 記錄每個字及其後兩個字的關聯
        self.ibook = {}  # 記錄每個字及其上一個字的關聯
        self.ibook2 = {}  # 記錄每個字及其前兩個字的關聯
        self.load_text()
        self.process_text()

    # 處理資料文字檔案
    def load_text(self):
        # 讀取資料檔案
        with open(self.path, 'r', encoding="utf-8") as f:
            input_text = f.read()
        # 使用 jieba 進行分詞/斷詞
        words_sp = jieba.cut(input_text)  # list of string
        # 過濾掉空白字串
        self.words_list = [word for word in words_sp if word.strip()]  # list of string

    # 建立所有字典/文字資料庫
    def process_text(self):
        for idx in range(len(self.words_list) - 2):

            # 定義每個字
            word = self.words_list[idx]
            nextword = self.words_list[idx + 1]
            nextnextword = self.words_list[idx + 2]

            # 更新每個字下一個字的字典
            self.update_dict(self.book, word, nextword)
            # 更新每個字下兩個字的字典
            self.update_dict(self.book2, word, nextnextword, is_second=True)
            # 更新每個字上一個字的字典
            self.update_dict(self.ibook, nextword, word, is_reverse=True)
            # 更新每個字上兩個字的字典
            self.update_dict(self.ibook2, nextnextword, word, is_reverse=True, is_second=True)

    # 計算每個字後或前個字出現的機率
    def update_dict(self, target_dict, key, value, is_reverse=False, is_second=False):
        # 更新字典的方法
        if key not in target_dict:
            target_dict[key] = {}
        if value not in target_dict[key]:
            target_dict[key][value] = 1
        else:
            target_dict[key][value] += 1

# 文字產生器
class TextGenerator:
    def __init__(self, text_processor, start='同學', stop='。', stop_count=20):
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
        X = np.cumsum(val)  # 計算累積和 (list of int)
        pick = np.random.randint(X[0], X[-1] + 1, size=10)[0]  # 將累積何的範圍內隨機取10個數字的第一項 (list of int->int)
        choose = np.where(pick <= X)[0][0]  # 選擇累積和小於等於隨機數字的第一項 (list of list of int->int)
        self.ans += list(self.text_processor.book[self.ptr].keys())[choose]
        self.ptr = list(self.text_processor.book[self.ptr].keys())[choose]

    def step_two(self):
        # 根據機率計算生成下一個字
        a = list(self.text_processor.book[self.ptr].keys())  # 下一個字有可能的list
        pa = []  # 每個字在整個檔案出現的頻率
        pda1 = []  # 前一個字生成當前字的機率
        pda2 = []  # 前兩個字生成當前字的機率
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
        index_pda = np.argmax(pad)  # 計算最大機率的index
        self.ans += a[index_pda]
        self.ptr = a[index_pda]

# 範例使用
if __name__ == "__main__":
    path = 'data_set.txt'
    text_processor = TextProcessor(path)
    text_generator = TextGenerator(text_processor)
    text_generator.generate_text()

