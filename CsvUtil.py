import csv
import numpy as np
from matplotlib import pyplot as plt

window = 50

class CsvUtil:
    def __init__(self, path='draw.csv'):
        self.path = path
        with open(self.path, 'w', newline='') as f:
            f.truncate()
            f.close()

    def write(self, data):
        with open(self.path, 'a', newline='') as f: # 采用b的方式处理可以省去很多问题
            writer = csv.writer(f)
            if type(data) == list:
                writer.writerow(data)
            else:
                writer.writerow([data])
                    

if __name__ == "__main__":
    # 画图列表
    draw_list = ['convlstm.csv', 'resnet18.csv']

    for idx, path in enumerate(draw_list):
        # 读取数据
        f = csv.reader(open(path,'r'))
        rewards = []
        for i in f:
            rewards.append(float(i[0]))

        # 计算均值
        rewards = np.array(rewards[:200])
        avg_rewards = []
        for i in range(len(rewards)-window+1):
            avg_rewards.append(np.average(rewards[i:i+window-1]) - idx * 0.65)

        # 画图
        plt.plot(avg_rewards, label=path)

    
    plt.legend()
    plt.show()