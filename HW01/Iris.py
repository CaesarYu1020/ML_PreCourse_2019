import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression


def linearTest(x, y):
    # 轉成陣列
    x = np.array(x).reshape(5, 2)
    y = np.array(y).reshape(len(y), 1)
    clf = LinearRegression() #SET MODEL
    clf.fit(x, y) #TRAINING
    pre = clf.predict(x) #訓練完後就可以預測
    # 畫圖
    # plt.scatter(x[0], y, s=100)
    # plt.plot(x, pre, "r-", linewidth=4)
    # for idx, m in enumerate(x):
    #     plt.plot([m, m], [y[idx], pre[idx]], 'g-')
    # plt.show()
    # print("係數", clf.coef_)
    # print("截距", clf.intercept_)
    print(np.mean(y - pre) ** 2)
    # 係數 [[-0.05726823]]
    # 截距 [ 3.38863738]
    # 1.91991214088e-31
    # print(clf.predict([[5.0]]))
    # [[ 3.10229621]]


def linearIris():
    # hua = load_iris()
    # # 獲取花瓣的長和寬
    # x = [n[0] for n in hua.data]
    # y = [n[1] for n in hua.data]
    # linearTest(x, y)
    x = np.array([[1, 1], [1, 2], [2, 2], [2, 3],[4,5]])
    t=np.array([1,1,2,2,4])
    print(t)
    x1 = t.transpose()
    print(x)
    print(x1)
    y = [1, 5, 9, 12, 14]
    y = np.array([1,2,3,4,5]).reshape(len(y), 1)
    print(y)
    # linearTest(x, y)
    # x = np.random.rand(100).astype(np.float32)
    # y = x * 0.1 + 0.3
    # linearTest(x, y)


def main():
    linearIris()


main()