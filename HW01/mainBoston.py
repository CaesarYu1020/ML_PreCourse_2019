import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pandas as pd

def drawFeatures():
    boston = load_boston()
    boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)

    y = boston.target
    plt.figure('各特徵與價錢關係')
    t=1;
    for tag in boston.feature_names:
        x=boston_data.get(tag)
        plt.subplot(4,4,t)
        t=t+1;
        plt.scatter(x, y,s=2)
        plt.xlabel(tag)
        plt.ylabel('price')
    plt.subplots_adjust(left=0.11, bottom=0.06, right=0.94, top=0.91, wspace=0.59, hspace=0.75)
def linearTest(x, y,draw=False):
    # 轉成陣列
    if draw==True:
        drawFeatures()
    x = np.array(x).reshape(len(x), 1)
    y = np.array(y).reshape(len(y), 1)
    clf = LinearRegression() #SET MODEL
    clf.fit(x, y) #TRAINING
    pre = clf.predict(x) #訓練完後就可以預測
    # 畫圖
    plt.figure('預測與實際結果對照')
    plt.scatter(x, y, s=20)
    for idx, m in enumerate(x):
        plt.plot([m, m], [y[idx], pre[idx]], 'g-',linewidth=1) # x,y and x,pre 截距
    plt.plot(x, pre, "r-", linewidth=3)
    plt.xlabel('RM')
    plt.ylabel('Price')

    print("係數", clf.coef_)
    print("截距", clf.intercept_)
    print(clf.score(x,y))
    print(np.mean(y - pre) ** 2)
    # 係數 [[-0.05726823]]
    # 截距 [ 3.38863738]
    # 1.91991214088e-31
    print(clf.predict([[5.0]]))
    # [[ 3.10229621]]

def linearBoston(draw=False):
    boston = load_boston()
    boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)
    x = boston_data.get('RM') # 我選擇RM和PRICE之間的關係  P.S. RM:每棟住宅的平均房間數
    y = boston.target #boston.target: price (MEDV)
    if draw==True: #是否劃出特徵關係圖
        linearTest(x, y,True)
    else:
        linearTest(x, y)
def main():
    plt.close('all')
    linearBoston(True)
    plt.show()
main()