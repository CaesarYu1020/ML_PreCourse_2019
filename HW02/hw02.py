# import pandas as pd

# df=pd.DataFrame([])
#df.loc(row,col) ('tag'去找)
#df.iloc(row,col) (index 去找)
#scatter 散步圖
#線性回歸 類神經網路 黑箱
#決策樹 Association rules 白箱


from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import pydotplus
import pandas as pd
import matplotlib.pyplot as plt
import time
#pip install graphviz
#pip install pydotplus
def drawTree(my_df, clf):
#sklearn.tree.export_graphviz(decision_tree, out_file=None,
#max_depth=None, feature_names=None, class_names=None,
#label=’all’, filled=False, leaves_parallel=False,
#impurity=True, node_ids=False, proportion=False,
#rotate=False, rounded=False, special_characters=False, precision=3)
#https://scikit-learn.org/…/sklearn.tree.export_graphviz.html
#將 Tree導出為 graphviz
    temp=my_df.drop(['Play Tennis'],axis=1)
    dot_data = tree.export_graphviz(clf, out_file=None,
    feature_names=['Outlook','Temp','Humidity','Wind'],
    class_names=['Yes','No'],
    filled=True, rounded=True,
    special_characters=True)
    #使用 pydotplus 產生 pdf 檔案
    graph = pydotplus.graph_from_dot_data(dot_data)
    try:
        graph.write_pdf('C:\\Users\\CaesarYu\\Desktop\\ML_PreCourse\\hw2test.pdf')
    except:
        print('FAIL')
def decisionTree():
# 讀入鳶尾花資料
#     iris = load_iris()
#     iris_X = iris.data
#     iris_y = iris.target
#     print(type(iris.data)) # 資料儲存為 ndarray
#     print(iris.feature_names) # 變數名稱可以利用 feature_names 屬性取得
    mydata=[
            ['Sunny','Hot','High','Weak','No'],
            ['Sunny','Hot','High','Strong','No'],
            ['Overcast','Hot','High','Weak','Yes'],
            ['Rain','Mild','High','Weak','Yes'],
            ['Rain','Cool','Normal','Weak','Yes'],
            ['Rain','Cool','Normal','Strong','No'],
            ['Overcast','Cool','Normal','Weak','Yes'],
            ['Sunny','Mild','High','Weak','No'],
            ['Sunny','Cold','Normal','Weak','Yes'],
            ['Rain','Mild','Normal','Strong','Yes']
            ]


    mydata2 = [
        [1, 1, 1, 1, 2],
        [1, 1, 1, 2, 2],
        [3, 1, 1, 1, 1],
        [2, 2, 1, 1, 1],
        [2, 3, 2, 1, 1],
        [2, 3, 2, 2, 2],
        [3, 3, 2, 1, 1],
        [1, 2, 1, 1, 2],
        [1, 4, 2, 1, 1],
        [2, 2, 2, 2, 1]
    ]
    my_df = pd.DataFrame(mydata2, columns=['Outlook','Temp','Humidity','Wind','Play Tennis'])
    # 轉換為 data frame
    print(my_df) # 觀察前五個觀測值
    #print(iris_df) # 觀察所有觀測值
    temp=my_df.drop(['Play Tennis'],axis=1)
    print(temp.get_values())
    # 切分訓練與測試資料
    train_X, test_X, train_y, test_y = train_test_split(temp.get_values(), my_df.get('Play Tennis'), test_size = 0.3)
    print(train_X)


    my_df.hist()  # 劃出直方圖

    # 劃出散佈圖

    # x,y是col tag!

    # Kernel Density Estimates 核心密度估計圖，直方圖的平滑機率分布圖
    my_df.plot.kde()


    plt.show()

    # 建立分類器
    clf = tree.DecisionTreeClassifier()
    #訓練資料
    my_clf = clf.fit(train_X, train_y)
    # 預測資料
    test_y_predicted = my_clf.predict(test_X)
    print(test_y_predicted)
    # 標準答案
    print(test_y)
    # 績效 - 精確度
    accuracy = metrics.accuracy_score(test_y, test_y_predicted)
    print(accuracy)
    drawTree(my_df, my_clf)

def testPlot():
    df = pd.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
    [6.4, 3.2, 1], [5.9, 3.0, 2]],
    columns=['length', 'width', 'species'])
    #df.plot.scatter(x='length', y='width', c='DarkBlue')
    print(df.loc[0:1,['length', 'width']])
    print(df.iloc[0:1,[0,1]])

def main():
    # testPlot()
    decisionTree()

main()