import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd

# def drawFeatures():
#     boston = load_boston()
#     boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)
#     y = boston.target
#     featurePlot=plt.figure()
#     t=1;
#     for tag in boston.feature_names:
#         x=boston_data.get(tag)
#         plt.subplot(4,4,t)
#         t=t+1;
#         plt.scatter(x, y)
#         plt.xlabel(tag)
#         plt.ylabel('price')
# plt.plot([2, 3], [4,6], 'g-',linewidth=1)
# plt.show()
boston = load_boston()
boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)
x=boston_data['RM']
print(x[0:10])