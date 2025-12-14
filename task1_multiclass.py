import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入不同的分类器
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 加载数据
iris = load_iris()
X = iris.data[:, :2]  # 使用前两个特征（Sepal Length和Sepal Width）用于可视化
y = iris.target

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 定义要比较的分类器
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM (Linear)': SVC(kernel='linear', probability=True),
    'SVM (RBF)': SVC(kernel='rbf', probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(max_depth=3),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

# 创建图形
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    'Iris Classifier Comparison (Three Classifications/Two Features)', fontsize=16)

# 创建网格用于决策边界
xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1, 300),
                     np.linspace(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1, 300))

# 为每个分类器绘制决策边界
for idx, (name, clf) in enumerate(classifiers.items()):
    row, col = divmod(idx, 3)
    ax = axes[row, col]

    # 训练分类器
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    # 预测整个网格
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # 绘制训练数据点
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y,
                         edgecolors='k', s=50, cmap=plt.cm.coolwarm)

    ax.set_xlabel('Sepal Length (Standardization)')
    ax.set_ylabel('Sepal Width (Standardization)')
    ax.set_title(f'{name}\nAccuracy rate: {accuracy:.2f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_classifiers_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印每个分类器的准确率
print("\n=== 分类器性能比较 ===")
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"{name}: 训练集准确率={train_acc:.4f}, 测试集准确率={test_acc:.4f}")
