from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 加载数据并选择两个类别（Setosa和Versicolor）
iris = load_iris()
X = iris.data[iris.target != 2]  # 移除Virginica，只保留两个类别
y = iris.target[iris.target != 2]

# 选择三个特征（Sepal Length, Sepal Width, Petal Length）
X = X[:, :3]  # 前三个特征
y = y  # 两个类别

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 使用SVM分类器（适合非线性决策边界）
clf = SVC(kernel='rbf', C=1.0, probability=True)
clf.fit(X_train, y_train)

# 创建3D图形
fig = plt.figure(figsize=(16, 8))

# 子图1：决策边界3D可视化
ax1 = fig.add_subplot(121, projection='3d')

# 创建网格点
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
z_min, z_max = X_scaled[:, 2].min() - 1, X_scaled[:, 2].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                     np.linspace(y_min, y_max, 20))

# 为每个网格点预测类别
for i, z_val in enumerate(np.linspace(z_min, z_max, 5)):
    zz = np.ones(xx.shape) * z_val
    X_grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = clf.predict(X_grid)
    Z = Z.reshape(xx.shape)

    # 绘制决策边界表面
    ax1.plot_surface(xx, yy, zz, alpha=0.1,
                     facecolors=plt.cm.coolwarm(Z/1.0))

# 绘制数据点
colors = ['red', 'blue']
for i in range(2):
    ax1.scatter(X_scaled[y == i, 0], X_scaled[y == i, 1],
                X_scaled[y == i, 2],
                c=colors[i], label=f'Class {i}', s=50, alpha=0.8)

ax1.set_xlabel('Sepal Length (Standardization)')
ax1.set_ylabel('Sepal Width (Standardization)')
ax1.set_zlabel('Petal Length (Standardization)')
ax1.set_title(
    '3D decision boundary visualization\n(Two categories/three features)')
ax1.legend()

# 子图2：从不同角度观察决策边界
ax2 = fig.add_subplot(122, projection='3d')

# 创建更密集的网格用于绘制光滑边界
xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 15),
                         np.linspace(y_min, y_max, 15),
                         np.linspace(z_min, z_max, 15))

X_grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
Z = clf.predict(X_grid)
Z = Z.reshape(xx.shape)

# 绘制3D决策边界（使用等值面）
verts, faces, _, _ = measure.marching_cubes(Z, level=0.5, spacing=(1, 1, 1))
verts[:, 0] = verts[:, 0] * (x_max - x_min) / 14 + x_min
verts[:, 1] = verts[:, 1] * (y_max - y_min) / 14 + y_min
verts[:, 2] = verts[:, 2] * (z_max - z_min) / 14 + z_min

ax2.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                 alpha=0.3, color='green')

# 绘制数据点
for i in range(2):
    ax2.scatter(X_scaled[y == i, 0], X_scaled[y == i, 1],
                X_scaled[y == i, 2],
                c=colors[i], label=f'Class {i}', s=50, alpha=0.8)

ax2.set_xlabel('Sepal Length (Standardization)')
ax2.set_ylabel('Sepal Width (Standardization)')
ax2.set_zlabel('Petal Length (Standardization)')
ax2.set_title(
    '3D decision boundary isosurface\n(Two categories/three features)')
ax2.view_init(elev=30, azim=45)
ax2.legend()

plt.tight_layout()
plt.savefig('task2_3d_boundary.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印分类器性能
print(f"SVM分类器准确率: {clf.score(X_test, y_test):.4f}")
