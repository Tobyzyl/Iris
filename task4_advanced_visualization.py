from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.cm as cm

# 加载数据并选择两个类别和三个特征
iris = load_iris()
X = iris.data[iris.target != 2, :3]  # 移除第三类，选择前三个特征
y = iris.target[iris.target != 2]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 尝试多种分类器
classifiers = {
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
}

# 评估分类器性能
print("=== 分类器性能比较 ===")
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_scaled, y, cv=5)
    print(f"{name}: 交叉验证准确率 = {scores.mean():.4f} (±{scores.std():.4f})")

# 使用最佳分类器（这里选择SVM）
best_clf = SVC(kernel='rbf', C=1.0, probability=True)
best_clf.fit(X_train, y_train)

# 创建高级可视化图形
fig = plt.figure(figsize=(24, 12))

# 1. 3D决策边界与概率图结合
ax1 = fig.add_subplot(231, projection='3d')

# 创建网格
grid_resolution = 20
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
z_min, z_max = X_scaled[:, 2].min() - 1, X_scaled[:, 2].max() + 1

xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                         np.linspace(y_min, y_max, grid_resolution),
                         np.linspace(z_min, z_max, grid_resolution))

X_grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
probabilities = best_clf.predict_proba(X_grid)[:, 1]
probabilities = probabilities.reshape(xx.shape)

# 绘制决策边界等值面（概率=0.5）

verts, faces, _, _ = measure.marching_cubes(
    probabilities, level=0.5, spacing=(1, 1, 1))
verts[:, 0] = verts[:, 0] * (x_max - x_min) / (grid_resolution - 1) + x_min
verts[:, 1] = verts[:, 1] * (y_max - y_min) / (grid_resolution - 1) + y_min
verts[:, 2] = verts[:, 2] * (z_max - z_min) / (grid_resolution - 1) + z_min

ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                 alpha=0.5, color='gray')


# 绘制数据点
colors_actual = ['red' if label == 0 else 'blue' for label in y]
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
            c=colors_actual, s=100, edgecolors='k', alpha=0.8)

ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Sepal Width')
ax1.set_zlabel('Petal Length')
ax1.set_title('3D decision boundary (isosurface with probability =0.5)')
ax1.view_init(elev=30, azim=45)

# 2. 3D概率图（透明度表示概率）
ax2 = fig.add_subplot(232, projection='3d')

# 绘制概率云
for level in np.linspace(0.2, 0.8, 4):
    mask = (probabilities > level - 0.1) & (probabilities < level + 0.1)
    if np.any(mask):
        x_points = xx[mask]
        y_points = yy[mask]
        z_points = zz[mask]
        prob_values = probabilities[mask]

        colors = cm.RdYlBu(prob_values)
        ax2.scatter(x_points, y_points, z_points,
                    c=colors, alpha=0.2, s=20, edgecolors='none')

# 绘制数据点
ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
            c=colors_actual, s=100, edgecolors='k', alpha=0.8)

ax2.set_xlabel('Sepal Length')
ax2.set_ylabel('Sepal Width')
ax2.set_zlabel('Petal Length')
ax2.set_title('3D probability cloud map')
ax2.view_init(elev=30, azim=45)

# 3. 2D投影：特征对概率图
feature_pairs = [(0, 1), (0, 2), (1, 2)]
subplot_positions = [233, 234, 235]  # 注意：这里的233,234,235是子图编号，不是数值233

for i, (feat1, feat2) in enumerate(feature_pairs):
    ax = fig.add_subplot(subplot_positions[i])

    # 创建2D网格
    xx_2d, yy_2d = np.meshgrid(
        np.linspace(X_scaled[:, feat1].min() - 1,
                    X_scaled[:, feat1].max() + 1, 100),
        np.linspace(X_scaled[:, feat2].min() - 1,
                    X_scaled[:, feat2].max() + 1, 100)
    )

    # 使用特征平均值填充第三维
    X_grid_2d = np.zeros((xx_2d.ravel().shape[0], 3))
    X_grid_2d[:, feat1] = xx_2d.ravel()
    X_grid_2d[:, feat2] = yy_2d.ravel()
    # 计算第三个特征的索引
    third_feat = 3 - feat1 - feat2 - 1  # 修正索引计算
    X_grid_2d[:, third_feat] = X_scaled[:, third_feat].mean()

    prob_2d = best_clf.predict_proba(X_grid_2d)[:, 1]
    prob_2d = prob_2d.reshape(xx_2d.shape)

    # 绘制2D概率图
    contour = ax.contourf(xx_2d, yy_2d, prob_2d,
                          alpha=0.8, cmap=cm.RdYlBu, levels=20)

    # 绘制数据点
    ax.scatter(X_scaled[:, feat1], X_scaled[:, feat2], c=y,
               edgecolors='k', s=50, cmap=cm.RdYlBu)

    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length']
    ax.set_xlabel(feature_names[feat1])
    ax.set_ylabel(feature_names[feat2])
    ax.set_title(
        f'{feature_names[feat1]} vs {feature_names[feat2]} Probability graph')
    plt.colorbar(contour, ax=ax, label='Class 1 Probability')

plt.suptitle(
    'Advanced Visualization: 3D Decision Boundary + Probabilistic Graph + Feature Analysis', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('task4_advanced_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 生成综合报告
print("\n" + "="*50)
print("实验总结报告")
print("="*50)
print(f"数据集: Iris (两分类)")
print(f"使用的特征: Sepal Length, Sepal Width, Petal Length")
print(f"分类器: SVM with RBF kernel")
print(f"测试集准确率: {best_clf.score(X_test, y_test):.4f}")
print(f"决策函数类型: {'非线性' if best_clf.kernel != 'linear' else '线性'}")
print(f"支持向量数量: {len(best_clf.support_vectors_)}")
print("="*50)
