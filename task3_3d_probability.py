import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.cm as cm

# 加载数据并选择两个类别
iris = load_iris()
X = iris.data[iris.target != 2]  # 移除Virginica
y = iris.target[iris.target != 2]

# 选择三个特征
X = X[:, :3]
y = y

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 训练分类器（使用probability=True）
clf = SVC(kernel='rbf', C=1.0, probability=True)
clf.fit(X_train, y_train)

# 创建图形
fig = plt.figure(figsize=(20, 10))

# 创建网格点
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
z_min, z_max = X_scaled[:, 2].min() - 1, X_scaled[:, 2].max() + 1

# 创建3D概率图
for view_idx, (elev, azim) in enumerate([(30, 45), (60, 120)]):
    ax = fig.add_subplot(1, 2, view_idx + 1, projection='3d')

    # 创建网格
    grid_resolution = 20
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                             np.linspace(y_min, y_max, grid_resolution),
                             np.linspace(z_min, z_max, grid_resolution))

    # 预测每个网格点的概率
    X_grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    probabilities = clf.predict_proba(X_grid)[:, 1]  # Class 1的概率
    probabilities = probabilities.reshape(xx.shape)

    # 绘制3D概率云
    # 使用透明度显示概率密度
    threshold = 0.1  # 只显示概率大于阈值的点

    for level in np.linspace(0.1, 0.9, 9):
        mask = (probabilities > level - 0.05) & (probabilities < level + 0.05)

        if np.any(mask):
            # 提取mask为True的点的坐标和概率值
            points = np.argwhere(mask)
            x_points = xx[mask]
            y_points = yy[mask]
            z_points = zz[mask]
            prob_values = probabilities[mask]

            # 根据概率值设置颜色和透明度
            colors = cm.RdYlBu(prob_values)
            ax.scatter(x_points, y_points, z_points,
                       c=colors, alpha=0.2, s=20, edgecolors='none')

    # 绘制原始数据点
    colors_actual = ['red' if label == 0 else 'blue' for label in y]
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
               c=colors_actual, s=100, edgecolors='k', alpha=0.8, label='Data Points')

    ax.set_xlabel('Sepal Length (Standardization)')
    ax.set_ylabel('Sepal Width (Standardization)')
    ax.set_zlabel('Petal Length (Standardization)')
    ax.set_title(
        f'3D probability graph visualization\nPerspective: elev={elev}, azim={azim}')
    ax.view_init(elev=elev, azim=azim)

    # 添加颜色条
    mappable = cm.ScalarMappable(cmap=cm.RdYlBu)
    mappable.set_array(probabilities.ravel())
    fig.colorbar(mappable, ax=ax, shrink=0.5,
                 aspect=5, label='Class 1 Probability')

plt.tight_layout()
plt.savefig('task3_3d_probability_map.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建交互式概率图（使用plotly）
# 创建网格
grid_resolution = 15
xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                         np.linspace(y_min, y_max, grid_resolution),
                         np.linspace(z_min, z_max, grid_resolution))

X_grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
probabilities = clf.predict_proba(X_grid)[:, 1]
probabilities = probabilities.reshape(xx.shape)

# 创建3D体积图
fig = go.Figure(data=go.Volume(
    x=xx.flatten(),
    y=yy.flatten(),
    z=zz.flatten(),
    value=probabilities.flatten(),
    isomin=0.1,
    isomax=0.9,
    opacity=0.1,
    surface_count=20,
    colorscale='RdBu',
    caps=dict(x_show=False, y_show=False, z_show=False)
))

# 添加数据点
fig.add_trace(go.Scatter3d(
    x=X_scaled[:, 0],
    y=X_scaled[:, 1],
    z=X_scaled[:, 2],
    mode='markers',
    marker=dict(
        size=8,
        color=y,
        colorscale='RdBu',
        line=dict(color='black', width=1)
    ),
    text=[f'Class {label}' for label in y],
    name='Data Points'
))

fig.update_layout(
    title='交互式3D概率图（两分类/三个特征）',
    scene=dict(
        xaxis_title='Sepal Length',
        yaxis_title='Sepal Width',
        zaxis_title='Petal Length'
    ),
    width=1000,
    height=800
)

# 保存为html文件
fig.write_html("task3_interactive_probability_map.html")
print("交互式3D概率图已保存为 'task3_interactive_probability_map.html'")
