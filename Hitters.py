import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import warnings

# 忽略警告信息，保持控制台整洁
warnings.filterwarnings('ignore')

# ================= 1. 数据加载与预处理 =================
print("--- 1. 数据加载与预处理 (使用模拟数据) ---")

# === 方案二：生成模拟数据 (无需联网) ===
np.random.seed(42)
n_samples = 322  # 模拟 Hitters 数据集的样本量

# 构造符合 Hitters 结构的字典
data = {
    'AtBat': np.random.randint(0, 700, n_samples),
    'Hits': np.random.randint(0, 250, n_samples),
    'HmRun': np.random.randint(0, 60, n_samples),
    'Runs': np.random.randint(0, 150, n_samples),
    'RBI': np.random.randint(0, 160, n_samples),
    'Walks': np.random.randint(0, 130, n_samples),
    'Years': np.random.randint(1, 25, n_samples),
    'CAtBat': np.random.randint(0, 10000, n_samples),
    'CHits': np.random.randint(0, 3000, n_samples),
    'CHmRun': np.random.randint(0, 500, n_samples),
    'CRuns': np.random.randint(0, 1800, n_samples),
    'CRBI': np.random.randint(0, 1800, n_samples),
    'CWalks': np.random.randint(0, 1500, n_samples),
    'League': np.random.choice(['A', 'N'], n_samples),
    'Division': np.random.choice(['E', 'W'], n_samples),
    'PutOuts': np.random.randint(0, 900, n_samples),
    'Assists': np.random.randint(0, 700, n_samples),
    'Errors': np.random.randint(0, 50, n_samples),
    'Salary': np.random.uniform(100, 2500, n_samples)  # 目标变量
}

# 随机制造一些空值，模拟真实情况
mask = np.random.choice([True, False], n_samples, p=[0.1, 0.9])
data['Salary'][mask] = np.nan

df = pd.DataFrame(data)
print("模拟数据生成成功！")
# ====================================

# 删除 Salary 为空的行
df = df.dropna(subset=['Salary'])

# 分离特征(X)和目标(y)
X = df.drop('Salary', axis=1)
y = np.log(df['Salary'])  # 对 Salary 进行对数变换，使其分布更接近正态

# 识别数值型和分类型特征
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # 数值特征标准化
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)  # 分类特征独热编码
    ]
)

# 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 拟合预处理器并转换数据
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 获取特征名称（用于绘图）
feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
print(f"处理后的特征数量: {X_train_processed.shape[1]}")
print("数据预处理完成。\n")


# ================= 2. 模型建立与评估 =================
print("--- 2. 模型建立与评估 ---")

# 定义 Alpha 搜索范围
alphas = np.logspace(-3, 3, 100)


def train_and_evaluate(model_cv, model_class, X_tr, y_tr, X_te, y_te, name, l1_ratio=None):
    print(f"正在训练 {name} ...")

    # 1. 使用 CV 模型寻找最佳 Alpha
    if name == "Elastic Net":
        model_cv.set_params(l1_ratio=l1_ratio)

    model_cv.fit(X_tr, y_tr)
    best_alpha = model_cv.alpha_

    # 2. 预测并计算 RMSE
    y_pred = model_cv.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))

    # 3. 统计非零系数
    non_zero = np.sum(model_cv.coef_ != 0)

    print(f"  -> 最佳 Alpha: {best_alpha:.4f}")
    print(f"  -> 测试集 RMSE: {rmse:.4f}")
    print(f"  -> 非零系数个数: {non_zero}\n")

    return rmse, non_zero, best_alpha, model_cv


# 定义三个模型
ridge_cv = RidgeCV(alphas=alphas, cv=10)
lasso_cv = LassoCV(alphas=alphas, cv=10, max_iter=10000, random_state=42)
enet_cv = ElasticNetCV(alphas=alphas, cv=10, max_iter=10000, random_state=42, l1_ratio=0.5)

# 运行训练
r_rmse, r_nz, r_alpha, r_model = train_and_evaluate(ridge_cv, Ridge, X_train_processed, y_train, X_test_processed, y_test, "Ridge")
l_rmse, l_nz, l_alpha, l_model = train_and_evaluate(lasso_cv, Lasso, X_train_processed, y_train, X_test_processed, y_test, "Lasso")
e_rmse, e_nz, e_alpha, e_model = train_and_evaluate(enet_cv, ElasticNet, X_train_processed, y_train, X_test_processed, y_test, "Elastic Net", l1_ratio=0.5)


# ================= 3. 绘制系数路径图 =================
print("--- 3. 正在绘制系数路径图 ---")


def plot_path(model_class, X, y, alphas, title, l1_ratio=None):
    coefs = []
    for a in alphas:
        if model_class == Ridge:
            m = model_class(alpha=a).fit(X, y)  # Ridge 不需要 max_iter
        elif model_class == ElasticNet:
            m = model_class(alpha=a, l1_ratio=l1_ratio, max_iter=10000).fit(X, y)
        else:  # Lasso
            m = model_class(alpha=a, max_iter=10000).fit(X, y)
        coefs.append(m.coef_)

    plt.figure(figsize=(10, 6))
    plt.plot(np.log(alphas), coefs)
    plt.xlabel('log(alpha)')
    plt.ylabel('Coefficients')
    plt.title(title)
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)  # 标记 log(alpha)=0 的位置
    plt.grid(True, alpha=0.3)

    # 保存图片到本地
    filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"图片已保存: {filename}")
    plt.show()  # 尝试弹窗显示


# 分别绘制三张图
plot_path(Ridge, X_train_processed, y_train, alphas, 'Ridge Path')
plot_path(Lasso, X_train_processed, y_train, alphas, 'Lasso Path')
plot_path(ElasticNet, X_train_processed, y_train, alphas, 'Elastic Net Path', l1_ratio=0.5)


# ================= 4. 1-SE 法则讨论 =================
print("\n--- 4. 1-SE 法则讨论 (以 Lasso 为例) ---")

# 获取 Lasso CV 的详细路径信息
mse_mean = l_model.mse_path_.mean(axis=1)
mse_std = l_model.mse_path_.std(axis=1) / np.sqrt(10)  # 计算标准误

# 找到最小 MSE 的索引
min_idx = np.argmin(mse_mean)
min_mse = mse_mean[min_idx]

# 计算阈值：最小 MSE + 1个标准误
threshold = min_mse + mse_std[min_idx]

# 找到满足条件（MSE < 阈值）的最大 Alpha 索引（即最稀疏的模型）
valid_indices = np.where(mse_mean <= threshold)[0]
one_se_idx = valid_indices[-1]

best_alpha_1se = l_model.alphas_[one_se_idx]

# 用这个 Alpha 重新拟合模型查看系数
lasso_1se_model = Lasso(alpha=best_alpha_1se, max_iter=10000).fit(X_train_processed, y_train)
nnz_1se = np.sum(lasso_1se_model.coef_ != 0)

print(f"原始最优 Alpha: {l_alpha:.4f} (非零系数: {l_nz})")
print(f"1-SE 推荐 Alpha: {best_alpha_1se:.4f} (非零系数: {nnz_1se})")

if nnz_1se < l_nz:
    print("结论：根据 1-SE 法则，推荐选择更稀疏的模型（变量更少），且性能损失在允许范围内。")
else:
    print("结论：1-SE 法则未选出更稀疏的模型，建议使用原始最优模型。")

print("\n程序运行结束！请查看上方输出的图片和结论。")