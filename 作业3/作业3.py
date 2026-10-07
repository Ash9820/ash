import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 数据加载与预处理 =================
print("--- 1. 数据加载与预处理 ---")

# 读取数据集
try:
    # 假设你已经把文件重命名为 data.csv 并放在了同目录下
    df = pd.read_csv('data.csv')
    print("数据加载成功！")
except Exception as e:
    print(f"读取文件失败: {e}")
    print("请确保 data.csv 文件在当前目录下。")
    raise SystemExit("程序终止。")

# 查看数据基本信息
print(f"数据集维度: {df.shape}")
print(f"当前包含的列名: {list(df.columns)}")

# --- 核心修改部分：智能匹配目标变量 ---
# 预设的目标变量名
# 假设列名就是 Watch_Hours
target_col = 'Watch_Hours'

# 检查目标列是否存在，如果不存在，尝试自动寻找替代列
if target_col not in df.columns:
    print(f"\n⚠️ 警告: 未找到预设列 '{target_col}'。")
    print("正在尝试自动匹配类似的列名...")

    # 常见的观看时长列名列表
    possible_targets = ['watch_time', 'total_watch_time', 'viewing_time', 'duration', 'minutes_watched']
    found_target = None

    for col in possible_targets:
        if col in df.columns:
            found_target = col
            break

    if found_target:
        print(f"✅ 找到替代目标变量: '{found_target}'")
        target_col = found_target
    else:
        print("❌ 无法自动匹配目标变量，请手动检查上方打印的列名列表。")
        print("请在代码中修改 target_col 的值。")
        raise SystemExit("程序终止。")

# 提取目标变量 (y)
y = df[target_col]

# 提取特征变量 (X)
# 排除 ID、分类变量（如性别、国家）和目标变量本身
exclude_cols = ['user_id', 'gender', 'country', 'subscription_type', 'plan_type', target_col]
# 动态排除：只保留在 df 中实际存在的排除项
exclude_cols = [c for c in exclude_cols if c in df.columns]

# 筛选数值型特征
numerical_features = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

X = df[numerical_features]
print(f"筛选出的数值型特征 ({len(numerical_features)}个): {numerical_features}")

# 处理缺失值（中位数填充）
X.fillna(X.median(), inplace=True)
y.fillna(y.median(), inplace=True)

# ================= 2. 划分数据集与 Z-score 标准化 =================
print("\n--- 2. 划分数据集与标准化 ---")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集样本数: {X_train_scaled.shape[0]}, 特征数: {X_train_scaled.shape[1]}")

# ================= 3. 构建 Elastic Net 模型 =================
print("\n--- 3. 训练 Elastic Net 模型 ---")

enet_cv = ElasticNetCV(
    l1_ratio=np.linspace(0.1, 1.0, 10),
    alphas=np.logspace(-4, 2, 50),
    cv=5,  # 改为5折加快速度，如果数据量大可改回10
    max_iter=10000,
    random_state=42,
    n_jobs=-1
)

enet_cv.fit(X_train_scaled, y_train)

# ================= 4. 模型评估与结果解读 =================
print("\n--- 4. 模型评估与结果解读 ---")

print(f"最优 Alpha (正则化强度): {enet_cv.alpha_:.4f}")
print(f"最优 L1 Ratio (L1占比): {enet_cv.l1_ratio_:.2f}")

y_pred = enet_cv.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"测试集 RMSE: {rmse:.2f}")
print(f"测试集 R² Score: {r2:.4f}")

# 特征重要性分析
coef_df = pd.DataFrame({
    'Feature': numerical_features,
    'Coefficient': enet_cv.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\n特征重要性排名 (按系数绝对值排序):")
print(coef_df)

zero_coefs = np.sum(enet_cv.coef_ == 0)
print(f"\n被 Elastic Net 剔除 (系数为0) 的特征数量: {zero_coefs} / {len(numerical_features)}")

# ================= 5. 可视化 =================
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='steelblue')
plt.xlabel('Coefficient Value')
plt.title(f'Elastic Net Feature Importance (Target: {target_col})')
plt.axvline(0, color='black', linestyle='--', alpha=0.3)
plt.tight_layout()

plt.savefig('netflix_elastic_net_importance.png', dpi=150, bbox_inches='tight')
print("\n特征重要性图已保存为: netflix_elastic_net_importance.png")
plt.show()

print("\n程序运行结束！")