import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==========================================
# 1. 数据加载
# ==========================================
# 尝试从 GitHub 下载
url_github = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/ISLR/Carseats.csv"
# 备用源 (如果 GitHub 连不上)
url_backup = "https://gitee.com/quanwei9958/ISLR-Python/raw/main/Data/Carseats.csv"

print("正在加载数据...")
try:
    df = pd.read_csv(url_github)
except Exception:
    print("GitHub 连接失败，尝试备用源...")
    try:
        df = pd.read_csv(url_backup)
    except Exception:
        print("备用源也失败。正在使用 statsmodels 内置数据...")
        df = sm.datasets.get_rdataset("Carseats", "ISLR").data

print(f"数据加载成功，形状: {df.shape}")

# ==========================================
# 2. 建立回归模型
# ==========================================

formula = "Sales ~ Price + Income + Advertising + C(ShelveLoc)"
model = ols(formula, data=df).fit()

print("\n" + "="*30 + " 模型摘要 " + "="*30)
print(model.summary())

# ==========================================
print("\n" + "="*30 + " VIF 分析 " + "="*30)

# 获取模型使用的自变量矩阵 (exog)
X = model.model.exog
var_names = model.model.exog_names

# 创建 VIF DataFrame
vif_data = pd.DataFrame()
vif_data["Variable"] = var_names
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

# 过滤掉截距项 (Intercept) 的 VIF
vif_data = vif_data[vif_data["Variable"] != "Intercept"]

print(vif_data)