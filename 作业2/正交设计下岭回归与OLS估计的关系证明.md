# 正交设计下岭回归与OLS估计的关系证明

## 1. 核心公式回顾

- **最小二乘估计 (OLS)**：
$$
\widehat{\beta}_{\text{OLS}} = (X^T X)^{-1} X^T Y
$$

- **岭回归估计 (Ridge)**：
$$
\widehat{\beta}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T Y \quad (\lambda > 0 \text{ 为正则化参数})
$$

## 2. 正交设计条件

正交设计下，设计矩阵 $X$ 满足：
$$
X^T X = I \quad (\text{单位矩阵})
$$

## 3. 代入正交条件推导

### 步骤1：计算 OLS 估计量
将 $X^T X = I$ 代入 OLS 公式：
$$
\widehat{\beta}_{\text{OLS}} = (I)^{-1} X^T Y = I \cdot X^T Y = X^T Y
$$

### 步骤2：计算 Ridge 估计量
将 $X^T X = I$ 代入岭回归公式：
$$
\widehat{\beta}_{\text{Ridge}} = (I + \lambda I)^{-1} X^T Y
$$

利用矩阵求逆性质 $(cI)^{-1} = \frac{1}{c}I$ ($c$ 为标量)，化简 $I + \lambda I$：
$$
I + \lambda I = (1 + \lambda)I \implies (I + \lambda I)^{-1} = \frac{1}{1 + \lambda}I
$$

因此，岭回归估计量可化简为：
$$
\widehat{\beta}_{\text{Ridge}} = \frac{1}{1 + \lambda}I \cdot X^T Y = \frac{1}{1 + \lambda} X^T Y
$$

### 步骤3：建立两者关系
由步骤1知 $\widehat{\beta}_{\text{OLS}} = X^T Y$，将其代入步骤2的结果：
$$
\widehat{\beta}_{\text{Ridge}} = \frac{1}{1 + \lambda} \cdot \widehat{\beta}_{\text{OLS}}
$$

## 结论
在正交设计 ($X^T X = I$) 下，岭回归估计量是最小二乘估计量的 $\frac{1}{1 + \lambda}$ 倍。