import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 读取Carseats数据集
url = "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv"
data = pd.read_csv(url)

# 2. 将ShelveLoc转换成虚拟变量
# drop_first=True表示删除Bad组，因此Bad是基准组
data = pd.get_dummies(
    data,
    columns=["ShelveLoc"],
    drop_first=True,
    dtype=int
)

# 3. 选择自变量和因变量
X = data[
    [
        "Price",
        "Income",
        "Advertising",
        "ShelveLoc_Good",
        "ShelveLoc_Medium"
    ]
]

y = data["Sales"]

# 添加常数项
X = sm.add_constant(X)

# 4. 建立多元线性回归模型
model = sm.OLS(y, X).fit()

# 输出模型拟合报告
print(model.summary())

# 5. 输出ShelveLoc的基准组
print("\nShelveLoc的基准组：Bad")

# 6. 输出Good组的系数
print(
    "ShelveLoc[Good]系数：",
    model.params["ShelveLoc_Good"]
)

# 7. 计算VIF
vif = pd.DataFrame()

vif["变量"] = X.columns[1:]

vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(1, X.shape[1])
]

print("\n各变量的VIF：")
print(vif)
