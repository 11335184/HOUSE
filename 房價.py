import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# 讀取資料
df = pd.read_csv("Taipei_house.csv")

# 加入新特色：「是否高樓層」
df['高樓層'] = (df['樓層'] / df['總樓層']) > 0.7
df['高樓層'] = df['高樓層'].astype(int)

# 選擇會影響房價的資料欄位（特徵）
features = ['行政區', '土地面積', '建物總面積', '屋齡', '樓層', '總樓層',
            '用途', '房數', '廳數', '衛數', '電梯', '車位類別', '經度', '緯度', '高樓層']
target = '總價'  # 預測的目標是房價

X = df[features]  # 特徵（輸入）
y = df[target]    # 目標（輸出）

# 把「文字」資料轉成機器能理解的數字（用 OneHotEncoder）
categorical = ['行政區', '車位類別']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical)],
    remainder='passthrough'
)

# 建立「線性回歸」模型
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 分成訓練資料和測試資料（8:2）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 訓練模型
model.fit(X_train, y_train)

# 預測房價
y_pred = model.predict(X_test)

# 顯示模型準確度
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== 預測結果 ===")
print(f"平均誤差：{mae:.0f} 萬元")
print(f"準確率 R²：{r2:.2f}")

