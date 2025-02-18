import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



class Dataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.X = self.df.drop(["Survived"], axis=1)
        self.Y = self.df["Survived"]

    def __len__(self):
        """
        クラスxxの組み込み関数len()をオーバーライド。
        self.dfの行数を返す。
        """
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        array[i]のようにindexを指定されたときに呼ばれる関数。
        DataLoaderオブジェクトをfor文でイテレートするときに呼ばれ、
        Survived以外のデータと、Survivedのデータに分けて返す。
        """
        print(type(self.X.iloc[idx, :]))
        print(type(self.Y.iloc[idx]))
        return self.X.iloc[idx, :].values, self.Y.iloc[idx]


class Net(nn.Module):
    def __init__(self, input_sz, hidden_sz, out_sz):
        super(Net, self).__init__()
        self.f1 = nn.Linear(input_sz, hidden_sz)
        self.f2 = nn.Linear(hidden_sz, out_sz)

    def forward(self, x):
        h1 = F.sigmoid(self.f1(x))
        y = self.f2(h1)
        return y


def impute_age(train_data: pd.DataFrame):
    """訓練データのAgeカラムの欠損値を補完します。"""

    features = [
        'Pclass',
        'Sex',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked'
    ]
    target = 'Age'

    categorical_features = ['Sex', 'Embarked']
    numerical_features = ['Pclass', 'SibSp', 'Parch', 'Fare']

    # カテゴリ変数をOne-Hot Encodingで処理
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 数値変数の欠損値を平均で埋める
    numerical_transformer = SimpleImputer(strategy='mean')

    # 前処理モデルを作成
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Ageが欠損していない行のみを取得
    train_age: pd.DataFrame = train_data.dropna(subset=['Age'])

    # Ageが欠損している行のみを取得
    test_age: pd.DataFrame = train_data[train_data['Age'].isnull()].copy()

    X_train = train_age[features]   # Ageを推測するために用いる特徴量
    y_train = train_age[target]     # 学習時に用いる正解ラベル

    X_test = test_age[features]     # Ageが欠損しているデータ

    # モデルの定義（ランダムフォレスト回帰）
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # パイプライン作成（前処理 + 予測モデル）
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 学習
    pipeline.fit(X=X_train, y=y_train)

    # 学習データの適合度を評価（MAE）
    y_pred: np.ndarray = pipeline.predict(X=X_train)
    mae = mean_absolute_error(y_true=y_train, y_pred=y_pred)
    print(f"学習データでのMAE: {mae:.2f}")

    # 欠損しているAgeを予測
    test_age['Age'] = pipeline.predict(X=X_test)

    # 予測したAgeを欠損部分に反映
    train_data.loc[train_data['Age'].isnull(), 'Age'] = test_age['Age']

    # 確認
    print(train_data[train_data['Age'].isnull()])

    return train_data


def process_data(df: pd.DataFrame):
    """
    データを成形します。
    予測に不要そうな列を消し、カテゴリ変数のSex列を数値に変換します。
    """
    drop_columns = ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]
    df = df.drop(drop_columns, axis=1)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    return df
