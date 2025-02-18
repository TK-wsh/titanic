import pandas as pd
from torch.utils.data import DataLoader

from app import Dataset, Net, impute_age, process_data

BATCH_SIZE = 32


if __name__ == '__main__':
    train_data = pd.read_csv('./input/train.csv')
    test_data = pd.read_csv('./input/test.csv')
    train_data.info()
    print(train_data.head())
    test_data.info()
    print(test_data.head())

    # Ageの欠損値を補完します。
    train_data = impute_age(train_data)

    # 学習データを成形します。
    train_data = process_data(train_data)

    print(train_data.head())
    print(train_data.info())

    # データセットを作成します。
    train_dataset: Dataset = Dataset(train_data)

    # データローダーを作成します。
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True)

    # xにtrain_dataの入力，yにtrain_dataのラベルが入る
    for x, y in train_dataloader:
        print(x, y)
        break

    # モデルを作成します。
    input_sz = 6
    hidden_sz = 3
    out_sz = 2

    net = Net(input_sz, hidden_sz, out_sz)

    # モデルを訓練します。
