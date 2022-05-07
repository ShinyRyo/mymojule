#訓練データとテストデータの分割
def devide_train_test(data, target):
    train=data[data[target[0]].notna()]
    test=data[data[target[0]].isna()]
    return train, test