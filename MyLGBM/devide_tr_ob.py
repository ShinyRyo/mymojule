#説明変数と目的変数のリストを抽出
def devide_tr_ob(train, test):
    train_labels=list(train.columns)
    test_labels=list(test.columns)
    target=train_labels
    feats=test_labels
    catfeats=list(train.select_dtypes(include= "object").columns)
    for i in test_labels:
        target.remove(i)
    return feats, catfeats, target