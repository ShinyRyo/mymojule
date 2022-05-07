#全てのカテゴリカルデータをラベルエンコーディング
def MyLabelEncoding(train, test):
    from sklearn.preprocessing import LabelEncoder
    feats, catfeats, target = devide_tr_ob(train, test)
    data=pd.concat([train, test])
    le= LabelEncoder() #ラベルエンコーダーをインスタンス化して使えるようにする
    labels={}
    for feat in catfeats: 
        le.fit(data[feat].astype(str))
        label={feat:list(le.classes_)}
        labels.update(label)
        data[feat]=le.transform(data[feat].astype(str))
    train, test = devide_train_test(data, target)
    return labels, train, test