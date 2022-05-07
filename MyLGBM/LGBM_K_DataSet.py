def LGBM_K_DataSet(train, feats, catfeats, target):
    from sklearn.model_selection import KFold
    folds = KFold(n_splits = 5 , random_state = 6, shuffle=True)
    KFoldDataSet=[]
    X=train[feats]
    y=train[target]
    for train_id , valid_id in folds.split(X, y):
        DataSet=[X.iloc[train_id],y.iloc[train_id],X.iloc[valid_id],y.iloc[valid_id]]
        KFoldDataSet.append(DataSet)
    import lightgbm as lgb
    LgbDataSet=[]
    for X_train,y_train,X_valid,y_valid in KFoldDataSet:
        train_data = lgb.Dataset(data = X_train, label = y_train, categorical_feature=catfeats,free_raw_data=False)
        valid_data = lgb.Dataset(data = X_valid, label = y_valid, categorical_feature=catfeats,free_raw_data=False)
        LgbDataSet.append([train_data, valid_data])
    return LgbDataSet