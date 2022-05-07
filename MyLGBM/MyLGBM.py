class MyLGBM:
    def __init__(self, train, test):
        self.train=train
        self.test=test

    def devide_tr_ob(self):
        train_labels=list(self.train.columns)
        test_labels=list(self.test.columns)
        target=train_labels
        feats=test_labels
        catfeats=list(train.select_dtypes(include= "object").columns)
        for i in test_labels:
            target.remove(i)
        return feats, catfeats, target
    #訓練データとテストデータの分割
    def devide_train_test(self):
        train=self.data[self.data[target[0]].notna()]
        test=self.data[self.data[target[0]].isna()]
        return train, test
    #全てのカテゴリカルデータをラベルエンコーディング
    def MyLabelEncoding(self):
        from sklearn.preprocessing import LabelEncoder
        feats, catfeats, target = devide_tr_ob(self.train, self.test)
        data=pd.concat([self.train, self.test])
        le= LabelEncoder() #ラベルエンコーダーをインスタンス化して使えるようにする
        labels={}
        for feat in catfeats: 
            le.fit(data[feat].astype(str))
            label={feat:list(le.classes_)}
            labels.update(label)
            data[feat]=le.transform(data[feat].astype(str))
        train, test = devide_train_test(data, target)
        return labels, train, test

    def LGBM_K_DataSet(self):
        from sklearn.model_selection import KFold
        folds = KFold(n_splits = 5 , random_state = 6, shuffle=True)
        KFoldDataSet=[]
        X=self.train[feats]
        y=self.train[target]
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

    def LGBM_train(train, test):
        LgbDataSet=LGBM_K_DataSet(self.train, self.test)
        params = {
            'objective' : 'regression',
            'metric': 'rmse',
            'random_state': 0,
        }
        for train_data,valid_data in LgbDataSet:
                lgb_model = lgb.train(params ,
                                    train_data ,
                                    valid_sets=[train_data , valid_data] ,
                                    categorical_feature=catfeats,
                                    verbose_eval=100,
                                    early_stopping_rounds=100,
                                    )
                # file = f'model{i}.pkl'
                # pickle.dump(lgb_model, open(model_path+file,'wb'))
                # print("save model")
                # dic={i:lgb_model}
                # lgb_models.update(dic)
        return lgb_model