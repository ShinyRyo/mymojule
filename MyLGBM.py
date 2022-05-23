class MyLGBM:
    """
    カテゴリカルデータとして、objectとboolを指定しました。
    """
    def __init__(self,data_path):
        self.data_path=data_path
        self.train=0
        self.test=0
        self.submit=0
        self.labels={}
        self.LgbDataSet=[]
        self.data=0
        self.feats=0
        self.catfeats=0
        self.target=0
        self.lgb_model=0

    def __call__(self):
        self.data_loader()
        self.devide_tr_ob()
        self.MyLabelEncoding()
        self.LGBM_K_DataSet()
        self.LGBM_train()

    #データを読み込む
    def data_loader(self):
        import os
        import pandas as pd
        for data in os.listdir(self.data_path):
            if "csv" in data:
                if "train" in data:
                    self.train=pd.read_csv(self.data_path+data)
                elif "test" in data:
                    self.test=pd.read_csv(self.data_path+data)
                elif "submit" in data:
                    self.submit=pd.read_csv(self.data_path+data, header=None)
            elif "tsv" in data:
                if "train" in data:
                    self.train=pd.read_csv(self.data_path+data, sep='\t')
                elif "test" in data:
                    self.test=pd.read_csv(self.data_path+data, sep='\t')
                elif "submit" in data:
                    self.submit=pd.read_csv(self.data_path+data, sep='\t', header=None)
        return self.train, self.test, self.submit
    #全てのカテゴリカルデータをラベルエンコーディング
    def MyLabelEncoding(self):
        from sklearn.preprocessing import LabelEncoder
        #feats, catfeats, target = devide_tr_ob(self.train, self.test)
        import pandas as pd
        data=pd.concat([self.train, self.test])
        le= LabelEncoder() #ラベルエンコーダーをインスタンス化して使えるようにする
        self.labels={}
        for feat in self.catfeats: 
            le.fit(data[feat].astype(str))
            label={feat:list(le.classes_)}
            self.labels.update(label)
            data[feat]=le.transform(data[feat].astype(str))
        self.train=data[:len(self.train)]
        self.test=data[len(self.train):][self.feats]
        return self.labels, self.train, self.test

    def LGBM_K_DataSet(self):
        import lightgbm as lgb
        from sklearn.model_selection import KFold
        folds = KFold(n_splits = 5 , random_state = 6, shuffle=True)
        KFoldDataSet=[]
        #feats, catfeats, target=devide_tr_ob(self.train, self.test)
        X=self.train[self.feats]
        y=self.train[self.target]
        for train_id , valid_id in folds.split(X, y):
            DataSet=[X.iloc[train_id],y.iloc[train_id],X.iloc[valid_id],y.iloc[valid_id]]
            KFoldDataSet.append(DataSet)
        self.LgbDataSet=[]
        for X_train,y_train,X_valid,y_valid in KFoldDataSet:
            train_data = lgb.Dataset(data = X_train, label = y_train, categorical_feature=self.catfeats,free_raw_data=False)
            valid_data = lgb.Dataset(data = X_valid, label = y_valid, categorical_feature=self.catfeats,free_raw_data=False)
            self.LgbDataSet.append([train_data, valid_data])
        return self.LgbDataSet

    def LGBM_train(self):
        import lightgbm as lgb
        params = {
            'objective' : 'regression',
            'metric': 'rmse',
            'random_state': 0,
        }
        #feats, catfeats, target=devide_tr_ob(self.train, self.test)
        self.lgb_model=0
        if self.target[0] in self.catfeats:
            self.catfeats.remove(self.target[0])
        for train_data,valid_data in self.LgbDataSet:
                self.lgb_model = lgb.train(params ,
                                    train_data ,
                                    valid_sets=[train_data , valid_data] ,
                                    categorical_feature=self.catfeats,
                                    verbose_eval=100,
                                    early_stopping_rounds=100,
                                    )
                # file = f'model{i}.pkl'
                # pickle.dump(lgb_model, open(model_path+file,'wb'))
                # print("save model")
                # dic={i:lgb_model}
                # lgb_models.update(dic)
        return self.lgb_model
    #説明変数と目的変数のリストを抽出
    def devide_tr_ob(self):
        train_labels=list(self.train.columns)
        test_labels=list(self.test.columns)
        self.target=train_labels
        self.feats=test_labels
        self.catfeats=list(self.train.select_dtypes(include= ["object",bool]).columns)
        for i in test_labels:
            self.target.remove(i)
        return self.feats, self.catfeats, self.target