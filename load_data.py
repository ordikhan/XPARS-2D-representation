import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTENC


class LoadData(object):
    def __init__(self, file_name):
        self.df = pd.read_excel(file_name)

    @staticmethod
    def improve_dataset(datas, categorical_feature):
        X = datas.to_numpy()
        feature_list = list(datas.columns)
        categorical_features = []
        for categorical_feature_ in categorical_feature:
            categorical_features.append(feature_list.index(categorical_feature_))

        y = X[:, -1]
        X = X[:, :-1]

        sm = SMOTENC(random_state=9, categorical_features=categorical_features, k_neighbors=2)
        X_res, y_res = sm.fit_resample(X, y)
        y_res = y_res.reshape((-1, 1))
        data= np.concatenate([X_res,y_res],axis=1)
        df = pd.DataFrame(data=data,  # values
                          columns=datas.columns)  # 1st row as the column names]
        return df

    def split_data_by_feature(self, feature_list):
        datas = [self.df[self.df[feature_list[0]] == 0], self.df[self.df[feature_list[0]] == 1]]
        for feature in feature_list[1:]:
            temp = []
            for data in datas:
                temp.append(data[data[feature] == 0])
                temp.append(data[data[feature] == 1])
            datas = temp
        return datas

    def split_data_by_feature_and_continue_feature(self, datas, continue_feature_name, continue_feature_list):
        datas = [data.sort_values(by=continue_feature_name) for data in datas]
        out = []
        for data in datas:
            for i in range(len(continue_feature_list)):
                out.append(data[data[continue_feature_name] < continue_feature_list[i]])
                data[data[continue_feature_name] < continue_feature_list[i]] = None
            out.append(data[data[continue_feature_name] >= continue_feature_list[-1]])
        return out

    def split_data_by_list_feature_age_pre(self, feature_list, age, pre):
        datas = self.split_data_by_feature(feature_list)
        datas = self.split_data_by_feature_and_continue_feature(datas, "age", age)
        datas = self.split_data_by_feature_and_continue_feature(datas, "blood_pressure", pre)
        return datas
