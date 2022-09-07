import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter
import numpy as np

class LifeLine:
    def __init__(self, FileAddress):
        self.dataframe = pd.read_excel(FileAddress,
                                       names=["IHHPCode", "sex", "whr", "smoker", "family_history", "diabetes", "age",
                                              "cholesterol", "blood_pressure", "dbp", "hdl", "ldl", "tg", "htn",
                                              "sbp_cat", "sbp1", "sbp2", "sbp3", "sbp4", "tchcat", "tch1", "tch2",
                                              "tch3", "tch4", "tch5",
                                              "FollowDu5th", "label"], header=0)

    def Beta3(self):
        self.dataframe_r = self.dataframe.loc[:,
                           {"sex", "age", "sbp2", "sbp3", "sbp4", "FollowDu5th",
                            "label"}]
        self.cph = CoxPHFitter()
        self.cph.fit(self.dataframe_r, duration_col='FollowDu5th', event_col='label')
        self.Beta = self.cph.hazards_.iloc[0].to_dict()
        return self.Beta

    def BaselineSurvival3(self):
        self.dataframe_r = self.dataframe.loc[:,
                           {"sex", "age", "sbp2", "sbp3", "sbp4", "FollowDu5th",
                            "label"}]
        self.cph = CoxPHFitter()
        self.cph.fit(self.dataframe_r, duration_col='FollowDu5th', event_col='label')
        BaselineS = self.cph.baseline_survival_
        p = BaselineS.to_numpy().reshape(-1)
        avrg = np.mean(p)
        return avrg
        # return float(BaselineS.loc[144])

    def Beta4(self):
        self.dataframe_r = self.dataframe.loc[:,
                           {"sex", "whr", "age", "sbp2", "sbp3", "sbp4", "FollowDu5th",
                            "label"}]
        self.cph = CoxPHFitter()
        self.cph.fit(self.dataframe_r, duration_col='FollowDu5th', event_col='label')
        self.Beta = self.cph.hazards_.iloc[0].to_dict()
        return self.Beta

    def BaselineSurvival4(self):
        self.dataframe_r = self.dataframe.loc[:,
                           {"sex", "whr", "age", "sbp2", "sbp3", "sbp4", "FollowDu5th",
                            "label"}]
        self.cph = CoxPHFitter()
        self.cph.fit(self.dataframe_r, duration_col='FollowDu5th', event_col='label')
        BaselineS = self.cph.baseline_survival_
        p = BaselineS.to_numpy().reshape(-1)
        avrg = np.mean(p)
        return avrg

    def Beta5(self):
        self.dataframe_r = self.dataframe.loc[:,
                           {"sex", "whr", "smoker", "age",  "sbp2", "sbp3", "sbp4",
                            "FollowDu5th", "label"}]
        self.cph = CoxPHFitter()
        self.cph.fit(self.dataframe_r, duration_col='FollowDu5th', event_col='label')
        self.Beta = self.cph.hazards_.iloc[0].to_dict()
        return self.Beta

    def BaselineSurvival5(self):
        self.dataframe_r = self.dataframe.loc[:,
                           {"sex", "whr", "smoker", "age", "sbp2", "sbp3", "sbp4",
                            "FollowDu5th", "label"}]
        self.cph = CoxPHFitter()
        self.cph.fit(self.dataframe_r, duration_col='FollowDu5th', event_col='label')
        BaselineS = self.cph.baseline_survival_
        p = BaselineS.to_numpy().reshape(-1)
        avrg = np.mean(p)
        return avrg

    def Beta6(self):
        self.dataframe_r = self.dataframe.loc[:,
                           {"sex", "whr", "smoker", "family_history", "age", "sbp2",
                            "sbp3", "sbp4",
                            "FollowDu5th", "label"}]
        self.cph = CoxPHFitter()
        self.cph.fit(self.dataframe_r, duration_col='FollowDu5th', event_col='label')
        self.Beta = self.cph.hazards_.iloc[0].to_dict()
        return self.Beta

    def BaselineSurvival6(self):
        self.dataframe_r = self.dataframe.loc[:,
                           {"sex", "whr", "smoker", "family_history", "age", "sbp2",
                            "sbp3", "sbp4",
                            "FollowDu5th", "label"}]
        self.cph = CoxPHFitter()
        self.cph.fit(self.dataframe_r, duration_col='FollowDu5th', event_col='label')
        BaselineS = self.cph.baseline_survival_
        p = BaselineS.to_numpy().reshape(-1)
        avrg = np.mean(p)
        return avrg

    def Beta7(self):
        self.dataframe_r = self.dataframe.loc[:,
                           {"sex", "whr", "diabetes", "smoker", "family_history", "age", "sbp2",
                            "sbp3", "sbp4",
                            "FollowDu5th", "label"}]
        self.cph = CoxPHFitter()
        self.cph.fit(self.dataframe_r, duration_col='FollowDu5th', event_col='label')
        self.Beta = self.cph.hazards_.iloc[0].to_dict()
        return self.Beta

    def BaselineSurvival7(self):
        self.dataframe_r = self.dataframe.loc[:,
                           {"sex", "whr", "smoker","diabetes", "family_history", "age", "sbp2",
                            "sbp3", "sbp4",
                            "FollowDu5th", "label"}]
        self.cph = CoxPHFitter()
        self.cph.fit(self.dataframe_r, duration_col='FollowDu5th', event_col='label')
        BaselineS = self.cph.baseline_survival_
        p = BaselineS.to_numpy().reshape(-1)
        avrg = np.mean(p)
        return avrg

# a=LifeLine('ParsDataSet7.xlsx')
# print(a.Beta3())
# print(a.Beta4())
# print(a.Beta5())
# print(a.Beta6())