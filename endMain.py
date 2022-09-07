import pandas as pd
from load_data import LoadData
from allsurvalOne import LifeLine
from ChartStatisticalCalculations import df_generator
from ga import GA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold as KFold


test_fpr = []
test_tpr = []
test_auc = []

pool = LifeLine('ParsDataSet7.xlsx')
beta = pool.Beta5()
S0 = pool.BaselineSurvival5()
age = [40, 50, 60, 70, 78]
PREMIURE = [1, 2, 3, 4]
s = ['sex', 'whr', "smoker"]
DL = [1, 2, 3, 4, 5]
dataframe = pd.read_excel('ParsDataSet7.xlsx',
                          names=["IHHPCode", "sex", "whr", "smoker", "family_history", "diabetes",
                                 "age",
                                 "cholesterol", "blood_preMIure", "dbp", "hdl", "ldl", "tg", "htn",
                                 "sbp_cat", "sbp1", "sbp2", "sbp3", "sbp4", "tchcat", "tch1", "tch2",
                                 "tch3", "tch4",
                                 "tch5", "FollowDu5th", "label"], header=0)

a = df_generator(dataframe=dataframe, S0=S0, Beta=beta, Dual_RF=s, AGE=age, SBP=PREMIURE, DL=True, List_DL=DL)

print(a)
a = a[::-1]

age = [45, 55, 65, 75]
pre = [120, 140, 160]
load_data = LoadData('ParsDataSet7.xlsx')
data = load_data.split_data_by_list_feature_age_pre(s, age, pre)

ga = GA(splited_data=data, a=a, iteration=50, population_number=100, len_age=5, len_pre=4, mutation_rate=0.3,
        mutation_use=0.5)
ga.run(iters=1)

load_data = LoadData('ParsDataSet7.xlsx')
data = load_data.split_data_by_list_feature_age_pre(s, age, pre)
print(ga.a.T[::-1])



plt.figure()
plt.plot(ga.best_costs)
plt.xlabel("number of cell improved by genetics")
plt.ylabel("area under ROC curve")
plt.savefig("bestcost" + ".png")

# np.savetxt("test_auc.txt", np.array(test_auc), delimiter=',')
