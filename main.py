import pandas as pd
from load_data import LoadData
from allsurvalOne import LifeLine
from ChartStatisticalCalculations import df_generator
from ga import GA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold as KFold
from matplotlib.ticker import MaxNLocator

kfold_n_split = 10
kf = KFold(n_splits=kfold_n_split, shuffle=True, random_state=6)
kfole_train = 'kfold_train.xlsx'
kfole_test = 'kfoldtest.xlsx'
file_name = 'ParsDataSet7.xlsx'
main_file = pd.read_excel(file_name)
y = main_file["label"]
kfold_get = kf.split(main_file, y)
test_fpr = []
test_tpr = []
test_auc = []
for j in range(kfold_n_split):
    print("\nstart kfold : ", j)
    result = next(kfold_get)
    train = main_file.iloc[result[0]]
    test = main_file.iloc[result[1]]
    # train = LoadData.improve_dataset(train, ['sex', 'whr',  "smoker", 'age', 'blood_pressure'])
    train.to_excel(kfole_train, header=True, index=False)
    test.to_excel(kfole_test, header=True, index=False)
    del test, train

    pool = LifeLine(kfole_train)
    beta = pool.Beta5()
    age = [40, 50, 60, 70, 78]
    PREMIURE = [1, 2, 3, 4]
    s = ['sex', 'whr', "smoker"]
    DL = [1, 2, 3, 4, 5]
    S0 = pool.BaselineSurvival5()

    dataframe = pd.read_excel(kfole_train,
                              names=["IHHPCode", "sex", "whr", "smoker", "family_history", "diabetes",
                                     "age",
                                     "cholesterol", "blood_preMIure", "dbp", "hdl", "ldl", "tg", "htn",
                                     "sbp_cat", "sbp1", "sbp2", "sbp3", "sbp4", "tchcat", "tch1", "tch2",
                                     "tch3", "tch4",
                                     "tch5", "FollowDu5th", "label"], header=0)

    a = df_generator(dataframe=dataframe, S0=S0, Beta=beta, Dual_RF=s, AGE=age, SBP=PREMIURE, DL=True,
                     List_DL=DL)

    a = a[::-1]

    age = [45, 55, 65, 75]
    pre = [120, 140, 160]
    load_data = LoadData(kfole_train)
    data = load_data.split_data_by_list_feature_age_pre(s, age, pre)

    ga = GA(splited_data=data, a=a, iteration=30, population_number=80, len_age=5, len_pre=4, mutation_rate=0.3,
            mutation_use=0.5)
    costs, number = ga.run(iters=1)

    fig, ax = plt.subplots()
    ax.plot(costs)
    for i, txt in enumerate(number):
        ax.annotate(txt, (i, costs[i]))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(range(len(costs)))
    plt.savefig("result_" + str(j) + "fold.png")

    load_data = LoadData(kfole_test)
    data = load_data.split_data_by_list_feature_age_pre(s, age, pre)
    test_auc.append(ga.test(data))
    print(ga.a.T[::-1])
    print("test auc is : ", test_auc[-1])
    plt.figure()
    plt.plot(ga.best_costs)
    plt.xlabel("number of cell improved by genetics")
    plt.ylabel("area under ROC curve")
    plt.savefig("bestcost" + str(j) + ".png")

np.savetxt("test_auc.txt", np.array(test_auc), delimiter=',')
