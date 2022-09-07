import math


def df_generator(dataframe, S0, Beta, Dual_RF, AGE, SBP, DL=False, List_DL=[]):
    FeatureMean = dataframe.mean(axis=0)
    beta = Beta

    if DL:
        column = (2 ** len(Dual_RF))
        row = len(SBP) * len(AGE)
        df = [[0 for col in range(column)] for row in range(row)]
        SS = (beta.get('age') * FeatureMean.age) + (beta.get("sex") * FeatureMean.sex) + (
                    (beta.get("whr")) * FeatureMean.whr) + \
             ((beta.get("sbp2")) * FeatureMean.sbp2) + ((beta.get("sbp3")) * FeatureMean.sbp3) + (
                     (beta.get("sbp4")) * FeatureMean.sbp4)
        for i in range(row):
            FeatureAge = AGE[(len(AGE) - 1) - (i // len(SBP))]
            FeatureSBP = SBP[(len(SBP) - 1) - (i % len(SBP))]

            # print(FeatureSBP)

            for j in range(column):
                VV = 0

                FeatureDL = List_DL[(j % len(List_DL))]
                twoD_length = len(List_DL)
                # Age
                VV += (beta.get("age")) * FeatureAge

                # Sbp
                if FeatureSBP == 4:
                    VV += (beta.get("sbp4")) * 1
                elif FeatureSBP == 3:
                    VV += (beta.get("sbp3")) * 1
                elif FeatureSBP == 2:
                    VV += (beta.get("sbp2")) * 1
                else:
                    VV += 0


                # sex
                if 0 <= j < 2:
                    VV += 0
                else:
                    VV += beta.get("sex") * 1

                # whr
                if 0 <= j < 1 or 2 <= j < 3:
                    VV += 0
                else:
                    VV += (beta.get("whr")) * 1

                df[i][j] = round((1 - (S0 ** (math.exp(VV - SS)))) * 100)
                # df[i][j] = ((1 - (S0 ** (math.exp(VV - SS)))) * 100)

        return df
