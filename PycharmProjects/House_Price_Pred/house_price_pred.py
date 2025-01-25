# 1. GEREKLILIKLER

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pandas.conftest import axis_1
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

import matplotlib

from HOUSE_PRICE_PREDICTON_SOLUTION import X_train

matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

df = pd.concat([train,test], ignore_index=False).reset_index()

df.head()

df.drop("index", axis=1, inplace=True)

df.head()

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Tail #####################")
    print(dataframe.tail)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles ##################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], method = "table", interpolation="nearest").T)


check_df(df)
df.dtypes

#Numerik ve Kategorik değişkenleri toplayalım----

def get_cols(dataframe, cat_th = 10, car_th=20):


    cat_cols =[col for col in dataframe.columns if dataframe[col].dtype == "object"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtype != "object"]

    cat_cols = cat_cols + num_but_cat

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "object"]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols


cat_cols, num_cols = get_cols(df)


# Kategorik değişken Analizi

def categoric_sum(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100*dataframe[col_name].value_counts() / len(dataframe)
                        }))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    categoric_sum(df,col)

# Sayısal değişken Analizi----

def numeric_var_anlys(dataframe, num_col, plot=False):
    quantiles = [col for col in np.arange(0.10,0.95,0.10)]
    print(dataframe[num_col].describe(quantiles).T)

    if plot:
        dataframe[num_col].hist(bins=50)
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show()
    print("---------------------------------")


for col in num_cols:
    numeric_var_anlys(df,col,plot=True)

# Hedef değişken Analizi

def target_with_cat(dataframe, target, categ_col):
    print(pd.DataFrame({"TARGET_SUMMARY" : dataframe.groupby(categ_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_with_cat(df,"SalePrice",col)

# Bağımlı değişkenin incelenmesi

df["SalePrice"].hist(bins=100)
plt.show()

# Bağımlı değişkenin logaritmasının incelenmesi

np.log1p(df["SalePrice"].hist(bins=50))
plt.show()

# Korelasyon Analizi----

corr = df[num_cols].corr()

print(corr)

# Korelasyonların Gösterilmesi---

sns.set(rc={"figure.figsize": (12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()
plt.title("Değişkenlerin korelasyon analizi")

# High correlated cols

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=False)


# Aykırı Değer Analizi------

def outlier_threshold(dataframe, variable, low_quantile = 0.15, up_quantile = 0.85):
    quantile1 =dataframe[variable].quantile(low_quantile)
    quantile3 = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile3 - quantile1

    up_limit = quantile3 + 1.5*interquantile_range
    low_limit = quantile1 - 1.5*interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü -----------

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col)

    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(df,col))


def replace_with_thresholds(dataframe, column):
    low_limit, up_limit = outlier_threshold(dataframe,column)
    dataframe.loc[dataframe[column] < low_limit, column] = low_limit
    dataframe.loc[dataframe[column] > up_limit, column] = up_limit

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df,col)



# Eksik Değer Analizi--------

def missing_value_method(dataframe, nan_name=False):
    nan_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[nan_columns].isnull().sum().sort_values(ascending=False)

    ratio = (100 * dataframe[nan_columns].isnull().sum() / dataframe.shape[0]).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss","ratio"])

    print(missing_df, end="\n")

    if nan_name:
        return nan_columns

missing_value_method(df, nan_name=True)

df["Alley"].value_counts()
df["BsmtQual"].value_counts()

# Bazı değişkenlerdeki boş değerler, evin o özelliğe sahip olmadığını ifade etmektedir.
no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

df["Alley"].isnull().sum()

for col in no_cols:
    df[col].fillna("No", inplace=True)

missing_value_method(df)

# Bu fonksiyon eksik değerlerin median veya mean ile doldurulmasını sağlar.

def filling_mean_median(data, num_method="mean", target="SalePrice", cat_length=20):
    # eksik değerlere sahip değişkenler listelenir.
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]

    temp_target = data[target]
    print("----BEFORE----")
    print(data[variables_with_na].isnull().sum(), "\n\n") # uygulama öncesi değişkenlerin eksik değer sayısı
    # değişken object veya sınıf sayısı cat_lengthe eşit veya küçük ise boş değerleri mode il doldur.
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "object") and (x.nunique() <= cat_length) else x, axis=0)

    # num_method mean ise object olmayan değişkenlerin boş değerleri mean ile dolduruluyor.
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if (x.dtype != "object") else x, axis=0)

    # num method median ise tipi object olmayan değişkenlerin boş değerleri median ile dolduruluyor.
    if num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if (x.dtype != "object") else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print("Imputation method is '" + num_method + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data

df = filling_mean_median(df, num_method="median", cat_length=17)


# Rare analizi,  Kategorik kolonlaron Dağılımının İncelenmesi ---

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN" : dataframe.groupby(col)[target].mean()

                            }), end="\n\n")

rare_analyser(df, "SalePrice", cat_cols)

# Nadir sınıfların tespit edilmesi--

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtype == "object" and
                    (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)
                    ]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

rare_encoder(df, 0.01)

# Feature Extraction--------------------


# Total Floor
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea # 64

df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.LotArea # 57

df["NEW_GarageLotRatio"] = df.GarageArea / df.LotArea # 69

# MasVnrArea
df["NEW_MasVnrRatio"] = df.MasVnrArea / df.NEW_TotalHouseArea # 36

df["NEW_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2 # 56

# Porch Area
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF # 93

# Total House Area
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF # 156

df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF

df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"] # 61


df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt # 31

df["NEW_HouseAge"] = df.YrSold - df.YearBuilt # 73

df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd # 40

df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt # 17

df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd) # 30

df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt # 48


drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]

# drop_list'teki değişkenleri düşür
df.drop(drop_list, axis=1, inplace=True)

df.head()

# Label Encoding - One Hot Encoding işlemleri

cat_cols, num_cols = get_cols(df)

def label_encod(dataframe, binary_col):
    labelEncoder = LabelEncoder()
    dataframe[binary_col] = labelEncoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype == "object" and df[col].nunique() == 2]


for col in binary_cols:
    label_encod(df,col)

# One Hot Encoding

def one_hot_encod(dataframe, categoric_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categoric_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encod(df,cat_cols,drop_first=True)

df.head()

# MODELLEME

train_df =df[df["SalePrice"].notnull()]
test_df = df[df["SalePrice"].isnull()]

X = train_df.drop(["Id","SalePrice"], axis=1)
y = train_df["SalePrice"]

# Train verisi ile model kurup, model başarısını değerlendirelim.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

models = [("LR", LinearRegression()),("KNN",KNeighborsRegressor()),("CART", DecisionTreeRegressor()),
          ("RF", RandomForestRegressor()), ("GBM", GradientBoostingRegressor()), ("XGB", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


RF = RandomForestRegressor().fit(X_train,y_train)
y_pred =RF.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))



# Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin
# - sıralamasını çizdir.

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()

model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X)


# test dataframeindeki boş olan salePrice değişkenlerini tahminleyelim.

model = LGBMRegressor()
model.fit(X, y)
predictions = model.predict(test_df.drop(["Id","SalePrice"], axis=1))

dictionary = {"Id":test_df.index, "SalePrice":predictions}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("housePricePredictions.csv", index=False)


