import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load_application_train():
    data = pd.read_csv(r"C:\Users\Hülya\PyCharmMiscProject\Feature_Engineering\diabetes.csv")
    return data

df = load_application_train()
df.head()
#GÖREV 1 EDA
#Adım 1
#Pregnancies: Kişinin kaç kez hamile kaldığı
#Glucose: Glikoz seviyesi
#BloodPressure: Kan basıncı (küçük/büyük değil, ölçülen tansiyon değeri)
#SkinThickness: Deri kıvrım kalınlığı
#Insulin: İnsülin seviyesi
#BMI: Vücut kitle indeksi
#DiabetesPedigreeFunction: Diyabet kalıtım / aile geçmişi etkisini gösteren değer
#Age: Yaş
#Outcome: Diyabet durumu
df.shape
df.columns
df.dtypes
df.info()
df.describe().T
df.isnull().sum()   #boş değer yok.


#Adım 2: Numerik ve kategorik değişkenleri yakalayınız
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype != "O"]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and df[col].dtype == "O"]

cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

num_cols = [col for col in df.columns if df[col].dtype != "O"]
num_cols = [col for col in num_cols if col not in num_but_cat]

print("Kategorik değişkenler:", cat_cols)
print("Numerik değişkenler:", num_cols)
print("Numerik ama kategorik:", num_but_cat)
print("Kategorik ama kardinal:", cat_but_car)

#Kategorik değişkenler: ['Outcome']
#Numerik değişkenler:
#['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#Numerik ama kategorik: ['Outcome']
#Kategorik ama kardinal: []
#8 bağımsız değişken ,1 hedef değişken var .hedef değişken outcome. hasta ya da değil.


#Adım 4:Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
#numerik değişkenlerin ortalaması)
#Hedef değişken "outcome" olarak belirtilmiş olup;
# 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir
# Hedef değişkene göre numerik değişken ortalamaları
df.groupby("Outcome")[num_cols].mean()
#Veri setinde hedef değişken dışında kategorik değişken bulunmadığından,
# kategorik değişkenlere göre hedef değişken ortalaması hesaplanamamıştır.


#Adım 5: Aykırı gözlem analizi yapınız
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False
#Numerik değişkenlerde aykırı değer kontrolü
for col in num_cols:
    print(col, check_outlier(df, col))

#Aykırı değerleri görmek için
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Insulin")   #insülin değişkenindeki aykırı gözlemler.
grab_outliers(df, "Age")
#tüm mümerik değerler için outlierlar
for col in num_cols:
    print(col)
    grab_outliers(df, col)

#aykırı değer sayısı
for col in num_cols:
    low, up = outlier_thresholds(df, col)
    print(col, df[((df[col] < low) | (df[col] > up))].shape[0])
#aykırı değerler için görsel.
import seaborn as sns
import matplotlib.pyplot as plt

for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()


#Adım 6: Eksik gözlem analizi yapınız
# Eksik Değerlerin Yakalanması
#############################################


df.head()

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()


def missing_values_table(dataframe):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)

#eksik veri yok.
#Adım 7: Korelasyon analizi yapınız
df[num_cols].corr()  #nümerik değişkenler için korelasyona baktım.
#görselleştirelim.
import matplotlib.pyplot as plt
import seaborn as sns

corr = df[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="RdBu")
plt.title("Korelasyon Matrisi")
plt.show()

#hedef değişkenle korelasyona bakalım.
df.corr(numeric_only=True)[["Outcome"]].sort_values(by="Outcome", ascending=False) #Outcome ile en ilişkili değişken Glucose değişkeni.

#Görev 2: Feature Engineering
#Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
#değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
#olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
#değerlere işlemleri uygulayabilirsiniz
#0 değerleri NaN yapma.
zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df[zero_columns] = df[zero_columns].replace(0, np.nan)
df.isnull().sum()
for col in zero_columns:
    df[col] = df[col].fillna(df[col].median())   #eksik değerleri medyanla doldurdum.

#aykrıı değerler için analiz fonksiyonu:
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))].any(axis=None)


# aykırı değerleri baskılama
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].astype(float)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


num_cols = [col for col in df.columns if df[col].dtype != "O" and col != "Outcome"]

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    if check_outlier(df, col):
        print(col, "aykırı değer var")
    else:
        print(col, "aykırı değer yok")



#Adım 2: Yeni değişkenler oluşturunuz.

# Yaş kategorisi
df.loc[df["Age"] < 35, "NEW_AGE_CAT"] = "young"
df.loc[(df["Age"] >= 35) & (df["Age"] < 55), "NEW_AGE_CAT"] = "middleage"
df.loc[df["Age"] >= 55, "NEW_AGE_CAT"] = "old"

# BMI kategorisi
df.loc[df["BMI"] < 18.5, "NEW_BMI_CAT"] = "underweight"
df.loc[(df["BMI"] >= 18.5) & (df["BMI"] < 24.9), "NEW_BMI_CAT"] = "normal"
df.loc[(df["BMI"] >= 25) & (df["BMI"] < 29.9), "NEW_BMI_CAT"] = "overweight"
df.loc[df["BMI"] >= 30, "NEW_BMI_CAT"] = "obese"

# Glucose kategorisi
df.loc[df["Glucose"] < 70, "NEW_GLUCOSE_CAT"] = "low"
df.loc[(df["Glucose"] >= 70) & (df["Glucose"] < 100), "NEW_GLUCOSE_CAT"] = "normal"
df.loc[(df["Glucose"] >= 100) & (df["Glucose"] < 126), "NEW_GLUCOSE_CAT"] = "prediabetes"
df.loc[df["Glucose"] >= 126, "NEW_GLUCOSE_CAT"] = "diabetes_risk"

# Yaş + BMI birlikte
df.loc[(df["Age"] < 35) & (df["BMI"] < 30), "NEW_AGE_BMI"] = "young_normal"
df.loc[(df["Age"] < 35) & (df["BMI"] >= 30), "NEW_AGE_BMI"] = "young_obese"

df.loc[(df["Age"] >= 35) & (df["Age"] < 55) & (df["BMI"] < 30), "NEW_AGE_BMI"] = "middleage_normal"
df.loc[(df["Age"] >= 35) & (df["Age"] < 55) & (df["BMI"] >= 30), "NEW_AGE_BMI"] = "middleage_obese"

df.loc[(df["Age"] >= 55) & (df["BMI"] < 30), "NEW_AGE_BMI"] = "old_normal"
df.loc[(df["Age"] >= 55) & (df["BMI"] >= 30), "NEW_AGE_BMI"] = "old_obese"

# Yaş ile glikoz çarpımı
df["NEW_AGE_GLUCOSE"] = df["Age"] * df["Glucose"]

# BMI ile glikoz çarpımı
df["NEW_BMI_GLUCOSE"] = df["BMI"] * df["Glucose"]


df.head()

#Adım 3:  Encoding işlemlerini gerçekleştiriniz
#kategorik değişkenleri one hot encodingle sayısala dönüştürüyoruz.
#Oluşturulan yeni değişkenler kategorik yapıda olup sınıflar arasında doğrudan sayısal bir
# sıralama ilişkisi bulunmadığından, label encoding yerine one-hot encoding tercih edilmiştir.
cat_cols = ["NEW_AGE_CAT", "NEW_BMI_CAT", "NEW_GLUCOSE_CAT", "NEW_AGE_BMI"]

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df.head()
[col for col in df.columns if "NEW" in col]

#Adım 4: Numerik değişkenler için standartlaştırma yapınız
#outcome benim hedef değişkenim onun haricinde kalan değişkenler için
#standart scaler kullandım.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols = [col for col in df.columns if df[col].dtype != "O" and col != "Outcome"]

df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()


#Adım 5: Model oluşturunuz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=17
)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy_score(y_test, y_pred)
accuracy_score(y_pred, y_test)









