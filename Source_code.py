import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('kc_house_data.csv', usecols=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built'])

#Melihat 5 baris teratas dari data.
#Independent variabel(x) adalah bedrooms, bathrooms, sqft_living, grade, yr_built.
#Dependent variabel(y) adalah price.
df.head()

#Mengetahui jumlah kolom dan baris dari data.
#Data ucup mempunya 6 kolom (features) dengan 21613 baris.
df.shape

#Melihat informasi data ucup mulai dari jumlah data, tipe data, memory yang digunakan dll.
#Dapat dilihat bahwa seluruh data sudah di dalam bentuk numerik.
df.info()

#Melihat statistical description dari data mulai dari mean, kuartil, standard deviation dll.
df.describe()

#Merubah tipe data dari bathrooms yang semula float menjadi int.
df['bathrooms'] = df['bathrooms'].astype('int')
#Mengganti nilai 33 menjadi 3.
df['bedrooms'] = df['bedrooms'].replace(33,3)

#Mencari dan menangani missing values.
#Ternyata datanya sudah tidak ada missing values.
df.isnull().sum()

#Univariate analysis bedrooms.
f = plt.figure(figsize=(12,4))
f.add_subplot(1,2,1)
sns.countplot(df['bedrooms'])
f.add_subplot(1,2,2)
plt.boxplot(df['bedrooms'])
plt.show()

#Univariate analysis bathrooms.

f = plt.figure(figsize=(12,4))
f.add_subplot(1,2,1)
sns.countplot(df['bathrooms'])
f.add_subplot(1,2,2)
plt.boxplot(df['bathrooms'])
plt.show()

#Univariate analysis sqft_living.

f = plt.figure(figsize=(12,4))
f.add_subplot(1,2,1)
df['sqft_living'].plot(kind='kde')
f.add_subplot(1,2,2)
plt.boxplot(df['sqft_living'])
plt.show()

#Univariate analysis grade.

f = plt.figure(figsize=(12,4))
f.add_subplot(1,2,1)
sns.countplot(df['grade'])
f.add_subplot(1,2,2)
plt.boxplot(df['grade'])
plt.show()

#Univariate analysis yr_built.

f = plt.figure(figsize=(20,8))
f.add_subplot(1,2,1)
sns.countplot(df['yr_built'])
f.add_subplot(1,2,2)
plt.boxplot(df['yr_built'])
plt.show()

#Bivariate analysis antara independent variable dan dependent variable.

plt.figure(figsize=(10,8))
sns.pairplot(data=df, x_vars=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built'], y_vars=['price'], size=5, aspect=0.75)

#Mengetahui nilai korelasi dari independent variable dan dependent variable.
df.corr().style.background_gradient().set_precision(2)

#Pertama, buat variabel x dan y.
x = df.drop(columns='price')
y = df['price']
#Kedua, ucup split data menjadi training and testing dengan porsi 80:20.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
#Ketiga, ucup bikin object linear regresi.
lin_reg = LinearRegression()
#Keempat, train the model menggunakan training data yang sudah displit.
lin_reg.fit(x_train, y_train)
#Kelima, cari tau nilai slope/koefisien (m) dan intercept (b).
print(lin_reg.coef_)
print(lin_reg.intercept_)
#Keenam, cari tahu accuracy score dari model menggunakan testing data yang sudah displit.
lin_reg.score(x_test, y_test)

#Prediksi harga rumah idaman Joko.
lin_reg.predict([[3,2,1800,7,1990]])
