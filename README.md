
# Financial Distress Prediction

## 1. Introduction

Predictive algorithms are increasingly used in business today in order to gain competitive advantages. With a specific problem and appropriate datasets, they helps business to forecast what is the best action course to follow. Decisions, thus, are made based on data-driven approach (rather than "intuition" and "experience").

In this report, we are going to use a Kaggle dataset (see [1]) to predict financial distress of a person, given their history, thus, facilitating the bank to decide if they should proceed with this person.

## 2. Data Pre-processing


```python
# Necessary imports.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
```


```python
# Necessary configs.
# Disable printing scientific notation.
np.set_printoptions(suppress=True)
```


```python
train_dataset = pd.read_csv("cs-training.csv", header = "infer", sep = ",", encoding = "utf-8")
```


```python
train_dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>SeriousDlqin2yrs</th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.766127</td>
      <td>45</td>
      <td>2</td>
      <td>0.802982</td>
      <td>9120.0</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0.957151</td>
      <td>40</td>
      <td>0</td>
      <td>0.121876</td>
      <td>2600.0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0.658180</td>
      <td>38</td>
      <td>1</td>
      <td>0.085113</td>
      <td>3042.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0.233810</td>
      <td>30</td>
      <td>0</td>
      <td>0.036050</td>
      <td>3300.0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>0.907239</td>
      <td>49</td>
      <td>1</td>
      <td>0.024926</td>
      <td>63588.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



The original dataset also includes "Data Dictionary.xls", which describes the features. The data needs to be double-checked if it follows the format expectation.

* **SeriousDlqin2yrs**: Person experienced 90 days past due delinquency or worse. Yes/No.
  + Check: The values should contain either 1 or 0.


```python
np.sort(train_dataset.loc[:, 'SeriousDlqin2yrs'].unique())
```




    array([0, 1], dtype=int64)



* **RevolvingUtilizationOfUnsecuredLines**: Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits. Percentage.
  + Check: Positive real numbers from 0 to 1.


```python
np.sort(train_dataset.loc[:, 'RevolvingUtilizationOfUnsecuredLines'].unique())
```




    array([     0.        ,      0.00000837,      0.00000993, ...,
            22198.        ,  29110.        ,  50708.        ])



This seems strange! "Data Dictionary.xls" explained the meaning behind this feature, but not how this feature is presented. The Kaggle Discussion Forum for this dataset (see [1], [2]) gave no actionable clue on this.


```python
((train_dataset.loc[:, 'RevolvingUtilizationOfUnsecuredLines'] > 1).sum() / train_dataset.shape[0])
```




    0.02214



About 2.21% of data have this issue. There is no clear explanation of how the data is computed. So, it is advised to keep "as is". The good news is this column contains neither NULL/NA/NaN/Blank nor categorical data.

* **age**: Age of borrower in years. Integer.
  + Check: Positive natural numbers from 1 to, let's say, 122. See [4].


```python
np.sort(train_dataset.loc[:, 'age'].unique())
```




    array([  0,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
            33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
            46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,
            59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
            72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,
            85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
            98,  99, 101, 102, 103, 105, 107, 109], dtype=int64)



The "0" age is likely erroneous, partly because no one can be 0 year old, and also because we can see an upward trend incremented by 1 from 21 (e.g.: 21, 22, 23), meaning that from 0 to 21, there is a big jump in value for no reason. So, rows associated with age "0" should be removed. However, let's first check how many percent of data are affected by this.


```python
(train_dataset.loc[:, 'age'] == 0).sum()
```




    1



The whole dataset has this number of rows:


```python
train_dataset.shape[0]
```




    150000



No need for second thought. This case must be filtered out.


```python
train_dataset = train_dataset[train_dataset.loc[:, 'age'] != 0]
```

* **NumberOfTime30-59DaysPastDueNotWorse**: Number of times borrower has been 30-59 days past due but no worse in the last 2 years. Integer.
  + Check: Positive natural numbers, including 0.


```python
np.sort(train_dataset.loc[:, 'NumberOfTime30-59DaysPastDueNotWorse'].unique())
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 96, 98], dtype=int64)



* **DebtRatio**: Monthly debt payments, alimony,living costs divided by monthy gross income. Percentage.
  + Check: Positive real numbers from 0 to 1.


```python
np.sort(train_dataset.loc[:, 'DebtRatio'].unique())
```




    array([      0.       ,       0.000026 ,       0.0000369, ...,
            307001.       ,  326442.       ,  329664.       ])



It looks like this issue is similar to the column "RevolvingUtilizationOfUnsecuredLines", where the percentage is expected to be within the range of [0,1]. Again, there is no clue in the Dataset description or Kaggle Forum on treating this glitch. So it is kept "as is".


```python
# For informational purpose.
((train_dataset.loc[:, 'DebtRatio'] > 1).sum() / train_dataset.shape[0])
```




    0.23424822832152215



* **MonthlyIncome**: Monthly income. Real.
  + Check: Positive real numbers.


```python
np.sort(train_dataset.loc[:, 'MonthlyIncome'].unique())
```




    array([       0.,        1.,        2., ...,  1794060.,  3008750.,
                 nan])



Because of having **nan**, let's check how many percents of data there are that have nan in this column:


```python
train_dataset.loc[:, 'MonthlyIncome'].isna().sum() / train_dataset.shape[0]
```




    0.19820798805325368



About 20% of data within this column is NaN. Therefore, it cannot just be removed, and must be imputed. There are many methods for data imputation. For example, using Regression Model (Build a Regression model, given the remaining variables. Then, make use of that model to predict the missing values). In this case, a simpler approach is preferred: calculate the mean value within the column, and replace the missing values with that mean value.


```python
from sklearn.preprocessing import Imputer
imp_monthly_income = Imputer(missing_values=np.nan, strategy='mean')
non_nan_values = imp_monthly_income.fit_transform(train_dataset.loc[:, ['MonthlyIncome']])
train_dataset.loc[:, ['MonthlyIncome']] = non_nan_values
# Double check
np.isnan(non_nan_values).sum()
```




    0



* **NumberOfOpenCreditLinesAndLoans**: Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards). Integer.
  + Check: Positive natural numbers.


```python
np.sort(train_dataset.loc[:, 'NumberOfOpenCreditLinesAndLoans'].unique())
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
           51, 52, 53, 54, 56, 57, 58], dtype=int64)



* **NumberOfTimes90DaysLate**: Number of times borrower has been 90 days or more past due.
  + Check: Positive natural numbers.


```python
np.sort(train_dataset.loc[:, 'NumberOfTimes90DaysLate'].unique())
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 17,
           96, 98], dtype=int64)



* **NumberRealEstateLoansOrLines**: Number of mortgage and real estate loans including home equity lines of credit. Integer.
  + Check: Positive natural numbers.


```python
np.sort(train_dataset.loc[:, 'NumberRealEstateLoansOrLines'].unique())
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 23, 25, 26, 29, 32, 54], dtype=int64)



* **NumberOfTime60-89DaysPastDueNotWorse**: Number of times borrower has been 60-89 days past due but no worse in the last 2 years. Integer.
  + Check: Positive natural numbers.


```python
np.sort(train_dataset.loc[:, 'NumberOfTime60-89DaysPastDueNotWorse'].unique())
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11, 96, 98], dtype=int64)



* **NumberOfDependents**: Number of dependents in family excluding themselves (spouse, children etc.). Integer.
  + Check: Positive natural numbers.


```python
np.sort(train_dataset.loc[:, 'NumberOfDependents'].unique())
```




    array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
            13.,  20.,  nan])




```python
train_dataset.loc[:, 'NumberOfDependents'].isna().sum() / train_dataset.shape[0]
```




    0.026160174401162674



Again, it would be better if the nan values can be replaced with average values.


```python
imp_number_of_dependent = Imputer(missing_values=np.nan, strategy='mean')
non_nan_values = imp_number_of_dependent.fit_transform(train_dataset.loc[:, ['NumberOfDependents']])
train_dataset.loc[:, ['NumberOfDependents']] = non_nan_values
# Double check
np.isnan(non_nan_values).sum()
```




    0



Lastly, the first column (index column) has no contributions to building the model, so it should be removed.


```python
train_dataset.drop(train_dataset.columns[0], axis=1, inplace=True)
```


```python
train_dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeriousDlqin2yrs</th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.766127</td>
      <td>45</td>
      <td>2</td>
      <td>0.802982</td>
      <td>9120.0</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.957151</td>
      <td>40</td>
      <td>0</td>
      <td>0.121876</td>
      <td>2600.0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.658180</td>
      <td>38</td>
      <td>1</td>
      <td>0.085113</td>
      <td>3042.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.233810</td>
      <td>30</td>
      <td>0</td>
      <td>0.036050</td>
      <td>3300.0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.907239</td>
      <td>49</td>
      <td>1</td>
      <td>0.024926</td>
      <td>63588.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Visibly, the target value (dependent variable) is "SeriousDlqin2yrs". Remaining independent variables are going to be used to predict/classify the target value.

## 3. Data Exploration

Some statistics:


```python
train_dataset.shape
```




    (149999, 11)




```python
train_dataset.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeriousDlqin2yrs</th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>149999.000000</td>
      <td>149999.000000</td>
      <td>149999.000000</td>
      <td>149999.000000</td>
      <td>149999.000000</td>
      <td>1.499990e+05</td>
      <td>149999.000000</td>
      <td>149999.000000</td>
      <td>149999.000000</td>
      <td>149999.000000</td>
      <td>149999.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.066840</td>
      <td>6.048472</td>
      <td>52.295555</td>
      <td>0.421029</td>
      <td>353.007426</td>
      <td>6.670227e+03</td>
      <td>8.452776</td>
      <td>0.265975</td>
      <td>1.018233</td>
      <td>0.240388</td>
      <td>0.757214</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.249746</td>
      <td>249.756203</td>
      <td>14.771298</td>
      <td>4.192795</td>
      <td>2037.825113</td>
      <td>1.288049e+04</td>
      <td>5.145964</td>
      <td>4.169318</td>
      <td>1.129772</td>
      <td>4.155193</td>
      <td>1.100403</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.029867</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.175074</td>
      <td>3.903000e+03</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.154176</td>
      <td>52.000000</td>
      <td>0.000000</td>
      <td>0.366503</td>
      <td>6.600000e+03</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.559044</td>
      <td>63.000000</td>
      <td>0.000000</td>
      <td>0.868257</td>
      <td>7.400000e+03</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>50708.000000</td>
      <td>109.000000</td>
      <td>98.000000</td>
      <td>329664.000000</td>
      <td>3.008750e+06</td>
      <td>58.000000</td>
      <td>98.000000</td>
      <td>54.000000</td>
      <td>98.000000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
</div>



One way to potentially improve the performance of a machine learning model is to have non-correlated features. More features also require more computation and system resources (RAM, CPU, etc.) Let's check the correlation between those features to see if we can remove any of them.


```python
corr = train_dataset.loc[:, train_dataset.columns[1:]].corr().as_matrix()
corr
```




    array([[ 1.        , -0.00589891, -0.00131348,  0.00396118,  0.006565  ,
            -0.01128081, -0.00106125,  0.00623485, -0.00104784,  0.00153893],
           [-0.00589891,  1.        , -0.06299453,  0.02418473,  0.0329842 ,
             0.14770035, -0.06100876,  0.03317225, -0.05716322, -0.20808514],
           [-0.00131348, -0.06299453,  1.        , -0.00654176, -0.0076355 ,
            -0.05531184,  0.98360282, -0.03056541,  0.98700557, -0.00252634],
           [ 0.00396118,  0.02418473, -0.00654176,  1.        , -0.00535496,
             0.04956478, -0.00831971,  0.12004734, -0.00753317, -0.0382864 ],
           [ 0.006565  ,  0.0329842 , -0.0076355 , -0.00535496,  1.        ,
             0.0823188 , -0.00948358,  0.11382389, -0.008259  ,  0.05854302],
           [-0.01128081,  0.14770035, -0.05531184,  0.04956478,  0.0823188 ,
             1.        , -0.07998461,  0.43396279, -0.07107697,  0.06451089],
           [-0.00106125, -0.06100876,  0.98360282, -0.00831971, -0.00948358,
            -0.07998461,  1.        , -0.04520496,  0.99279618, -0.00957878],
           [ 0.00623485,  0.03317225, -0.03056541,  0.12004734,  0.11382389,
             0.43396279, -0.04520496,  1.        , -0.0397221 ,  0.123364  ],
           [-0.00104784, -0.05716322,  0.98700557, -0.00753317, -0.008259  ,
            -0.07107697,  0.99279618, -0.0397221 ,  1.        , -0.01027686],
           [ 0.00153893, -0.20808514, -0.00252634, -0.0382864 ,  0.05854302,
             0.06451089, -0.00957878,  0.123364  , -0.01027686,  1.        ]])




```python
def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');
    plt.yticks(range(len(corr.columns)), corr.columns);
```


```python
%matplotlib inline
plot_corr(train_dataset.loc[:, train_dataset.columns[1:]], 6)
```


![png](output_60_0.png)


High correlation can be seen between those columns: "NumberOfTimes90DaysLate", "NumberOfTime30-59DaysPastDueNotWorse" and "NumberOfTime60-89DaysPastDueNotWorse". Intuitively, by reading the feature names, one can easily understand why they are highly correlated. There is still a slight chance that even those having high correlation can turn out to be very different. See [5] for "**Anscombe's quartet**". Let's draw some plots to feel the data better.


```python
%matplotlib inline
temp_data_frame = train_dataset.loc[:, ['NumberOfTimes90DaysLate', 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse']]

def drawSubplot(x, y):
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, y, '.')
    plt.plot(x, m*x + b, '-')
    plt.xlabel(x.name, fontsize=10)
    plt.ylabel(y.name, fontsize=10)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,6))

# NumberOfTimes90DaysLate vs. NumberOfTime30-59DaysPastDueNotWorse
plt.subplot(1, 3, 1)
drawSubplot(temp_data_frame.iloc[:, 0], temp_data_frame.iloc[:, 1])

# NumberOfTimes90DaysLate vs. NumberOfTime60-89DaysPastDueNotWorse
plt.subplot(1, 3, 2)
drawSubplot(temp_data_frame.iloc[:, 0], temp_data_frame.iloc[:, 2])

# NumberOfTime30-59DaysPastDueNotWorse vs. NumberOfTime60-89DaysPastDueNotWorse
plt.subplot(1, 3, 3)
drawSubplot(temp_data_frame.iloc[:, 1], temp_data_frame.iloc[:, 2])

plt.show()
```


![png](output_62_0.png)


Although the points are not perfectly collinear (lying on a single line), it is understandable that they have high correlation as when a feature has a small value, the other feature also has an approximately small value. The all 3 features seemingly contribute equally to the to-be-built model.

In short, given the limited description on features, we choose to keep "NumberOfTimes90DaysLate" and remove the other two.


```python
train_dataset.drop(['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse'], axis=1, inplace=True)
```

The **clean** data is ready to feed into machine learning algorithms.


```python
train_dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeriousDlqin2yrs</th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.766127</td>
      <td>45</td>
      <td>0.802982</td>
      <td>9120.0</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.957151</td>
      <td>40</td>
      <td>0.121876</td>
      <td>2600.0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.658180</td>
      <td>38</td>
      <td>0.085113</td>
      <td>3042.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.233810</td>
      <td>30</td>
      <td>0.036050</td>
      <td>3300.0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.907239</td>
      <td>49</td>
      <td>0.024926</td>
      <td>63588.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_dataset.dtypes
```




    SeriousDlqin2yrs                          int64
    RevolvingUtilizationOfUnsecuredLines    float64
    age                                       int64
    DebtRatio                               float64
    MonthlyIncome                           float64
    NumberOfOpenCreditLinesAndLoans           int64
    NumberOfTimes90DaysLate                   int64
    NumberRealEstateLoansOrLines              int64
    NumberOfDependents                      float64
    dtype: object



## 4. Model Building

The most optimal machine learning algorithm for a specific problem might not be figured out without trial-and-error. One algorithm is good for this problem and/or this dataset can turn out to be less ideal for other problem and/or other datasets. Linear Regression is good if the decision boundary is linearly separable. Decision Tree, in the other hand, assumes the boundary is parallel to axes ("features").

In reality, in order to achieve high performance, one should focus on, first and foremost, data quality and quantity. In general, the more data, the better. Secondly, using domain knowledge to engineer promising features for the specific business problem. After that, one reasonably uses a variety of Machine Learning algorithms to build models. Each algorithm involves many hyper-parameters to be tuned. Carrying out a lot of trial-and-error; eventually, an optimal algorithm (with its optimal hyper-parameters) begins to surface.

In this case, **Random Forest (RF)** will be employed to tackle our prediction problem. RF is well-known to give decent performance. It can run in parallel effectively (each tree is built separately). RF is also less proned to over-fitting.


```python
rfc = RandomForestClassifier(n_jobs=-1, oob_score = True)

"""
See [7] for all tunable hyper-parameters.
See [8] for recommended hyper-parameters.

n_estimators: The number of trees in the forest => The more, the better, but more resource usage.
max_features: The number of features to consider when looking for the best split.
criterion: The function to measure the quality of a split.
max_depth: The maximum depth of the tree => The deeper, the more overfitting (capture noise!)
"""
param_grid = { 
           "n_estimators" : [50, 100],
           "max_features" : [3, 6],
           "criterion" : ["gini", "entropy"],
           "max_depth": [2, 4, 6]}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv = 5)

CV_rfc.fit(train_dataset.iloc[:, 1:], train_dataset.iloc[:, 0])
print(CV_rfc.best_params_)
```

    {'criterion': 'gini', 'max_depth': 6, 'max_features': 6, 'n_estimators': 50}
    

Given the found optimal hyper-parameters {'criterion': 'gini', 'max_depth': 6, 'max_features': 6, 'n_estimators': 100}, we start building the final RF model.


```python
rf_final = RandomForestClassifier(n_jobs=-1, criterion='gini', max_features=6, max_depth=6, n_estimators=100, oob_score=True)
rf_final.fit(train_dataset.iloc[:, 1:], train_dataset.iloc[:, 0])
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=6, max_features=6, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                oob_score=True, random_state=None, verbose=0, warm_start=False)



This RF model can be used to predict the target value in the training dataset. However, one should expect that the accuracy will be **optimistically high** as the model **learned through the training dataset**!


```python
preds = rf_final.predict(train_dataset.iloc[:, 1:])
```

Confusion matrix:


```python
pd.crosstab(train_dataset.iloc[:, 0], preds, rownames=['Actual'], colnames=['Predicted'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>139077</td>
      <td>896</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8478</td>
      <td>1548</td>
    </tr>
  </tbody>
</table>
</div>



The accruracy is: ~93.75%! This accuracy is not representative for the RF model despite that this model is the final one, and will be used against the test set. The **appropriate accuracy of the model upon the training set** should be acquired from Cross Validation, specifically from the GridSearchCV in our case.


```python
# Mean cross-validated score of the best_estimator
print(CV_rfc.best_score_)
```

    0.936152907686
    

## 5. Validation

The original dataset is already split into train/test. "cs-training.csv" for training, which has been used above. "cs-test.csv" for testing. Before validation, the test data must also be cleaned in the same way the training data passed through. That is,

* Remove first index column.
* Remove 'NumberOfTime30-59DaysPastDueNotWorse' and 'NumberOfTime60-89DaysPastDueNotWorse'.
* Impute NaN in 'MonthlyIncome' and 'NumberOfDependents' **using the Imputers created**.

However, in the test set, there is no label for "SeriousDlqin2yrs". So, we cannot measure the accuracy of the RF model upon the test set. Normally, this accuracy is the **reported** accuracy. If both numbers are close to each other, the generalization goal is achieved.

Note: The dataset "GiveMeSomeCredit" is given as a contest in Kaggle. Labels in test set, thus, are hidden.

## 6. Summary

The business challenge of GiveMeSomeCredit (Kaggle) is to predict if a person will have financial distress in the future. Thus, if a yes, the bank should not give a loan to that person as they will not be able to pay back, making a loss for the bank. Originally, the dataset has 10 features. There are 3 features that are highly correlated; so we removed two. The machine learning algorithm used to solve this prediction problem is Random Forest, which is well-known to be decent in terms of performance.

Instead of separating the training dataset into "train/validation", we used Cross Validation (5 folds) to better estimate the model performance, given a set of hyper-parameters. GridSearchCV was used to seek optimal parameters. Although Random Forest has many tunable parameters, we focus on most common ones (`n_estimators`, `max_features`, `criterion`, `max_depth`).

The accuracy of the built RF model:
* For training set (5-Folds CV): 93.6%
* For testing set: labels not available offline.

## 7. Future Works

* GridSearchCV with more parameters for the `param_grid`.
  + It is suggested that `n_estimators` should be {500, 1000}.
  + Experiment with `min_samples_leaf` to reduce over-fitting.
* Try XGBoost as it is popular within Kaggle Community. (See [9])

## References

[1] https://www.kaggle.com/c/GiveMeSomeCredit

[2] https://www.kaggle.com/c/GiveMeSomeCredit/discussion/918

[3] https://www.kaggle.com/c/GiveMeSomeCredit/discussion/874

[4] https://en.wikipedia.org/wiki/Oldest_people

[5] https://en.wikipedia.org/wiki/Anscombe%27s_quartet

[6] https://en.wikipedia.org/wiki/No_free_lunch_theorem

[7] http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

[8] https://stackoverflow.com/questions/36107820/how-to-tune-parameters-in-random-forest-using-scikit-learn

[9] https://en.wikipedia.org/wiki/Xgboost

## Appendix

* My computer's resource usage when GridSearchCV is running. Note that we used `n_jobs=-1` to enable Python to exploit all CPU cores available.

![CPU](CPU.png)
