# STAT1013-final-projet
This is the final project of STAT1013

---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell markdown" id="9xZnRXM7x0Cv">

# CUHK-STAT1013: Practical Assignment Part 1: Sharing Your Idea and Data

</div>

<div class="cell markdown" id="9Fy05KAkyJI0">

## Diabetes dataset background

**Description**:

This dataset is originally from the National Institute of Diabetes and
Digestive and Kidney Diseases, describing the health condition of
individual patient. The objective of the dataset is to diagnostically
predict whether a patient has diabetes, based on certain diagnostic
measurements included in the dataset. In particular, all patients here
are females at least 21 years old of Pima Indian heritage.

**Github**:
<https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset/blob/master/diabetes.csv>

**Sample size**: 768

**Feature documentation**:

| Feature                  | Class      | Dtype   |
|:-------------------------|:-----------|:--------|
| Pregnancies              | Tensor     | int64   |
| Glucose                  | Tensor     | int64   |
| BloodPressure            | Tensor     | int64   |
| SkinThickness            | Tensor     | int64   |
| Insulin                  | ClassLabel | int64   |
| BMI                      | Tensor     | float64 |
| DiabetesPedigreeFunction | Tensor     | float64 |
| Age                      | Tensor     | int64   |
| Outcome                  | Tensor     | int64   |

</div>

<div class="cell markdown" id="k85zO7zxys4H">

## Hypothesis

-   Tell us what your idea is and why you have chosen to pursue this
    idea.
    -   We are interested in "**Does patients with diabetes have a
        higher glucose level compared with the patients without
        diabetes?**", so that we can predict whether the patients has
        the potential to get diabetes by investigating the glucose
        level.
-   What two groups you are comparing:
    -   **G1**: Patients with diabetes; **G2**: Patients without
        diabetes
    -   `Outcome == 1` means patient has diabetes, and `Outcome ==0`
        means patient doesn NOT have diabetes.
-   What you will be measuring (i.e., what your response variable will
    be)
    -   `Glucose`
-   Is your response variable quantitative rather than categorical?
    -   `Glucose` is quantitative variable.
-   Make a prediction about what kind of difference you expect to see
    between your samples and WHY.
    -   We'd expect that **G1** \> **G2** since [high glucose level is
        most often linked with
        diabetes.](https://my.clevelandclinic.org/health/diseases/9815-hyperglycemia-high-blood-sugar#:~:text=Hyperglycemia%20(high%20blood%20glucose)%20means,lead%20to%20serious%20health%20problems.)
-   Talk about how you will gather your data
    -   From Github link:
        <https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset/blob/master/diabetes.csv>
-   If you had unlimited resources (time, money, staff, etc.) how would
    you collect your data?
    -   \(i\) Attempt to collect more data from more institues; (ii)
        investigate if the provided dataset is a good random sampling
        subset of the diabetes population.

</div>

<div class="cell markdown" id="3GOdPWT03PQB">

## Prepare your dataset

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:206}"
id="mUxJb4hxvpHQ" outputId="e1464c04-c3fa-4cdf-f4e2-8e0f6e5e530b">

``` python
## load dataset from github

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')
df.head(5)
```

<div class="output execute_result" execution_count="12">

       Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
    0            6      148             72             35        0  33.6   
    1            1       85             66             29        0  26.6   
    2            8      183             64              0        0  23.3   
    3            1       89             66             23       94  28.1   
    4            0      137             40             35      168  43.1   

       DiabetesPedigreeFunction  Age  Outcome  
    0                     0.627   50        1  
    1                     0.351   31        0  
    2                     0.672   32        1  
    3                     0.167   21        0  
    4                     2.288   33        1  

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="-3kCoL459y57" outputId="62edc891-dbea-4c55-aaa7-0b52e3d77344">

``` python
df.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB

</div>

</div>

<div class="cell markdown" id="55xAIxVa3hpQ">

-   Tell us what groups you want to compare in the dataset
    -   **G1** (Outcome = 1) vs. **G2** (Outcome = 0)
    -   Compare the average of Glucose level of the two groups: **G1**:
        Patients with diabetes, and **G2**: Patients without diabetes

</div>

<div class="cell markdown" id="13PdL3ht3902">

-   Print first 5 records of each group, respectively.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:206}"
id="UNL0WXav3hLj" outputId="a73a543f-5400-4d2b-ef64-5395a9c574f2">

``` python
## First 5 records of G1 (Patients with diabetes)
(df[df['Outcome'] == 1]).head(5)
```

<div class="output execute_result" execution_count="36">

       Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
    0            6      148             72             35        0  33.6   
    2            8      183             64              0        0  23.3   
    4            0      137             40             35      168  43.1   
    6            3       78             50             32       88  31.0   
    8            2      197             70             45      543  30.5   

       DiabetesPedigreeFunction  Age  Outcome  
    0                     0.627   50        1  
    2                     0.672   32        1  
    4                     2.288   33        1  
    6                     0.248   26        1  
    8                     0.158   53        1  

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:206}"
id="dhe52HVB4T1O" outputId="1651824f-52a4-41f8-fd5c-916deac4c5e4">

``` python
## First 5 records of G2 (Patients without diabetes)
(df[df['Outcome'] == 0]).head(5)
```

<div class="output execute_result" execution_count="37">

        Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
    1             1       85             66             29        0  26.6   
    3             1       89             66             23       94  28.1   
    5             5      116             74              0        0  25.6   
    7            10      115              0              0        0  35.3   
    10            4      110             92              0        0  37.6   

        DiabetesPedigreeFunction  Age  Outcome  
    1                      0.351   31        0  
    3                      0.167   21        0  
    5                      0.201   30        0  
    7                      0.134   29        0  
    10                     0.191   30        0  

</div>

</div>

<div class="cell code" id="zEgfWXaKGvNC">

``` python
## Any other data description and visualization you want to add.

## Open question, be flexible and no example can be provided.
```

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:300}"
id="v4T5whgBJVEA" outputId="2c9e616f-0d95-4dc8-e77b-10ed459d43f5">

``` python
df[df['Outcome'] == 1].describe(include='all')
```

<div class="output execute_result" execution_count="54">

           Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin  \
    count   268.000000  268.000000     268.000000     268.000000  268.000000   
    mean      4.865672  141.257463      70.824627      22.164179  100.335821   
    std       3.741239   31.939622      21.491812      17.679711  138.689125   
    min       0.000000    0.000000       0.000000       0.000000    0.000000   
    25%       1.750000  119.000000      66.000000       0.000000    0.000000   
    50%       4.000000  140.000000      74.000000      27.000000    0.000000   
    75%       8.000000  167.000000      82.000000      36.000000  167.250000   
    max      17.000000  199.000000     114.000000      99.000000  846.000000   

                  BMI  DiabetesPedigreeFunction         Age  Outcome  
    count  268.000000                268.000000  268.000000    268.0  
    mean    35.142537                  0.550500   37.067164      1.0  
    std      7.262967                  0.372354   10.968254      0.0  
    min      0.000000                  0.088000   21.000000      1.0  
    25%     30.800000                  0.262500   28.000000      1.0  
    50%     34.250000                  0.449000   36.000000      1.0  
    75%     38.775000                  0.728000   44.000000      1.0  
    max     67.100000                  2.420000   70.000000      1.0  

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:300}"
id="fNFQ9yPsLp-t" outputId="41f79a9a-9f5b-4bc3-9d71-b6d4e9cf42ee">

``` python
df[df['Outcome'] == 0].describe(include='all')
```

<div class="output execute_result" execution_count="55">

           Pregnancies   Glucose  BloodPressure  SkinThickness     Insulin  \
    count   500.000000  500.0000     500.000000     500.000000  500.000000   
    mean      3.298000  109.9800      68.184000      19.664000   68.792000   
    std       3.017185   26.1412      18.063075      14.889947   98.865289   
    min       0.000000    0.0000       0.000000       0.000000    0.000000   
    25%       1.000000   93.0000      62.000000       0.000000    0.000000   
    50%       2.000000  107.0000      70.000000      21.000000   39.000000   
    75%       5.000000  125.0000      78.000000      31.000000  105.000000   
    max      13.000000  197.0000     122.000000      60.000000  744.000000   

                  BMI  DiabetesPedigreeFunction         Age  Outcome  
    count  500.000000                500.000000  500.000000    500.0  
    mean    30.304200                  0.429734   31.190000      0.0  
    std      7.689855                  0.299085   11.667655      0.0  
    min      0.000000                  0.078000   21.000000      0.0  
    25%     25.400000                  0.229750   23.000000      0.0  
    50%     30.050000                  0.336000   27.000000      0.0  
    75%     35.300000                  0.561750   37.000000      0.0  
    max     57.300000                  2.329000   81.000000      0.0  

</div>

</div>

<div class="cell markdown" id="3X5IKJ8kLvp7">

From the above table, we can see that the mean of glucose level
(141.257) of patienets with diabetes is higher than that (109.98) of
patienets without diabetes.

</div>

<div class="cell code" id="TxobOpJ6Ji01">

``` python
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [10, 5]

sns.set()
```

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:339}"
id="ZePCsZ5hI9IM" outputId="2fdc8a9b-3fab-4452-acc9-eae67546e9d5">

``` python
## Show the distribution of the data according to the glucose level
sns.violinplot(data=df, x='Glucose')
plt.show()
```

<div class="output display_data">

![](2bade63962def15f5e348e0cb2cf4f364f6fa43f.png)

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:339}"
id="iZ814wTFKf4F" outputId="e65ad038-6b29-4354-82d9-29cf02aea683">

``` python
## Show the ditribution of glucose level based on whether the patients have diavetes or not
sns.stripplot(data=df, x='Glucose', y='Outcome')
plt.show()
```

<div class="output display_data">

![](8ea3736f3056126a331c620f45429587a7e09285.png)

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:339}"
id="c_kC0vhkLHa1" outputId="38f20df6-e1a9-452c-a0c5-ca5c407908e9">

``` python
sns.violinplot(data=df, x='Outcome',y='Glucose')
plt.show()
```

<div class="output display_data">

![](a132e7f2f4155332dfe3326e7c3b5b716052a9ac.png)

</div>

</div>

<div class="cell markdown" id="n9ojxpp6LQNj">

From the above graph, we can see that the median of glucose level of
patients with diabetes is higher than that of patients without diabetes.
And the overall glucose level distribution for the patient with diabetes
is higher.

</div>
