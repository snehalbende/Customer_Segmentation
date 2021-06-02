# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:29:55 2021

@author: sneha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import scipy
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import datetime as dt
data = pd.read_csv('C:/Users/sneha/Desktop/masters/project/online retail/online_retail.csv',encoding = 'unicode_escape')
data.columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate','Price','Customer ID','Country']
data.info()
data.head()
data.describe()

data.corr()
sns.heatmap(data.corr())
data.shape


df = data.copy()
#df.columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate','Price','Customer_id','Country']
df.head()


data1 = data[['Country','Customer ID']].drop_duplicates()

data1.groupby(['Country'])['Customer ID'].aggregate('count').reset_index().sort_values('Customer ID', ascending = False)

df = data.query("Country == 'United Kingdom'").reset_index(drop=True)
df.head()
df.info()
#df.rename(columns = {0 :'InvoiceNo'}, inplace = True)
df.info()
# data cleaning 
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# checking if missing values are present in the dataset
df.isnull().sum()
df.describe()

df = df[pd.notnull(df['Customer ID'])]
df.shape
# removing outliers from the dataset
df = df[df.Quantity > 0]

sns.boxplot(x=df["Quantity"])
df = df[df.Quantity < 8000 ]
sns.boxplot(x=df["Quantity"])
df = df[df.Price < 8000 ]
sns.boxplot(x=df["Price"])
df.shape


# calculating total price of the item 

df['totalPrice'] = df['Quantity'] * df['Price']


#Calculating the RFM values 
recent = dt.datetime(2011,12,10)

score = df.groupby('Customer ID').agg({'InvoiceDate' : lambda x :(recent - x.max()).days,
                                       'Invoice': lambda x: len(x),
                                       'totalPrice' : lambda x:x.sum()})

#Convert Invoice Date into type int
score['InvoiceDate'] = score['InvoiceDate'].astype(int)

score.rename(columns={'InvoiceDate': 'Recency', 
                         'Invoice': 'Frequency', 
                         'totalPrice': 'Monetary'}, inplace=True)

score.reset_index().head()
score.describe()

# checking the distribution and skewness of RFM values
p = score['Recency']
ax = sns.displot(p)
p1 =score.query('Frequency <1000')['Frequency']
ax = sns.displot(p1)
p2 = score.query('Monetary < 10000')['Monetary']
ax = sns.displot(p2)

#Splitting the segments
q1 = score.quantile(q=[0.2, 0.4, 0.6, 0.8])
q1 = q1.to_dict()

def r_score(x):
    if x <= q1['Recency'][0.2]:
        return 5
    elif x <= q1['Recency'][0.4]:
        return 4
    elif x <= q1['Recency'][0.6]:
        return 3
    elif x <= q1['Recency'][0.8]:
        return 2
    else:
        return 1

def fm_score(x, c):
    if x <= q1[c][0.2]:
        return 1
    elif x <= q1[c][0.4]:
        return 2
    elif x <= q1[c][0.6]:
        return 3
    elif x <= q1[c][0.8]:
        return 4
    else:
        return 5   
    
    
score['R'] = score['Recency'].apply(lambda x: r_score(x))
score['F'] = score['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
score['M'] = score['Monetary'].apply(lambda x: fm_score(x, 'Monetary'))

#Calculate and Add RFMGroup value column showing combined concatenated score of RFM
score['RFM_Score'] = score.R.map(str) + score.F.map(str) + score.M.map(str)


map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t loose',
    r'3[1-2]': 'About to sleep',
    r'33': 'Need attention',
    r'[3-4][4-5]': 'loyal customers',
    r'41': 'Promising',
    r'51': 'New customers',
    r'[4-5][2-3]': 'Potential loyalists',
    r'5[4-5]': 'Champions'
}

score['Loyalty_level'] = score['R'].map(str) + score['F'].map(str)
score['Loyalty_level'] = score['Loyalty_level'].replace(map, regex=True)


# counting number of customers for each loyalty level

count = score['Loyalty_level'].value_counts().sort_values(ascending = True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(count)),count,color='lightcoral')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(count)))
ax.set_yticklabels(count.index)

for i, bar in enumerate(bars):
        value = bar.get_width()
        
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/count.sum())),
                va='center',
                ha='left'
               )

plt.show()




# normalizing data before applying machine learning algorithm


def z(n):
    if n <= 0:
        return 1
    else:
        return n

score['Recency'] = [z(x) for x in score.Recency]
score['Monetary'] = [z(x) for x in score.Monetary]
log = score[['Recency','Frequency','Monetary']].apply(np.log,axis = 1).round(3)



sc = StandardScaler()
final_data = sc.fit_transform(log)
final_data = pd.DataFrame(final_data, index = score.index, columns = log.columns)


# # determinibg optimal k value using elbow method
err = {}
for i in range(1,20):
    model = KMeans(n_clusters= i, init= 'k-means++', max_iter= 1000)
    model.fit(final_data)
    err[i] = model.inertia_


plt.figure(figsize=(12, 6))
sns.pointplot(x = list(err.keys()), y = list(err.values()))
plt.title('Finding optimal k value for clusters')
plt.xlabel('K Value')
plt.ylabel('Number of clusters')

Km_model = KMeans(n_clusters= 3, init= 'k-means++', max_iter= 2000)
Km_model.fit(final_data)
#Find the clusters for the observation given in the dataset
score['Cluster'] = Km_model.labels_
score.head()



plt.figure(figsize=(12,6))

##Scatter Plot Frequency Vs Recency
Colors = ["red", "green", "blue"]
score['Color'] = score['Cluster'].map(lambda p: Colors[p])
ax = score.plot(    
    kind="scatter", 
    x="Recency", y="Frequency",
    figsize=(10,8),
    c = score['Color']
)


# # Create a cluster label column in the original DataFrame
# cluster_labels = Km_model.labels_
# temp = score[['Recency','Frequency','Monetary']]
# temp1 = final_data.assign(Cluster = cluster_labels)
# temp2 = temp.assign(Cluster = cluster_labels)

# # Calculate average RFM values and size for each cluster
# summary_k4 = temp2.groupby(['Cluster']).agg({'Recency': 'mean',
#                                                     'Frequency': 'mean',
#                                                     'Monetary': ['mean', 'count'],}).round(0)


# cluster_avg = temp2.groupby(['Cluster']).mean()
# population_avg = temp.head().mean()
# relative_imp = cluster_avg / population_avg - 1
# relative_imp.round(2)



# # Plot heatmap
# plt.title('Relative importance of attributes')
# sns.heatmap(data=relative_imp, annot=True, cmap='RdYlGn')
# plt.show()



# survival analysis for finding retention rate


# cohert analysis

def month(value):
    return dt.datetime(value.year,value.month,1)

# copying the data into another dataframe
df1 = df.copy()
#applying funtion month to get first date on the month
df1['Invoice_Month'] = df1['InvoiceDate'].apply(month)
# grouping all the customers by month
t1 = df1.groupby('Customer ID')['Invoice_Month']
t1.head()
# getting month cohort months 
df1['Cohort_Month'] = t1.transform('min')
df1[['Customer ID','InvoiceDate','Invoice_Month', 'Cohort_Month']].head()

# defining a funtion to seperate invoice year and month
def parse_date(d):
    year = d.dt.year
    month = d.dt.month
    return year, month


year_i, month_i = parse_date(df1['InvoiceDate'])

year_c, month_c = parse_date(df1['Cohort_Month'])


# Calculate differences
diff1 = year_i - year_c
diff2 = month_i - month_c

# Calculating cohort index
df1['Cohort_Index'] = diff1 * 12 + diff2+ 1

df1[['Customer ID','InvoiceDate','Invoice_Month', 'Cohort_Month', 'Cohort_Index']].head()
g = df1.groupby(['Cohort_Month', 'Cohort_Index'])
surv_data = g['Customer ID'].apply(pd.Series.nunique).reset_index()

surv_data.columns = ['Cohort_Month', 'Cohort_Index', 'Number of distinct Customers']

retention_counts = surv_data.pivot(index='Cohort_Month', columns='Cohort_Index', 
                                     values='Number of distinct Customers')


sizes = retention_counts.iloc[:,0]

# calculating retentation rate
rate = retention_counts.divide(sizes, axis=0)

rate.round(3)*100

rate.index = rate.index.strftime('%Y-%m')


plt.figure(figsize=(25, 25))
plt.title('Retention Rate', fontsize = 16)
sns.heatmap(rate, annot=True,cmap='YlGnBu', vmin = 0.0 , vmax = 0.6)
plt.ylabel('Cohort Month')
plt.xlabel('Cohort Index')
plt.yticks( rotation='360')
plt.show()



