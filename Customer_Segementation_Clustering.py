# it a libary used for data wangeling and data clearing 
import pandas as pd
# it is a visulization libary that is used for satistical visualisization data
import seaborn as sns
# it is used to plot graphs and diagrams
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# so that we can remove error due to using outdate varibles or function that are important 
import warnings
warnings.filterwarnings('ignore')

# Bringing in our data set 
df = pd.read_csv("https://raw.githubusercontent.com/Gaelim/Mall-Customer-Segmentation/main/Mall_Customers.csv")

# Looking at the head of the dataframe
df.head()
CustomerID	Gender	Age	Annual Income (k$)	Spending Score (1-100)
0	1	Male	19	15	39
1	2	Male	21	15	81
2	3	Female	20	16	6
3	4	Female	23	16	77
4	5	Female	31	17	40

Univariate Analysis
# Looking at the 5 statistic variable to summarize the dataset for us
df.describe()
CustomerID	Age	Annual Income (k$)	Spending Score (1-100)
count	200.000000	200.000000	200.000000	200.000000
mean	100.500000	38.850000	60.560000	50.200000
std	57.879185	13.969007	26.264721	25.823522
min	1.000000	18.000000	15.000000	1.000000
25%	50.750000	28.750000	41.500000	34.750000
50%	100.500000	36.000000	61.500000	50.000000
75%	150.250000	49.000000	78.000000	73.000000
max	200.000000	70.000000	137.000000	99.000000

# Let create a Distribution so that we can look at our Annuel income.
sns.distplot(df['Annual Income (k$)'])
# Our first Historgram so we can see the shape of our data 

# Let use a for loop to create other visuals so we only have to do this once
# And we we to use a numerical varible to loop through a univariant analysis

#  To get the columns in string format
df.columns

# Observe that CustomerId and Gender was Excluded in the columns
columns = [ 'Age', 'Annual Income (k$)','Spending Score (1-100)']

# where i is a placeholder and plt.figure() will create a visual 
for i in columns:
    plt.figure()
    sns.distplot(df[i])
    
sns.kdeplot(df['Annual Income (k$)'], shade=True, hue=df['Gender']);

# Observe that CustomerId and Gender was Excluded in the columns
columns = [ 'Age', 'Annual Income (k$)','Spending Score (1-100)']

# where i is a placeholder and plt.figure() will create a visual 
for i in columns:
    plt.figure()
    sns.kdeplot(df[i], shade=True, hue=df['Gender'])
    
# Let goona be looking deeper using a boxplot
# Observe that CustomerId and Gender was Excluded in the columns
columns = ['Age','Annual Income (k$)','Spending Score (1-100)']

# where i is a placeholder and plt.figure() will create a visual 
# Each one of this boxplot gives us a different view of the data
for i in columns:
    plt.figure()
    sns.boxplot(data= df, x= 'Gender', y= df[i])

# what we can see is that it looks as if we have more female than male 
df['Gender'].value_counts()

Bivariant Analysis
# we know that in bivarient analysis we are looking at two variable and from the plot below we can see a clustering
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')


# Quick way for me to identify clusters at a glance btw two varible then drop customerId from the plot
df = df.drop('CustomerID', axis=1)
sns.pairplot(df, hue='Gender',height=3.5)
sns.pairplot(df, hue='Age',height=4.5)

# i will want to see the group value based on Gender which shows that women earn less than male 
# but spend higher than male
df.groupby(['Gender'])['Age','Annual Income (k$)','Spending Score (1-100)'].mean()

# To get the co-relation btw this two varible we need to use 
df.corr()

# So now we have a Correlation plot
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Now that we are through with our EDA we need to initial our k-Mean
# algorithm by doing clustering in univaraite, bivariate and multivariate

Clustering - Univaraite, Bivariate and Multivariate
# No of cluster of anything we put in is going to be 8
# Add bracket to turn the data into a dataframe
# we are gonna initial
# we are going to fit 
# we are going to predict 
# We can use n_clusters to decide the amount of cluster we want to use but default is 8
clustering1 = KMeans(n_clusters=3)
clustering1.fit(df[['Annual Income (k$)']])

# this does not mean much unless we are going to compare it with our initial data
clustering1.labels_

df['Income Cluster'] = clustering1.labels_
df.head()

# NOW we are going to do some summaries on univaraite cluster 
# and to count to see how many values are in each income cluster
df['Income Cluster'].value_counts()

# This gives us a centeroid score that the distance btw the centeroid 
clustering1.inertia_

# how to know the number  of cluster to use, am going to put all the score am going to get in this empty  list
# we want to get the inertia score 
intertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    intertia_scores.append(kmeans.inertia_)
    
intertia_scores

# just to get the names of the columns
df.columns

# Let start doing our analysis avaerge
df.groupby('Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()

Bivariate Clustering
Clustering2 = KMeans(n_clusters=5)
Clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
Clustering2.labels_
df['Spending and Income Cluster'] = Clustering2.labels_
df.head()
Gender	Age	Annual Income (k$)	Spending Score (1-100)	Income Cluster	Spending and Income Cluster
0	Male	19	15	39	2	4
1	Male	21	15	81	2	3
2	Female	20	16	6	2	4
3	Female	23	16	77	2	3
4	Female	31	17	40	2	4
intertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    intertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11), intertia_scores2)
# n_clusters=5 form the elbow of the plot
intertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    intertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11), intertia_scores2)
[<matplotlib.lines.Line2D at 0x12da705e0>]

# Adding the clusterbto map by using the x and y cluster
centers = pd.DataFrame(Clustering2.cluster_centers_)
centers.columns = ['x','y']
# form the plot it looks like we are going to have the cluster of five 
# Let do some analysis by visualising a bivariate using a scatter plot
plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Spending and Income Cluster',palette='tab10')
# To save the visual as a png image for presentation 
plt.savefig('Clustering_bivariant.png')
<AxesSubplot:xlabel='Annual Income (k$)', ylabel='Spending Score (1-100)'>

'Spending and Income Cluster'
# we are comparing the cluster with the gender and the dominance level but our target 
# cluster will be cluster 1 with high annual income and spending score
pd.crosstab(df['Spending and Income Cluster'], df['Gender'], normalize='index') * 100
Gender	Female	Male
Spending and Income Cluster		
0	59.259259	40.740741
1	53.846154	46.153846
2	45.714286	54.285714
3	59.090909	40.909091
4	60.869565	39.130435
#  let look at the average cluster
df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()
Age	Annual Income (k$)	Spending Score (1-100)
Spending and Income Cluster			
0	42.716049	55.296296	49.518519
1	32.692308	86.538462	82.128205
2	41.114286	88.200000	17.114286
3	25.272727	25.727273	79.363636
4	45.217391	26.304348	20.913043
# cluster 1 has a high annual income with a lower age and high spending score which will be ideal to run campaign on
# but the lowest age is cluster 3
# cluster 4 has a low annual income and a high spending score
Multivariate Clustering
# we need to import some library and StandardScaler which will allow you to scalar that data
# We need to put the data on the same scalar to get the algorithm to work correctly 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
df.head()
Gender	Age	Annual Income (k$)	Spending Score (1-100)	Income Cluster	Spending and Income Cluster
0	Male	19	15	39	2	4
1	Male	21	15	81	2	3
2	Female	20	16	6	2	4
3	Female	23	16	77	2	3
4	Female	31	17	40	2	4
# for this dataframe we does need the Gander we have to 
# find a way to represent this categorical data to numerical data
# We just need Gender    Age    Annual Income (k$)
# we can use one_hot_Encoding or get dummies
# let 0 be Female and 1 be male
# drop_first=True to drop the second column
dff=pd.get_dummies(df, drop_first=True)
dff.head()
Age	Annual Income (k$)	Spending Score (1-100)	Income Cluster	Spending and Income Cluster	Gender_Male
0	19	15	39	2	4	1
1	21	15	81	2	3	1
2	20	16	6	2	4	0
3	23	16	77	2	3	0
4	31	17	40	2	4	0
dff.columns
Index(['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Income Cluster',
       'Spending and Income Cluster', 'Gender_Male'],
      dtype='object')
dff=dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]
dff.head()
Age	Annual Income (k$)	Spending Score (1-100)	Gender_Male
0	19	15	39	1
1	21	15	81	1
2	20	16	6	0
3	23	16	77	0
4	31	17	40	0
dff = scale.fit_transform(dff)
# To scale the data dff
dff.all()
dff=pd.DataFrame(scale.fit_transform(dff))
dff.head()
0	1	2	3
0	-1.424569	-1.738999	-0.434801	1.128152
1	-1.281035	-1.738999	1.195704	1.128152
2	-1.352802	-1.700830	-1.715913	-0.886405
3	-1.137502	-1.700830	1.040418	-0.886405
4	-0.563369	-1.662660	-0.395980	-0.886405
intertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    intertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11), intertia_scores3)
[<matplotlib.lines.Line2D at 0x12db327d0>]

df
df
Gender	Age	Annual Income (k$)	Spending Score (1-100)	Income Cluster	Spending and Income Cluster
0	Male	19	15	39	2	4
1	Male	21	15	81	2	3
2	Female	20	16	6	2	4
3	Female	23	16	77	2	3
4	Female	31	17	40	2	4
...	...	...	...	...	...	...
195	Female	35	120	79	1	1
196	Female	45	126	28	1	2
197	Male	32	126	74	1	1
198	Male	32	137	18	1	2
199	Male	30	137	83	1	1
200 rows × 6 columns

#  to save file as a csv file
df.to_csv('Clustering.csv')
# Clustering is an unsupervised machine learning process which learns from the data itself

