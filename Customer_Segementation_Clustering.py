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
