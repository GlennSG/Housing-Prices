
# coding: utf-8

# We are going to create a model to predict house's value in Melbourne. In this case, value is the price of the property, and we also would like to know what factors create value in a house.
# 
# Data Source: https://www.kaggle.com/anthonypino/melbourne-housing-market#Melbourne_housing_FULL.csv

# ## Data Preprocessing

# In[364]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pylab
get_ipython().run_line_magic('matplotlib', 'inline')


# In[404]:


def normalize(data):
    x = (data-data.min())/(data.max()-data.min())
    return x

def transform(data):
    x,lam = stats.boxcox(data)
    return x

def visual(df,plot_type):
    plt.rcParams["figure.figsize"] = (23,13)
    fig,ax = plt.subplots(nrows=6,ncols=4)
    i = 0
    j = 0
    
    if plot_type == "hist":
        for col in df.columns:
            if col == "Price":
                ax[i,j].hist(df[col],alpha=0.5,label=col,color="orange")
                ax[i,j].set_title(col + " Distribution")
            else:
                ax[i,j].hist(df[col],alpha=0.5,label=col)
                ax[i,j].set_title(col + " Distribution")
            if j == 3:
                i += 1
                j = 0
            else:
                j += 1
    if plot_type == "box":
        for col in df.columns:
            ax[i,j].boxplot(df[col])
            ax[i,j].set_title(col)
            if j == 3:
                i += 1
                j = 0
            else:
                j += 1
    if plot_type == "qq":
        norm = np.random.normal(0, 1, df.shape[0])
        norm.sort()
        for col in df.columns:
            sorted_col = list(df[col])
            sorted_col.sort()
            if col == "Price":
                ax[i,j].plot(norm,sorted_col,color="orange")
                ax[i,j].set_title(col + " Distribution")
            else:
                ax[i,j].plot(norm,sorted_col)
                ax[i,j].set_title(col + " Distribution")
            
            z = np.polyfit(norm,df[col], 1)
            p = np.poly1d(z)
            
            # trend line 
            # ax[i,j].plot(norm,p(norm),"k--",linewidth=2)
            
            if j == 3:
                i += 1
                j = 0
            else:
                j += 1

    fig.tight_layout()
    plt.show()


# In[366]:


data = pd.read_csv("Melbourne_housing_FULL.csv")
data.head(3)


# In[367]:


data.shape


# In[368]:


data.columns


# In[369]:


data["Date"] = pd.to_datetime(data["Date"])


# In[370]:


data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day


# In[371]:


data.dtypes


# In[372]:


cate_col = data.loc[:,data.dtypes == np.object]


# In[373]:


for col in cate_col:
    data[col + "_updated"] = pd.factorize(data[col])[0]


# In[374]:


data.dtypes


# In[375]:


house_data = data.loc[:,((data.dtypes != np.object) & (data.columns != "Date"))]


# In[376]:


house_data.dtypes


# In[377]:


# select K Best, first need to check & replace NaN values
house_data.isnull().sum()


# In[378]:


missing_val_col = house_data.columns[house_data.isna().any()].tolist()

for col in missing_val_col:
    house_data[col] = house_data[col].fillna(house_data[col].mean())


# In[379]:


house_data.isnull().sum()


# In[380]:


house_data.shape


# ## Visualizing Data Distribution and Outliers

# In[381]:


# apply normalization to data to get more accurate visuals
house_data = house_data.apply(normalize)


# In[382]:


visual(house_data,"hist")


# Most of the distributions (besides latitude and longitude) appear to be slightly skewed. May need to normalize and transform the data to better fit our models.

# In[383]:


## Check for Outliers
visual(house_data,"box")


# Appear to be some outliers in our dataset. We will most likely need to remove them to get a more accurate model.

# In[384]:


house_data.shape


# In[385]:


z = np.abs(stats.zscore(house_data))


# In[386]:


house_data = house_data[(z<3).all(axis=1)]


# In[387]:


house_data.shape


# In[388]:


visual(house_data,"box")


# Appears that there are still some outliers present in the data, but they are within two z-scores so they are not too significant and our model should still perform adequately.

# In[389]:


# replace all 0 with small digit (0.00001)
house_data = house_data.replace(0,0.00001)


# In[390]:


# transform non_categorical, non normally distributed data
non_norm_cols = house_data.loc[:,((house_data.columns != "Longtitude")& (house_data.columns != "Lattitude"))]
for col in non_norm_cols:
    house_data[col] = transform(house_data[col])


# In[391]:


visual(house_data,"hist")


# In[405]:


visual(house_data,"qq")


# Not exactly perfect normal distributions for some of the independent variables, but much better improvement than previous. The plots with a "staircase" distribution indicates columns containing discrete values. 

# ## Taking samples
# 
# Our data set is kinda big (about 30,000 data values) which could pose an efficiency issue later on when we decide to use models like Support Vector Machines, Random Forest or XGBoost. Lets see if we can create a viable sample that best reflects our population (dataset). 

# In[407]:


import math

# supported confidence levels: 50%, 68%, 90%, 95%, and 99%
confidence_lvl_constant = [50,.67],[68,.99],[90,1.64],[95,1.96],[99,2.57]

# calculate the sample size
def sample_size(population_size,confidence_level,confidence_interval):
    Z = 0.0
    p = 0.5
    e = confidence_interval/100.0
    N = population_size
    n_0 = 0.0
    n = 0.0
    
    # Loop through supported confidence levels and find the num std
    # Deviations for that confidence interval
    for i in confidence_lvl_constant:
        if i[0] == confidence_level:
            Z = i[1]
    
    if Z == 0.0:
        return -1
    
    # calc sample size
    n_0 = ((Z**2)*p*(1-p))/(e**2)
    
    # adjust sample size for finite population
    n = n_0 / (1 + ((n_0 - 1)/float(N)))
    
    return int(math.ceil(n)) # the sample size

def main():
    sample_sz = 0
    population_sz = house_data.shape[0]
    confidence_level = 95.0
    confidence_interval = 2.0
    
    sample_sz = sample_size(population_sz,confidence_level,confidence_interval)
    
    print("Sample size: ",sample_sz)

if __name__ == "__main__":
    main()


# We used an equation to find the necessary sample size for our dataset:
# 
# ##### Necessary Sample Size = (Z-score)^2 x StdDev x (1 - StdDev) / (margin of error)
# 
# So we need at least 2,219 data values to get a representative sample of our entire dataset (population). 

# In[408]:


house_sample = house_data.sample(n = 2219,random_state = 42)


# In[410]:


house_sample.shape


# In[412]:


house_sample.head(3)


# Next possible steps: compare population and sample to confirm sample is representative sample of population.
