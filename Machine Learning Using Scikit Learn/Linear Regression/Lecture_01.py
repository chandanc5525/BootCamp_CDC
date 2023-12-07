#!/usr/bin/env python
# coding: utf-8

# # Prediction Model for Ecommerce Dataset
# 
# ### Prepaed By: CHANDAN D.CHAUDHARI
# © CopyRight

# ## Business Problem: 
# ## Develop a predictive model to estimate the 'Yearly Amount Spent' by customers based on their demographic and engagement features. The objective is to identify key factors that influence customer spending and to create a tool that can assist in personalized marketing strategies.

# In[1]:


import numpy as np
import pandas as pd

# Import Data Visualization Library
import matplotlib.pyplot as plt
import seaborn as sns

# Import Filter Warning Library
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# import Dataset using Pandas library function

data = pd.read_csv('G:\BootCamp_CDC\Machine Learning Using Scikit Learn\Linear Regression\Ecommerce Customers')
data


# In[3]:


data.columns


# In[4]:


# Pairplot to visualize relationships between numerical columns and the target
sns.pairplot(data, x_vars=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'], y_vars=['Yearly Amount Spent'])
plt.show()

# Correlation matrix to quantify relationships
correlation_matrix = data.corr()
print(correlation_matrix['Yearly Amount Spent'])


# ## Analyze More Insights For Avatar Column in Ecommerce Dataset

# In[5]:


data['Avatar'].unique()


# In[6]:


plt.figure(figsize=(30,9))  # Adjust the figure size as needed
sns.countplot(x='Avatar', data=data, order=data['Avatar'].value_counts().index)
plt.xticks(rotation=90, ha='right')  # Rotate x-labels for better readability
plt.tight_layout()  # Adjust layout for better spacing
plt.show()


# In[7]:


plt.figure(figsize=(14, 8))  # Adjust the figure size as needed
sns.boxplot(x='Avatar', y='Yearly Amount Spent', data=data, order=data['Avatar'].value_counts().index)
plt.xticks(rotation=90, ha='right')  # Rotate x-labels for better readability
plt.title('Avatar vs Yearly Amount Spent')
plt.tight_layout()
plt.show()


# In[8]:


data['Avatar'].value_counts()


# In[9]:


data[data['Avatar'] == 'SlateBlue']


# In[10]:


data[data['Avatar'] == 'MintCream']


# In[11]:


data[data['Avatar'] == 'AliceBlue']


# In[12]:


data.columns


# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = data[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = data['Yearly Amount Spent']


# In[14]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state= 42)
X_train


# In[15]:


LR = LinearRegression().fit(X_train,y_train)
Prediction = LR.predict(X_test)


# In[16]:


R_score_test = LR.score(X_test,y_test) 
R_score_test
R_score_train = LR.score(X_train,y_train) 
R_score_train


# In[17]:


print(f"The Train Model R Square Value is {R_score_train*100} %")
print(f"The Test Model R Square Value is {R_score_test*100} %")


# In[18]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Prediction))
print('Root Mean Squar Error:', np.sqrt(metrics.mean_squared_error(y_test, Prediction)))


# In[19]:


sns.distplot((y_test-Prediction))
plt.tight_layout()
plt.grid()
plt.show()


# ## Verification using Ridge and Lasso Regression Techniques

# In[20]:


from sklearn.linear_model import Ridge,Lasso

Ridge = Ridge().fit(X_train,y_train)
Lasso = Lasso().fit(X_train,y_train)

Ridge_Acc = Ridge.predict(X_test)
Ridge_score = Ridge.score(X_test,y_test)
Lasso_Acc = Lasso.predict(X_test)
Lasso_score = Lasso.score(X_test,y_test)


# In[21]:


# For Ridge Linear Regression Model
print(f"The Test Model R Square Value is {Ridge_score*100} %")


# In[22]:


# For Lasso Regression Model
print(f"The Test Model R Square Value is {Lasso_score*100} %")


# ## Final Results 

# In[23]:


Models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
R_Scores = ['98.08%', '98.08%', '98.03%']

Results = pd.DataFrame({'Model': Models, 'R_Score': R_Scores})

# Displaying the DataFrame
Results

