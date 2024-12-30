#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# In[2]:


df=pd.read_csv("C:/Users/harma/Desktop/Financial Sample.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


# Eliminating the white spaces 
df.columns = df.columns.str.strip()

 


# In[6]:


df.columns


# In[7]:


df.dtypes


# In[8]:


# Changing the Datatypes 
num_col= ['Manufacturing Price','Sale Price','Gross Sales','Discounts','Sales','COGS','Profit']
for col in num_col:
    df[col] = df[col].astype(str)  # Convert to string if not already
    df[col] = df[col].replace({'\$': '', '₹': '', ',': ''}, regex=True)  # Remove currency symbols and commas
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, replacing errors with NaN



# In[9]:


# Changing data types to datatime 
df['Date'] = pd.to_datetime(df['Date'],errors='coerce')


# In[10]:


# Changing  Data type to category 
Cat_col = ['Segment','Country','Product','Discount Band']
for col in Cat_col:
    df[col]=df[col].astype('category')


# In[11]:


# reviewing the datatype 
df.dtypes


# In[12]:


df.head()


# In[13]:


# Checking for null values 
df.isnull().sum()


# In[14]:


#filling the null values with 0
df['Profit'] = df['Profit'].fillna(0)


# In[15]:


df.isnull().sum()


# In[16]:


# Plotting histogram for Sales
sns.histplot(df['Sales'], bins=30, kde=True, color='blue')
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()


# In[17]:


sns.boxplot(x=df['Profit'])


# In[18]:


# Kernel Density Estimate (kDE) to understand the Distribution of Discounts 
sns.kdeplot(df['Discounts'])


# In[19]:


# Correlation Matrix to Understand the relationships between Variable 
correlation_matrix = df[['Units Sold', 'Sale Price', 'Gross Sales', 'Discounts', 'Sales', 'COGS', 'Profit']].corr()


# In[20]:


sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')


# In[21]:


# Checking For Outliers 


# In[22]:


z_scores = np.abs((df['Profit'] - df['Profit'].mean()) / df['Profit'].std())
outliers = df[z_scores > 3]


# In[23]:


Q1 = df['Profit'].quantile(0.25)
Q3 = df['Profit'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Profit'] < (Q1 - 1.5 * IQR)) | (df['Profit'] > (Q3 + 1.5 * IQR))]


# In[24]:


# Total Sales By Segments 
sns.barplot(x='Segment', y='Sales', data=df)


# In[25]:


# Total Profit By Country 
sns.barplot(x='Country', y='Profit', data=df)


# In[26]:


# Total Sales By Month 
df.groupby('Month Name')['Sales'].sum().plot(kind='line')


# In[27]:


# Profit Trend by Year 
sns.lineplot(x='Year', y='Profit', data=df)


# In[28]:


# Relationship between Discount and Sales 
sns.scatterplot(x='Discounts', y='Sales', data=df)


# In[29]:


# Relationship Betwween Sales and Profit 
sns.regplot(x='Sales', y='Profit', data=df)


# In[30]:


# Pairwise Relationships Between Key Metrics 
sns.pairplot(df[['Units Sold', 'Sale Price', 'Gross Sales', 'Discounts', 'Sales', 'COGS', 'Profit']])


# In[31]:


# Performig Predictive modelling- Simple Linear regression  
x=df[['COGS']]
y=df['Profit']


# In[32]:


from sklearn.model_selection import train_test_split
x_train ,x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[33]:


from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(x_train , y_train)


# In[34]:


y_pred = model.predict(x_test)


# In[35]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE:', mse)
print('R² Score:', r2)


# In[36]:


import matplotlib.pyplot as plt

# Scatter plot of actual data
plt.scatter(x_test, y_test, color='blue', label='Actual')

# Regression line
plt.plot(x_test, model.predict(x_test), color='red', label='Regression Line')

# Add labels and legend
plt.xlabel('COGS')
plt.ylabel('Profit')
plt.title('Regression Line: COGS vs Profit')
plt.legend()
plt.show()


# In[41]:


# Predict on test data
y_pred = model.predict(x_test)

# Scatter plot of actual vs predicted
plt.scatter(y_test, y_pred, color='purple', alpha=0.6)

# Add a 45-degree reference line for perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)

# Add labels and title
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('Actual vs Predicted Values')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




