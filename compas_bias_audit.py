
# coding: utf-8

# # COMPAS Dataset Bias Audit  
# This notebook presents a bias audit on the COMPAS dataset using Python and fairness metrics.
# 
# ---
# 
# **Objective:**  
# Investigate potential algorithmic bias with respect to race and gender in criminal risk assessments.
# 
# **Tools Used:**  
# - Pandas
# - AIF360 (AI Fairness 360)
# - Matplotlib & Seaborn
# 
# ---
# 
# 

# In[17]:


import matplotlib.pyplot as plt


# In[8]:


from aif360.metrics import BinaryLabelDatasetMetric


# In[2]:


from aif360.datasets import CompasDataset
import pandas as pd


# In[1]:


# If not already installed, uncomment the line below to install AI Fairness 360
# !pip install aif360

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


# In[5]:


# Load the COMPAS dataset using AIF360's built-in loader
dataset_orig = CompasDataset()

# Display first few rows of the dataset
df = dataset_orig.convert_to_dataframe()[0]
df.head()


# In[6]:


print("Features:", dataset_orig.feature_names)
print("Protected Attribute:", dataset_orig.protected_attribute_names)
print("Privileged Class:", dataset_orig.privileged_protected_attributes)
print("Unprivileged Class:", dataset_orig.unprivileged_protected_attributes)


# In[9]:


# Define privileged and unprivileged groups (race = Caucasian vs. African-American)
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

# Compute dataset-level metrics
metric = BinaryLabelDatasetMetric(dataset_orig, 
                                   unprivileged_groups=unprivileged_groups,
                                   privileged_groups=privileged_groups)

print("Disparate Impact:", metric.disparate_impact())
print("Statistical Parity Difference:", metric.statistical_parity_difference())


# In[15]:


# Check actual column names from the COMPAS dataset
df = dataset_orig.convert_to_dataframe()[0]
print(df.columns.tolist())



# In[29]:


# Convert to DataFrame for plotting
df = dataset_orig.convert_to_dataframe()[0]

# Filter by race and recidivism predictions
african_american = df[df['race'] == 0]
caucasian = df[df['race'] == 1]

# Count false positives
af_fp_rate = len(african_american[(african_american['two_year_recid'] == 0) & 
                                  (african_american['two_year_recid'] > 1)]) / len(african_american)

cau_fp_rate = len(caucasian[(caucasian['two_year_recid'] == 0) & 
                            (caucasian['two_year_recid'] > 4 )]) / len(caucasian)

# Plotting
labels = ['African-American', 'Caucasian']
values = [af_fp_rate, cau_fp_rate]

plt.bar(labels, values, color=['red', 'green'])
plt.title("False Positive Rate by Race")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.show()


# In[22]:


from aif360.algorithms.preprocessing import Reweighing


# In[23]:


# Apply reweighing algorithm
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf = RW.fit_transform(dataset_orig)

# Metrics after mitigation
metric_transf = BinaryLabelDatasetMetric(dataset_transf, 
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)

print("Post-Mitigation Disparate Impact:", metric_transf.disparate_impact())
print("Post-Mitigation Statistical Parity Difference:", metric_transf.statistical_parity_difference())


# In[25]:


from sklearn.preprocessing import StandardScaler


# In[2]:


import pandas as pd

# Load the dataset from URL
url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
df = pd.read_csv(url)

# Preview the dataset
df.head()


# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Drop the target from the features and keep only numeric columns
X = df.drop(columns=['two_year_recid'], errors='ignore')
X = X.select_dtypes(include='number')

# Define target
y = df['two_year_recid']

# Create pipeline: impute missing values, then scale
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),   # or 'median' if more robust
    StandardScaler()
)

# Transform features
X_prepared = pipeline.fit_transform(X)

# Train logistic regression
model = LogisticRegression()
model.fit(X_prepared, y)

# Predict
y_pred = model.predict(X_prepared)



