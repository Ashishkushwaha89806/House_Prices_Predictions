import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

#1 load  the data...

housing=pd.read_csv('housing.csv')

#2 create a stratified test set based on  income category
housing ["income_cat"]=pd.cut(housing["median_income"],
                              bins=[0.0,1.5,3.0,4.5,6,np.inf],
                              labels=[1,2,3,4,5]
                              )
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set=housing.loc[train_index].drop("income_cat",axis=1)
    strat_test_set=housing.loc[test_index].drop("income_cat",axis=1)

#work on a copy of training data
housing=strat_train_set.copy()

#3 Separate predictors and labels
housing_labels=housing["median_house_value"].copy()
housing=housing.drop("median_house_value",axis=1)

#4 Separate numerical and categorical columns..
num_attribs=housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribus=["ocean_proximity"]

# 5 pipelines 
#Numerical pipeline 
num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
])

#Categorical pipeline

cat_pipeline=Pipeline([
    #"ordinal",OrdinalEncode() #use this if you perfer ordinal encoding

    ("onehot",
     OneHotEncoder(handle_unknown="ignore"))
])

# full pipeline 
full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",cat_pipeline,cat_attribus),
])

housing_prepared=full_pipeline.fit_transform(housing)
print(housing_prepared)

#Linear Regression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

#decision tree
tree_reg=DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared,housing_labels)

#random forest 
forest_reg=RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared,housing_labels)

#predict using training data 
lin_preds=lin_reg.predict(housing_prepared)
tree_preds=tree_reg.predict(housing_prepared)
forest_preds=forest_reg.predict(housing_prepared)

# Calculate RMSE

# lin_rmse=root_mean_squared_error(housing_labels,lin_preds)
# tree_rmse=root_mean_squared_error(housing_labels,tree_preds)
# forest_rmse=root_mean_squared_error(housing_labels,forest_preds)
# POOR GENERALIZATION....
 #using  CROSS -VALIDATION A BETTER EVALUATION STRATEGY  

lin_rmse=-cross_val_score(lin_reg,housing_prepared,housing_labels,
                          scoring="neg_root_mean_squared_error",cv=10)
tree_rmse=-cross_val_score(tree_reg,housing_prepared,housing_labels,
                          scoring="neg_root_mean_squared_error",cv=10)
forest_rmse=-cross_val_score(forest_reg,housing_prepared,housing_labels,
                          scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(lin_rmse).describe())
print(pd.Series(tree_rmse).describe())
print(pd.Series(forest_rmse).describe())
