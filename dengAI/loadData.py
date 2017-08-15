import pandas as pd
import numpy as np

trainData = pd.read_csv("dengue_features_train.csv")
trainLables = pd.read_csv("dengue_labels_train.csv")
data = pd.merge(trainData,trainLables, how = 'outer', on = ["city", "year","weekofyear"])

#replace date with absolute day number
from  datetime import datetime
def dateToDay(string):
    dt = datetime.strptime(string,"%Y-%m-%d")
    first = datetime.strptime("1990-04-30","%Y-%m-%d")
    delta = dt-first
    return delta.days

dataTmp = data["week_start_date"].apply(dateToDay)
data2 = data
data2["week_start_date"] = dataTmp

#replace city names with numbers
data2 = data2.replace({"sj":0,"iq":1})

#add aditional entry to store whether line conatins a corrected value
data2["corrected_data"] = 0

#replace nans with interpolated value where possible
for row, rowData in data2.iterrows():
    for col in range(data2.shape[1]):
        value = rowData.iloc[col]
        if not pd.notnull(value) :
            #try to correct value by entries above and below
            valueAbove = data2.iloc[row-1, col]
            valueBelow = data2.iloc[row+1, col]
            if pd.notnull(valueAbove) and pd.notnull(valueBelow):
                data2.iloc[row,col] = (valueAbove+valueBelow)/2
                data2.iloc[row,25] = 1      #marke row as modified


#remove rows that still hava a nan entry
data2 = data2[~data2.isnull().any(axis=1)]

#separate into two cities
dataCity1 = data2[:866]
dataCity2 = data2[866:]
