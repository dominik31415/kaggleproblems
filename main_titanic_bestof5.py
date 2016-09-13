#Game plan:
#extend data set by a few columns (title, unknown age, split Pclasses, family size) and default up nan-s
#implmenet 5 different learning algorithms: Kneighbours, tree, gaussian+linear SVM, forest
#optimize  each by cross validation
#use majority vote for final prediction

import pandas as pd
import numpy as np
train = pd.read_csv(open('train.csv','rb'))
test = pd.read_csv(open('test.csv','rb'))
test0 = test

#independent pre-processing
# two people in the training set have no embarked entry, i'll just remove them for now
train = train.ix[train['Embarked'].notnull()]
# for the one data point with missing fare, I simply replace its value with the median fare for its Class
test.ix[test['Fare'].isnull()]
#the guy is Pclass 3
tmp = np.median(train.ix[train['Pclass']==3]['Fare'])
#print(tmp)
test.loc[152,'Fare']=tmp


#data preprocessing for both sets
#for normalization
max_fare = max(train['Fare'].values)
max_age = max(train['Age'].values)

def prepare_data(data):
	
	#normalize fares
	data['Fare'] = data['Fare']/max_fare


	#drop the columns I am not going to use
	data = data.drop('PassengerId',axis=1)
	data = data.drop('Cabin',axis=1)
	data = data.drop('Ticket',axis=1)

	#I need to convert strings to numerical values 
	#convert entries to numerical values in the Sex and Embarked feature
	dic = {'male':0,'female':1,'S':0,'C':0.5,'Q':1.0}
	data = data.replace(dic)
	
	#missing age values: default nan to 0 but add a second column, specifying wether age was given or not
	#there might be some information hidden in  the fact that the age is not known
	tmp = data['Age'].notnull().values
	data['AgeGiven'] = tmp
	data.loc[data['Age'].isnull(),'Age']=0
	#also normalize age
	data['Age'] = data['Age']/max_age


	#Pclass is a classifier, better to split into 3 different boolean variables for the neural network
	tmp1 = (data['Pclass']==1)
	tmp2 = (data['Pclass']==2)
	tmp3 = (data['Pclass']==3)
	data['Class1'] = tmp1
	data['Class2'] = tmp2
	data['Class3'] = tmp3
	data = data.drop('Pclass',axis=1)

	#try adding a few other variables
	data['FamilySize'] = data['SibSp'].values + data['Parch'].values

	#Name will be deleted, but maybe the title means something
	import re
	# A function to get the title from a name.
	def get_title(name):
		# Titles always consist of capital and lowercase letters, and end with a period.
		title_search = re.search(' ([A-Za-z]+)\.', name)
		# If the title exists, extract and return it.
		if title_search:
			return title_search.group(1)
		return ""

	# Get all the titles and print how often each one occurs.
	titles = data["Name"].apply(get_title)

	# Map each title to an integer.  
	#Children get 0, 'normal' adults 1, and titles that require some seniority get 2
	#title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona":9}
	title_mapping = {"Mr": 1, "Miss": 0, "Mrs": 1, "Master": 0, "Dr": 2, "Rev": 2, "Major": 2, "Col": 2, "Mlle": 1, "Mme": 1, "Don": 1, "Lady": 2, "Countess": 2, "Jonkheer": 2, "Sir": 1, "Capt": 2, "Ms": 1, "Dona":1}
	for k,v in title_mapping.items():
		titles[titles == k] = v/2.0 #normalize it right away
	data["Title"] = titles
	data = data.drop('Name',axis=1)
	
	return data

################


#extract values
d2 = prepare_data(train)
y2 = d2['Survived'].values
d2 = d2.drop('Survived',axis=1)
d2 = d2.values.astype(float)

from sklearn.cross_validation import train_test_split
d0,d1,y0,y1 = train_test_split(d2,y2,test_size=0.3,random_state=1)


test = prepare_data(test)
test= test.values.astype(float)

### idividual classifiers, each is optimized in one dimension only
#classifiers with 2 parameters are optimized in a combination of by hand and automatically
from sklearn.neighbors import KNeighborsClassifier
nax = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]
E = [0]*len(nax)
for ind,N in enumerate(nax):
	kn = KNeighborsClassifier(weights='distance',n_neighbors=N)
	kn.fit(d0,y0)
	pred = kn.predict(d1)
	E[ind] = sum(pred==y1)

print('neighbours, best result :',max(E))
print('best parameter: ', nax[np.argmax(E)])
kn_opt = nax[np.argmax(E)]

##############
from sklearn.tree import DecisionTreeClassifier
E = [0]*40
for N in range(40):
	tr = DecisionTreeClassifier(criterion='entropy',max_depth=N+1,random_state=0)
	tr.fit(d0,y0)
	pred = tr.predict(d1)
	E[N] = sum(pred==y1)

print('tree, best result :',max(E))
print('best parameter: ', np.argmax(E))
tr_opt=np.argmax(E)

##############
from sklearn.ensemble import RandomForestClassifier
E = [0]*40
for N in range(40):
	fo = RandomForestClassifier(criterion='entropy',n_estimators=N+1,n_jobs=3,random_state=1)
	fo.fit(d0,y0)
	pred = fo.predict(d1)
	E[N] = sum(pred==y1)

print('forest, best result :',max(E))
print('best parameter: ', np.argmax(E))
fo_opt=np.argmax(E)

##############
from sklearn.svm import SVC
Cax = np.exp(np.arange(-6,-2,0.05))
E = [0]*len(Cax)
for ind,C0 in enumerate(Cax):
	sv =  SVC(kernel = 'linear', C=C0,random_state=0)
	sv.fit(d0,y0)
	pred = sv.predict(d1)
	E[ind] = sum(pred==y1)
	#print(ind)


import matplotlib.pyplot as plt
plt.figure()
plt.plot(Cax,E)
plt.xlabel('C penalty on misclassification')
plt.ylabel('accuracy')
plt.xscale('log')
plt.title('SVM performance on calibration set')
plt.show(block=False)
print('SVM linear, best result :',max(E))
print('best parameter: ', Cax[np.argmax(E)])
svl_opt = Cax[np.argmax(E)]


#try imporving it with gaussian kernel
from sklearn.svm import SVC
Cax = 0.01 # the correct way would be to do a 2D optimization, i just tried a few values by hand
gax = np.exp(np.arange(-6,13,0.2))
E = [0]*len(gax)
for indg,g0 in enumerate(gax):
	sv =  SVC(kernel = 'rbf', C=C0,gamma=g0,random_state=0)
	sv.fit(d0,y0)
	pred = sv.predict(d1)
	E[indg] = sum(pred==y1)
		#print(indg)

plt.figure()
plt.plot(gax,E)
plt.xlabel('gamma')
plt.ylabel('accuracy')
plt.xscale('log')
plt.title('SVM performance on calibration set')
plt.show(block=False)
svg_opt = gax[np.argmax(E)]
print('SVM gaussian, best result :',max(E))
print('best parameter: ', gax[np.argmax(E)])


# now just implement best out of five
class mixedClassifier():
	def __init__(self):
		self.kn = KNeighborsClassifier(weights='distance',n_neighbors=kn_opt)
		self.tr = DecisionTreeClassifier(criterion='entropy',max_depth=tr_opt+1,random_state=0)
		self.svl =  SVC(kernel = 'rbf', C=svl_opt,random_state=0)
		self.svg =  SVC(kernel = 'rbf', C=1,gamma=svg_opt,random_state=0)
		self.fo = RandomForestClassifier(criterion='entropy',n_estimators=fo_opt+1,n_jobs=3,random_state=1)		
		self.obj =[self.kn,self.tr,self.svg,self.svl,self.fo]

		
	def fit(self,x,y):
		for cc in self.obj:
			cc.fit(x,y)
		
	def predict(self,x):
		self.p = np.zeros((len(x),))
		for ind,cc in enumerate(self.obj):
			self.p = self.p + cc.predict(x)
		return (self.p>2.5) # gve all of them the same weight, since they perform about equally well


mc = mixedClassifier()
mc.fit(d0,y0)
pred1 = mc.predict(d1)	
print('mixed classifier result: ',sum(pred1==y1))	
		
mc = mixedClassifier()
mc.fit(d2,y2)
pred = mc.predict(test)

output = pd.DataFrame(columns = ['PassengerId','Survived'])
output['PassengerId'] = test0['PassengerId'].values
output['Survived'] = pred.astype(int)
output.to_csv('mixedClassifier.txt',index=False)


