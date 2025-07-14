#About this file
#Attribute Information: (classes: edible=e, poisonous=p)
#cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
#cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
#cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
#bruises: bruises=t,no=f
#odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
#gill-attachment: attached=a,descending=d,free=f,notched=n
#gill-spacing: close=c,crowded=w,distant=d
#gill-size: broad=b,narrow=n
#gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
#stalk-shape: enlarging=e,tapering=t
#stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
#stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
#stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
#stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
#stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
#veil-type: partial=p,universal=u
#veil-color: brown=n,orange=o,white=w,yellow=y
#ring-number: none=n,one=o,two=t
#ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
#spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
#population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
#habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

import pandas as pd

df = pd.read_csv("C:/Users/ayushi/Downloads/mushrooms (1).csv")

df.tail()

df.head()

df.shape

df.info()

df.describe()

df.columns

y=df['class']
x=df[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']]

x.shape,y.shape

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
r=l.fit_transform(y)

y

r

l

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=5057)

x_train.shape,x_test.shape,y_test.shape,y_train.shape

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression

for i in x:
    x[i]=l.fit_transform(x[i])

li=LinearRegression()
li.fit(x_train,y_train)

lr=LogisticRegression()
lr.fit(x_train,y_train)

knn = KNeighborsClassifier()
knn.fit(x_train,y_train)

svc=SVC()
svc.fit(x_train,y_train)

dt= DecisionTreeClassifier()
dt.fit(x_train,y_train)

gb=GradientBoostingClassifier()
gb.fit(x_train,y_train)

rf=RandomForestClassifier()
rf.fit(x_train,y_train)

y_pred1=lr.predict(x_test)
y_pred2=knn.predict(x_test)
y_pred3=svc.predict(x_test)
y_pred4=dt.predict(x_test)
y_pred5=rf.predict(x_test)
y_pred6=gb.predict(x_test)

from sklearn.metrics import accuracy_score

print('acc lr',accuracy_score(y_test,y_pred1))
print('acc knn',accuracy_score(y_test,y_pred2))
print('acc dt',accuracy_score(y_test,y_pred3))
print('acc rf',accuracy_score(y_test,y_pred4))
print('acc gb',accuracy_score(y_test,y_pred5))
print('acc svc',accuracy_score(y_test,y_pred6))

finaldata=pd.DataFrame({'models':['lr','knn','gb','rb','dt','svc'],
             'acc':[accuracy_score(y_test,y_pred1)*100,
                    accuracy_score(y_test,y_pred2)*100,
                    accuracy_score(y_test,y_pred3)*100,
                    accuracy_score(y_test,y_pred4)*100,
                    accuracy_score(y_test,y_pred5)*100,
                    accuracy_score(y_test,y_pred6)*100]})

finaldata

import seaborn as sns

sns.barplot(finaldata['models'],finaldata['acc'])





