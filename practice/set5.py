from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('data.csv')

X = df[['age','experience']]
y = (df['salary'] > df['salary'].median()).astype(int)
df.dropna()

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3)

model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test,y_pred))


y_prob = model.predict_proba(X_test)[:,1]

fpr,tpr,_ = roc_curve(y_test, y_prob)

plt.plot(fpr,tpr)
plt.show()