
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris



iris=load_iris()
X,y= iris.data,iris.target


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=42)

scaler=StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

model=LogisticRegression()
model.fit(X_train,y_train)

predictions= model.predict(X_test)
accuracy=accuracy_score(y_test,predictions)
print("Accuracy:",accuracy)
print(classification_report(y_test,predictions))
