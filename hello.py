import pandas as pd
import pickle

data = pd.read_csv("upload.csv")
from sklearn.model_selection import train_test_split

training_set, test_set = train_test_split(data,test_size = 0.2, random_state = 1)
X_train = training_set["q_a"]
Y_train = training_set["label"]
X_test = test_set["q_a"]
Y_test = test_set["label"]

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(min_df=1,stop_words='english')
x_train_counts = cv.fit_transform(X_train)

cv.vocabulary_.get(u'algorithm')
# print(X_train,Y_train)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(x_train_counts,Y_train)
# print(classifier.score(x_train_counts,Y_train))
pickle.dump(classifier,open('model.pkl','wb'))


x_new_counts = cv.transform(X_test)
pred = classifier.predict(x_new_counts)
#print(pred)
#print(pred)

# x_test_counts = cv.fit_transform(X_test)

# Y_pred = classifier.predict(x_test_counts)
# test_set["Predictions"] = Y_pred

# print(test_set)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,pred)
print("Confusion matrix is \n", cm)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy*100)