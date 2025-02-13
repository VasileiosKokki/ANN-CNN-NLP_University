import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from generate_dataset import generate_binary_problem
from logistic_regression import LogisticRegressionEP34
from sklearn.linear_model import LogisticRegression

c = np.array([[0, 8], [0, 8]])
# c = np.array([[0, 3], [0, 3]])

# Καλούμε την generate_binary_problem για να παράξουμε N = 1000 σημεία με κέντρα στο (0, 0) και (8, 8),
X, y = generate_binary_problem(N=1000, centers=c)

# Χωρίζουμε το σύνολο δεδομένων σε 70% σύνολο εκπαίδευσης και 30% σύνολο δοκιμής
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70)

# Αρχικοποιούμε ένα αντικείμενο LogisticRegressionEP34
model = LogisticRegressionEP34()

# Καλούμε την fit για τα δεδομένα εκπαίδευσης
model.fit(X_train, y_train, show_line=True, batch_size=None)

# Για σύγκριση (αρκετά πιο γρήγορο)
# model = LogisticRegression()
# model.fit(X_train, y_train)

# Υπολογίζουμε την ευστοχία του μοντέλου μετά την εκπαίδευση.
y_hat_test = model.predict(X_test)
# <0.5 = ανήκει στην κλάση Α, >=0.5 ανήκει στην κλάση Β
y_hat_test = (y_hat_test >= 0.5).astype(int)
# print(y_test, y_hat_test)
accuracy = accuracy_score(y_test, y_hat_test)
print("Model accuracy on test set:", accuracy)

c_m = confusion_matrix(y_test, y_hat_test)
print(c_m)