import numpy as np
import time
from cpuinfo import get_cpu_info
import psutil
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from generate_dataset import generate_binary_problem
from logistic_regression import LogisticRegressionEP34
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import pandas as pd

dataset = load_breast_cancer()

accuracies = []

start_time = time.time()

n = 20
for i in range(n):

    print(f"Step {i + 1} of {n}")

    # Χωρίζουμε το σύνολο δεδομένων σε 70% δεδομένα εκπαίδευσης και 30% δοκιμής.
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.70)

    # Κανονικοποιούμε τα δεδομένα στο [0, 1]
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X_train)
    Xn_train = scaler.transform(X_train)
    Xn_test = scaler.transform(X_test)

    # Εκπαιδεύoυμε ένα μοντέλο λογιστικής παλινδρόμησης (το δικό μας)
    model = LogisticRegressionEP34()  # lr=10**(-2) by default
    model.fit(Xn_train, y_train, show_line=False, batch_size=64)
    # Για σύγκριση, αρκετά πιο αργό (30 secs)
    # model.fit(Xn_train, y_train, show_line=False, batch_size=None)

    # Αξιολογούμε την ευστοχία στο σύνολο δοκιμής
    y_hat_test = model.predict(Xn_test)
    y_hat_test = (y_hat_test >= 0.5).astype(int)
    # print(y_test, y_hat_test)
    accuracy = accuracy_score(y_test, y_hat_test)
    accuracies.append(accuracy)
    print("Model accuracy on test set:", accuracy)

    c_m = confusion_matrix(y_test, y_hat_test)
    print(f"{c_m}\n")

end_time = time.time()


mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("Mean Accuracy:", mean_accuracy)
print("Standard Deviation of Accuracy:", std_accuracy)

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

cpu_info = get_cpu_info()
processor_name = cpu_info['brand_raw']  # User-friendly name
ram = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB

print(f"Processor: {processor_name}")
print(f"Total RAM: {ram:.2f} GB")