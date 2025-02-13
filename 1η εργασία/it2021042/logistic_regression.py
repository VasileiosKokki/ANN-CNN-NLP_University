import numpy as np
from matplotlib import pyplot as plt
from numpy import random


class LogisticRegressionEP34():
    def __init__(self, lr=10**(-2)):
        self.w = None           # Διάνυσμα numpy το οποίο αντιστοιχεί στα βάρη, w, του μοντέλου
        self.b = None           # Αριθμός που αντιστοιχεί στον όρο μεροληψίας, b, του μοντέλου
        self.lr = lr            # Ρυθμός εκμάθησης για την εκπαίδευση του μοντέλου
        self.f = None           # η πρόβλεψη p(1|x)
        self.l_grad_w = None    # η παράγωγος της απώλειας ως προς το w
        self.l_grad_b = None    # η παράγωγος της απώλειας ως προς το b
        self.N = None           # το πλήθος των δεδομένων N με τα οποία εκπαιδεύτηκε το μοντέλο
        self.p = None           # η διάσταση p του χώρου των χαρακτηριστικών


    def init_parameters(self):
        # αρχικοποίηση των παραμέτρων w και b με δειγματοληψία από Γκαουσιανή κατανομή με μέση τιμή μ = 0 και τυπική απόκλιση σ = 0.1
        self.w = random.randn(self.p) * 0.1  # Initialize as a vector with one weight per feature
        self.b = random.randn() * 0.1  # Initialize as a scalar

    def forward(self, X):
        # Step 1: Calculate X @ w + b for each row in X
        linear_combination = X @ self.w + self.b

        # Step 2: Apply the logistic (sigmoid) function
        p = 1 / (1 + np.exp(-linear_combination))

        self.f = p

        # Return the result (no need to since we save it)
        # return p

    def predict(self, X):
        # Step 1: Calculate X @ w + b for each row in X
        linear_combination = X @ self.w + self.b

        # Step 2: Apply the logistic (sigmoid) function
        p = 1 / (1 + np.exp(-linear_combination))

        # Return the result
        return p

    def loss(self, X, y):
        # Step 1: Calculate probabilities using the forward function
        predictions = self.predict(X)

        # Step 2: Compute binary cross-entropy loss
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

        return loss

    def backward(self, X, y):
        # Step 1: Calculate the forward pass to get predictions
        predictions = self.f

        # Step 2: Calculate error (y - p_model)
        errors = y - predictions

        # Step 3: Compute gradients without looping
        self.l_grad_w = - (X.T @ errors) / self.N
        self.l_grad_b = - np.sum(errors) / self.N

        # Return the result (no need to since we save them)
        # return self.l_grad_w, self.l_grad_b

    def step(self):
        # Update weights and bias using the calculated gradients and learning rate
        self.w = self.w - self.lr * self.l_grad_w
        self.b = self.b - self.lr * self.l_grad_b

    def fit(self, X, y, iterations=10000, batch_size=None, show_step=1000, show_line=False):
        # 1) Verify inputs are numpy arrays and dimensions are compatible
        if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            raise ValueError("X and y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Incompatible dimensions: X has {} samples, but y has {}.".format(X.shape[0], y.shape[0]))

        # 2) Initialize parameters based on the number of features (columns) in X
        self.N = X.shape[0]
        self.p = X.shape[1]
        self.init_parameters()

        # 3) Randomly shuffle the data
        indices = np.arange(self.N)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        # Training loop
        for iteration in range(1, iterations + 1):
            # 4) Get batch indices
            if batch_size is None:
                X_batch, y_batch = X, y
            else:
                batch_indices = np.arange((iteration * batch_size) % self.N, ((iteration + 1) * batch_size) % self.N)
                X_batch, y_batch = X[batch_indices], y[batch_indices]

            # 5) Forward, backward, and gradient descent steps
            self.forward(X_batch)
            self.backward(X_batch, y_batch)
            self.step()

            # 6) Display loss every `show_step` iterations
            if iteration % show_step == 0:
                current_loss = self.loss(X, y)
                print(f"Iteration {iteration}, Loss: {current_loss:.4f}")
                if show_line:
                    self.show_line(X, y)  # Assuming `show_line()` is a defined method

    def show_line(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot data points for two classes, as well as the line
        corresponding to the model.
        """
        if (self.p != 2):
            print("Not plotting: Data is not 2-dimensional")
            return
        idx0 = (y == 0)
        idx1 = (y == 1)
        X0 = X[idx0, :2]
        X1 = X[idx1, :2]
        plt.plot(X0[:, 0], X0[:, 1], 'gx')
        plt.plot(X1[:, 0], X1[:, 1], 'ro')
        min_x = np.min(X, axis=0)
        max_x = np.max(X, axis=0)
        xline = np.arange(min_x[0], max_x[0], (max_x[0] - min_x[0]) / 100)
        yline = (self.w[0]*xline + self.b) / (-self.w[1])
        plt.plot(xline, yline, 'b')
        plt.show()