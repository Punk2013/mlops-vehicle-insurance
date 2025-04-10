from sklearn.svm import SVR

# NOT USED NOW
class VehicleInsuranceModel:
    def __init__(self, **params):
        self.model = SVR(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def partial_fit(self, X, y):
        self.model.partial_fit(X, y)

    def predict(self, X):
        self.model.predict(X)

    def get_params(self, deep=True):
        return self.model.get_params()