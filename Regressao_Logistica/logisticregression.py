

import numpy as np

class RegressaoLogistica():
    def __init__(self, a, epocas):
        self.a = a
        self.epocas = epocas
        self.w = None

    def fit_gd(self, X, y):
        N, D = X.shape
        self.w = np.zeros((D + 1, 1))  # Inicializando os pesos
        # Adicionando uma coluna de uns a X
        X_com_1 = np.hstack((np.ones((N, 1)), X))

        # Treinamento usando gradiente descendente
        for _ in range(self.epocas):
            # Calcular as previsões
            y_pred = self.sigmoid(np.dot(X_com_1, self.w))
            # Calcular o gradiente
            erro = y_pred - y.reshape(-1, 1)
            grad = np.dot(X_com_1.T, erro) / N
            # Atualizar pesos
            self.w -= self.a * grad
        

    def predict(self, X):
        # Adicionando uma coluna de uns a X
        X_com_1 = np.hstack((np.ones((X.shape[0], 1)), X))
        z = np.dot(X_com_1, self.w) 
        # Calculando as previsões
        y_pred = self.sigmoid(z)
        
        # Convertendo as previsões para classes binárias
        y_pred_binary = (y_pred > 0.5).astype(int)
        return y_pred_binary.flatten()

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def calcula_custo(self, y, h):
        m = len(y)
        custo = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return custo


    