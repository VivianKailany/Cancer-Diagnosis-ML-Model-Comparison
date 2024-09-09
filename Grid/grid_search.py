import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


# Função para Grid Search
def Grid_Search_SVM(X, y, param_grid, kf):
    melhor_params = None
    melhor_media = -np.inf
    melhor_modelo = None

    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            for kernel in param_grid['kernel']:
                params = {'C': C, 'gamma': gamma, 'kernel': kernel}
                metricas = []

                for treino_index, validacao_index in kf.split(X):
                    X_treino, X_validacao = X[treino_index], X[validacao_index]
                    y_treino, y_validacao = y[treino_index], y[validacao_index]

                    # Normalizar os dados
                    scaler = StandardScaler()
                    X_treino_normalizado = scaler.fit_transform(X_treino)
                    X_validacao_normalizado = scaler.transform(X_validacao)

                    # Criar e treinar o modelo
                    model = SVC(C=C, gamma=gamma, kernel=kernel)
                    model.fit(X_treino_normalizado, y_treino)
                    y_pred = model.predict(X_validacao_normalizado)

                    # Avaliar o modelo
                    metrica = f1_score(y_validacao, y_pred)
                    metricas.append(metrica)

                media = np.mean(metricas)
                
                # Atualizar os melhores parâmetros
                if media > melhor_media:
                    melhor_media = media
                    melhor_params = params
                    melhor_modelo = SVC(C=C, gamma=gamma, kernel=kernel)
                    melhor_modelo.fit(StandardScaler().fit_transform(X), y)  # Ajuste no conjunto completo

    return melhor_modelo, melhor_params, melhor_media

# Função para Grid Search para DecisionTreeClassifier
def Grid_Search_DecisionTree(X, y, param_grid, kf):
    melhor_params = None
    melhor_media = -np.inf
    melhor_modelo = None

    for max_depth in param_grid['max_depth']:
        for min_samples_leaf in param_grid['min_samples_leaf']:
            params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
            metricas = []

            for treino_index, validacao_index in kf.split(X):
                X_treino, X_validacao = X[treino_index], X[validacao_index]
                y_treino, y_validacao = y[treino_index], y[validacao_index]

                # Normalizar os dados
                scaler = StandardScaler()
                X_treino_normalizado = scaler.fit_transform(X_treino)
                X_validacao_normalizado = scaler.transform(X_validacao)

                # Criar e treinar o modelo
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                model.fit(X_treino_normalizado, y_treino)
                y_pred = model.predict(X_validacao_normalizado)

                # Avaliar o modelo
                metrica = f1_score(y_validacao, y_pred)
                metricas.append(metrica)

            media = np.mean(metricas)
            
            # Atualizar os melhores parâmetros
            if media > melhor_media:
                melhor_media = media
                melhor_params = params
                melhor_modelo = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                melhor_modelo.fit(StandardScaler().fit_transform(X), y)  # Ajuste no conjunto completo

    return melhor_modelo, melhor_params, melhor_media