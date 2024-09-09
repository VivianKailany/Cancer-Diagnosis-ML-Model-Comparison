import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from Metricas.metricas import acuracia, precisao, revocacao, f1

def validacao_cruzada(modelo, X, y, num_folds=10):
    skf = StratifiedKFold(n_splits=num_folds)
    acuracias = []
    precisoes = []
    revocacoes = []
    f1_scores = []

    for treino_index, validacao_index in skf.split(X, y):

        X_treinamento, X_validacao = X[treino_index], X[validacao_index]
        y_treinamento, y_validacao = y[treino_index], y[validacao_index]
        
        # Normalizar os dados de treinamento
        escala = StandardScaler()
        X_normalizado_treino = escala.fit_transform(X_treinamento)

        # Normalizar os dados de validacao
        X_normalizado_validacao = escala.transform(X_validacao)

        modelo.fit_gd(X_normalizado_treino, y_treinamento)
        previsoes = modelo.predict(X_normalizado_validacao)
        
        acuracia_fold = acuracia(y_validacao, previsoes)
        acuracias.append(acuracia_fold)
        
        precisao_fold = precisao(y_validacao, previsoes, classe_positiva=1)
        precisoes.append(precisao_fold)
        
        revocacao_fold = revocacao(y_validacao, previsoes, classe_positiva=1)
        revocacoes.append(revocacao_fold)
        
        f1_fold = f1(precisao_fold, revocacao_fold)
        f1_scores.append(f1_fold)
        
    media_acuracia = np.mean(acuracias)
    media_precisao = np.mean(precisoes)
    media_revocacao = np.mean(revocacoes)
    media_f1 = np.mean(f1_scores)

    return media_acuracia, media_precisao, media_revocacao, media_f1

def calcular_estatisticas_metricas(metricas):
    media_metricas = np.mean(metricas)
    desvio_padrao_metricas = np.std(metricas)
    return media_metricas, desvio_padrao_metricas