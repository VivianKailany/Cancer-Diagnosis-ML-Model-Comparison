# Cancer-Diagnosis-ML-Model-Comparison

Projeto de comparação de modelos de aprendizado de máquina para a detecção de câncer de mama.

## Dataset

O conjunto de dados utilizado é o *Breast Cancer Wisconsin Dataset*, que pode ser obtido através do repositório público da UCI Machine Learning Repository. Ele contém 569 instâncias e 30 atributos, representando características de células mamárias, classificadas como benignas ou malignas.

- **Fonte do Dataset**: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

## Métricas de Avaliação

As seguintes métricas foram usadas para comparar o desempenho dos modelos:

- **Acurácia**: Proporção de previsões corretas entre todas as amostras.
- **Precisão**: Proporção de verdadeiros positivos entre as previsões positivas.
- **Revocação**: Proporção de verdadeiros positivos entre todas as amostras positivas.
- **F1-Score**: Média harmônica entre precisão e recall.

## Resultados

Os resultados comparativos dos três modelos, usando validação cruzada com 10 folds, são apresentados abaixo:

| Modelo               | Acurácia | Precisão | Revocação | F1-Score |
|----------------------|----------|----------|-----------|----------|
| Regressão Logística   | 99%      | 100%     | 97%       | 98%      |
| SVM                  | 98%      | 100%     | 95%       | 97%      |
| Árvore de Decisão     | 93%      | 92%      | 90%       | 91%      |

## Conclusão

A análise comparativa mostrou que o modelo de Regressão Logística apresentou um desempenho ligeiramente superior em termos de acurácia e equilíbrio geral entre as previsões positivas e negativas. A Regressão Logística é uma boa escolha para este problema específico de diagnóstico de câncer de mama.
