
<!-- README.md is generated from README.Rmd. Please edit that file -->

# Análise de churn

## Objetivo do projeto

A identificação de clientes que possam se desligar capacidade essencial
para qualuer empresa atualmente.  
Neste projeto pretendo explorar um conjunto de dados de clientes de uma
empresa e tentar prever o probabilidade de churn através de técnicas de
machine learning.

-   Para visualizar o projeto: [clique
    aqui](https://github.com/RodrigoFP51/Analise_churn/blob/master/Churn.md)

## Dados

Dados utilizados:
[Kaggle](https://www.kaggle.com/shubh0799/churn-modelling)

-   CustomerId: identificação do cliente;
-   Surname: sobrenome do cliente;
-   CreditScore: pontuação de credito do cliente;
-   Geography: país de onde o cliente pertence;
-   Gender: sexo do cliente;
-   Age: idade do cliente;
-   Tenure: tempo de que o cliente está com a empresa;
-   Balance: saldo da conta corrente;
-   NumOfProducts: número de produtos bancários adquiridos;
-   HasCrCard: se tem cartão de credito ou não;
-   IsActiveMember: se é um cliente com conta ativa;
-   EstimatedSalary: salário estimado do cliente;
-   Exited: se o cliente deixou de ser cliente do banco ou não;

## Modelos

-   O preprocessamento das variáveis envolveram os seguintes métodos:
    -   Transformação logarítmica nas variáveis numéricas.
    -   Normalização das variáveis numéricas.
    -   Criação de dummies com one hot encoding.
    -   Remoção de variáveis preditoras altamente correlacionadas.
    -   Remoção de variáveis com variância = zero.
    -   Aplicação do algoritmo ADASYN para balanceamento de classes.
-   Depois de testados diversos modelos de machine learning através do
    método de validação cruzada, chegou-se a conclusão de que o melhor
    modelo é um extreme gradient boosting (XGBoost).

## Resultados

-   O modelo XGBoost atingiu um F1-Score de 0.61 e acurácia de 0.84.  
-   A escolha das métricas foram justificadas no projeto.

## Conclusão

As variáveis idade, tenure (que indica quanto tempo o cliente está com a
empresa) e número de produtos são as mais importantes para prever o
churn, segundo o modelo. Enquanto isso, aspectos geográficos do cliente
ou o score de crédito não apresentam grande poder preditivo, o que
indica um bom caminho que a empresa pode seguir para tratar o problema.

![variable importance
plot](Churn_files/figure-gfm/unnamed-chunk-22-1.png)
