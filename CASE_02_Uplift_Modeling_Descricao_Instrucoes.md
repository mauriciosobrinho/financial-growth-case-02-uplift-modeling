# CASE 2 - Uplift Modeling / Incremental Targeting

## 1. Contexto do desafio

Uma instituição financeira digital executa campanhas mensais para estimular a contratação de um produto financeiro. A campanha possui orçamento e capacidade operacional para atingir apenas 20% da população elegível.

O problema não é simplesmente identificar quem possui maior probabilidade de converter.

A pergunta de negócio é:

> Quem possui maior probabilidade de converter especificamente por causa da intervenção?

Essa distinção separa propensity modeling de causal targeting.

Um modelo de propensity tende a priorizar clientes com alta probabilidade de conversão, incluindo clientes que poderiam contratar mesmo sem receber campanha. Para Growth, o objetivo é concentrar investimento nos clientes cujo comportamento pode ser alterado pela ação.

## 2. Objetivo analítico

Estimar heterogeneidade de efeito de tratamento e produzir um ranking de clientes baseado no efeito incremental esperado.

Formalmente:

`CATE(x) = E[Y(1) - Y(0) | X=x]`

onde:

- `Y(1)` representa o outcome potencial sob tratamento;
- `Y(0)` representa o outcome potencial sob controle;
- apenas um dos dois é observado para cada cliente.

O case deve transformar esse efeito incremental em uma política de targeting limitada a 20% da população.

## 3. Ajustes metodológicos importantes em relação à formulação inicial

A composição original do case foi preservada, mas alguns pontos foram refinados para tornar o exercício causalmente correto e mais próximo de uma avaliação Senior.

### 3.1 Treatment assignment versus message delivery

`treatment_assigned` é a variável de randomização e deve ser o tratamento principal para a análise causal.

`message_delivered` ocorre depois da atribuição e representa non-compliance operacional.

Usar apenas clientes com mensagem entregue ou tratar `message_delivered` como se fosse a variável randomizada pode introduzir viés de seleção.

O efeito primário recomendado é, portanto, Intention-to-Treat, ITT.

### 3.2 Channel é pós-atribuição

`channel` representa a execução operacional do tratamento.

No case principal de tratamento binário, channel não deve ser utilizado como feature pré-tratamento.

Uma análise causal específica por canal exigiria um problema de multi-treatment ou desenho experimental apropriado e deve ser tratada como extensão, não como atalho.

### 3.3 Propensity benchmark

O benchmark "Top Propensity" deve representar um modelo de resposta tradicional.

Uma implementação recomendada é treinar um modelo preditivo entre clientes tratados para estimar:

`P(Y=1 | T=1, X)`

e aplicar esse modelo à população elegível.

Esse ranking tende a privilegiar quem provavelmente converteria sob tratamento, mas não necessariamente quem apresenta maior efeito incremental.

### 3.4 Avaliação temporal

Como existem seis ondas mensais, a generalização deve ser testada temporalmente.

Protocolo recomendado:

- Train: ondas 1 a 4;
- Validation: onda 5;
- OOT Final Test: onda 6.

O champion de uplift deve ser escolhido antes de consultar a onda 6.

## 4. Base fornecida

Arquivo único de modelagem:

`data/raw/case_02_uplift_modeling_raw.csv`

Características:

- 300.000 clientes únicos;
- 6 ondas mensais de campanha;
- aproximadamente 50 mil clientes por onda;
- atribuição de tratamento randomizada aproximadamente 50/50 em cada onda;
- outcome binário `conversion_30d`;
- outcomes financeiros `revenue_30d` e `margin_30d`;
- non-compliance por falha de entrega;
- heterogeneidade real de efeito;
- drift temporal moderado;
- problemas deliberados de qualidade e timing de variáveis.

Nenhuma divisão Train, Validation ou OOT é fornecida pronta.

## 5. Ground truth privado

Existe um arquivo privado apenas para auditoria da geração sintética:

`private_do_not_use_in_modeling/case_02_ground_truth_private.csv`

Campos:

- `customer_id`
- `campaign_wave`
- `p_control`
- `p_treatment`
- `true_cate`
- `latent_group`

Esse arquivo NÃO pode ser disponibilizado durante a resolução e NÃO pode ser utilizado em treinamento, tuning, seleção de modelo ou avaliação do case.

Ele existe somente para, depois de concluída a solução, comparar o efeito estimado com o Data Generating Process conhecido.

## 6. Estrutura latente do DGP

A geração sintética produz aproximadamente quatro perfis causais:

- 35% Persuadables
- 25% Sure Things
- 30% Lost Causes
- 10% Sleeping Dogs

Esses grupos não aparecem no dataset de modelagem.

Eles emergem de interações não lineares entre comportamento transacional, engajamento digital, afinidade, histórico de campanhas, renda, atividade e fatores latentes.

### Persuadables

Baixa ou média probabilidade de conversão sem campanha e aumento relevante sob tratamento.

### Sure Things

Alta probabilidade de converter independentemente da campanha.

### Lost Causes

Baixa probabilidade de conversão tanto no controle quanto no tratamento.

### Sleeping Dogs

Clientes para os quais a intervenção pode reduzir a probabilidade de conversão.

O objetivo de uplift não é classificar diretamente esses quatro grupos, mas estimar heterogeneidade de efeito suficientemente bem para melhorar a política de targeting.

## 7. Variáveis

A base contém as principais famílias comportamentais do Case 1:

- identificação e tempo;
- segmentação;
- RFM;
- Pix;
- saldo e conta;
- cartão;
- produtos;
- atividade digital;
- CRM;
- tendências comportamentais.

E acrescenta variáveis de experimento e outcome:

- `treatment_assigned`
- `message_delivered`
- `channel`
- `campaign_cost`
- `post_campaign_product_page_views_7d`
- `conversion_30d`
- `revenue_30d`
- `margin_30d`

O dicionário completo está em:

`case_02_data_dictionary.csv`

## 8. Problemas deliberadamente presentes

A base foi construída para exigir investigação.

Existem:

- missing crescente em variáveis digitais ao longo das ondas;
- missing potencialmente informativo;
- caudas longas em valores financeiros;
- outliers;
- zeros legítimos em saldo;
- códigos sentinela;
- forte correlação entre janelas de 30 e 90 dias;
- mudança moderada de composição da população;
- drift de baseline conversion;
- non-compliance entre assignment e delivery;
- variáveis pós-tratamento que não devem ser usadas como covariáveis causais;
- heterogeneidade real de tratamento;
- efeito negativo em uma parcela da população.

## 9. Sanity checks experimentais obrigatórios

Antes de qualquer uplift model:

1. verificar proporção de treatment e control;
2. verificar randomização por onda;
3. comparar covariáveis pré-tratamento entre grupos;
4. calcular Standardized Mean Differences;
5. verificar taxa de entrega entre tratados;
6. analisar attrition/missing;
7. verificar positivity/overlap;
8. calcular ATE/ITT simples por difference in means;
9. construir intervalo de confiança para o ATE.

Como o tratamento é randomizado, não é necessário estimar propensity score de tratamento para identificação causal primária.

## 10. Baseline causal

### Difference in means

Calcular:

`E[Y | T=1] - E[Y | T=0]`

globalmente e por onda.

Esse estimador fornece o ATE/ITT médio, mas não resolve targeting individualizado.

## 11. Modelos de uplift

Comparar, quando disponíveis:

- Two-model / T-Learner;
- S-Learner;
- X-Learner;
- Causal Forest;
- Uplift Tree.

Observação:

Two-model e T-Learner representam essencialmente a mesma família conceitual. Não devem ser tratados artificialmente como paradigmas distintos apenas para aumentar o benchmark.

Como extensão, podem ser avaliados:

- DR-Learner;
- R-Learner.

A prioridade é clareza metodológica, não quantidade de algoritmos.

## 12. Propensity benchmark

Construir um modelo tradicional de resposta usando apenas covariáveis pré-tratamento.

Uma estratégia recomendada:

- treinar entre observações tratadas;
- target = `conversion_30d`;
- estimar `P(Y=1 | T=1, X)`;
- aplicar o modelo a todos os clientes elegíveis.

Esse score será utilizado para criar a política "Top Propensity 20%".

## 13. Particionamento temporal

Construir dentro do notebook:

- Train: ondas 1 a 4;
- Validation: onda 5;
- OOT Final Test: onda 6.

Não usar random split como estratégia principal.

### Regra crítica

O OOT deve permanecer completamente isolado durante:

- feature selection;
- tuning;
- escolha do uplift champion;
- escolha de hiperparâmetros;
- escolha da política final.

O champion deve ser congelado antes da avaliação na onda 6.

## 14. Métricas de uplift

Avaliar:

- Uplift@10%
- Uplift@20%
- Qini Curve
- Qini Coefficient
- AUUC
- cumulative incremental gain
- incremental conversions
- incremental margin
- policy value

As curvas devem ser avaliadas no Validation e apenas depois no OOT final.

## 15. Comparação de políticas

Criar três políticas com capacidade de 20%:

### A. Random

Selecionar aleatoriamente 20%.

### B. Top Propensity

Selecionar os 20% com maior score de resposta preditiva.

### C. Top Uplift

Selecionar os 20% com maior CATE estimado pelo champion.

Comparar:

- clientes impactados;
- conversion rate observada;
- uplift estimado;
- incremental conversions;
- incremental revenue;
- incremental margin;
- campaign cost;
- net incremental value;
- ROI;
- policy value.

## 16. Métrica econômica

A política final deve maximizar valor incremental e não conversão bruta.

Uma forma conceitual:

`Expected Incremental Value = Estimated Uplift x Expected Margin - Expected Campaign Cost`

Não assumir que toda conversão observada entre tratados foi causada pela campanha.

## 17. Non-compliance

A análise principal deve permanecer ITT usando `treatment_assigned`.

Analisar separadamente:

- assignment rate;
- delivery rate;
- conversão por assignment;
- conversão por delivery.

Não fazer análise per-protocol e chamá-la de efeito causal sem discutir viés.

Como extensão avançada, pode-se discutir CACE/LATE com instrumental variables, usando assignment como instrumento para delivery, mas isso não é obrigatório para a solução principal.

## 18. Leakage e post-treatment bias

Variáveis como:

- `message_delivered`
- `channel`
- `campaign_cost`
- `post_campaign_product_page_views_7d`

não devem ser usadas indiscriminadamente como features pré-tratamento em modelos de CATE.

O notebook deve classificar variáveis por timing e justificar o conjunto de covariáveis utilizado.

## 19. Interpretação

A conclusão deve responder:

1. Existe efeito médio da campanha?
2. Esse efeito é estatisticamente e economicamente relevante?
3. Existe heterogeneidade de efeito?
4. O uplift model consegue ordenar clientes por ganho incremental?
5. Top Uplift supera Top Propensity?
6. Qual política gera maior margem incremental?
7. Existe grupo com efeito negativo?
8. O resultado é estável temporalmente?
9. Como validar essa política em produção?

## 20. Limitações causais

Discutir:

- SUTVA;
- interference;
- non-compliance;
- generalização entre ondas;
- tratamento binário versus canais diferentes;
- estabilidade temporal;
- múltiplas campanhas;
- mudanças de comportamento ao longo do tempo.

---

# MASTER PROMPT DE IMPLEMENTAÇÃO - CASE 2

Você atuará como Senior Data Scientist responsável por implementar um case técnico completo de Uplift Modeling aplicado a Growth em Financial Services.

O projeto deve ser desenvolvido como se fosse uma entrega de desafio técnico para uma posição Senior Data Scientist.

## PRINCÍPIO CENTRAL

A solução deve priorizar:

1. problema de negócio;
2. desenho experimental;
3. causalidade;
4. qualidade dos dados;
5. rigor estatístico;
6. heterogeneidade de tratamento;
7. validação temporal;
8. comparação de políticas;
9. impacto econômico;
10. comunicação.

Evite overengineering.

A primeira versão deve ser compreensível em UM notebook Jupyter executável de ponta a ponta.

## REPOSITÓRIO

Criar:

`financial-growth-case-02-uplift-modeling`

Estrutura mínima:

```text
README.md
requirements.txt

data/
    raw/

notebooks/
    01_end_to_end_case.ipynb

outputs/
    figures/
    tables/
    metrics/
    models/

presentation/

src/
    somente funções reutilizáveis que aumentem clareza
```

## ARQUIVO DE ENTRADA

Utilizar exclusivamente:

`data/raw/case_02_uplift_modeling_raw.csv`

Não utilizar nenhum arquivo da pasta `private_do_not_use_in_modeling`.

## ESTRUTURA DO NOTEBOOK

1. Contexto e problema de negócio
2. Pergunta causal
3. Unidade de análise
4. Population definition
5. Treatment definition
6. Outcome definition
7. Decision timestamp
8. Observation window
9. Performance window
10. Bibliotecas e seeds
11. Carga da base raw
12. Data Quality Report
13. Tipagem das variáveis
14. Missing values
15. Cardinalidade
16. Duplicidades
17. Outliers
18. Distribuições
19. Campaign waves
20. Treatment allocation
21. Conversion rate por treatment
22. Randomization checks
23. Covariate balance
24. Standardized Mean Differences
25. Non-compliance analysis
26. Leakage e post-treatment review
27. Drift temporal
28. Hipóteses de negócio
29. Seleção de covariáveis pré-tratamento
30. Construção das partições temporais
31. Difference in means / ATE / ITT
32. Intervalo de confiança
33. Propensity response benchmark
34. S-Learner
35. T-Learner / Two-model
36. X-Learner
37. Causal Forest, se disponível
38. Uplift Tree, se disponível
39. Hyperparameter tuning moderado
40. Qini e AUUC no Validation
41. Escolha do champion sem consultar OOT
42. Congelamento da solução
43. Teste final na onda OOT
44. Uplift por quantis
45. Segment analysis
46. Política Random 20%
47. Política Top Propensity 20%
48. Política Top Uplift 20%
49. Incremental conversions
50. Incremental revenue
51. Incremental margin
52. Campaign cost
53. Policy value
54. ROI
55. Limitações
56. Monitoramento
57. Próximos passos
58. Conclusão executiva

## DESENHO EXPERIMENTAL

`treatment_assigned` é a variável de randomização.

A análise causal principal deve ser Intention-to-Treat.

Não substituir `treatment_assigned` por `message_delivered`.

Não condicionar a análise principal apenas aos clientes com mensagem entregue.

## COVARIÁVEIS

Modelos de uplift devem usar apenas informações conhecidas antes da randomização.

Identificar e remover variáveis pós-treatment.

Documentar a decisão.

## RANDOMIZATION CHECK

Avaliar globalmente e por campaign_wave:

- tamanho dos grupos;
- treatment rate;
- event rate;
- médias e distribuições das principais features;
- Standardized Mean Differences.

Não usar p-values de balance como único diagnóstico.

## PARTIÇÃO TEMPORAL

Construir:

Train:
ondas 1 a 4.

Validation:
onda 5.

OOT Final Test:
onda 6.

O OOT deve permanecer isolado até o champion estar congelado.

## BASELINE

Calcular inicialmente:

`ATE_ITT = mean(Y | T=1) - mean(Y | T=0)`

com intervalo de confiança.

## PROPENSITY RESPONSE BENCHMARK

Treinar um modelo preditivo de conversão no grupo tratado usando somente covariáveis pré-treatment.

Estimar:

`P(Y=1 | T=1, X)`

Aplicar a todos os clientes e produzir o ranking Top Propensity.

O objetivo é demonstrar empiricamente a diferença entre response prediction e incremental response.

## UPLIFT MODELS

Comparar:

- S-Learner
- T-Learner / Two-model
- X-Learner
- Causal Forest, se biblioteca disponível
- Uplift Tree, se biblioteca disponível

Como extensão, DR-Learner ou R-Learner podem ser incluídos se não prejudicarem a clareza.

## MÉTRICAS

- Uplift@10%
- Uplift@20%
- Qini Curve
- Qini Coefficient
- AUUC
- cumulative incremental gain
- incremental conversions
- policy value

Não escolher o champion apenas pela aparência de uma curva.

## CHAMPION

Escolher o champion na onda de Validation.

Congelar:

- covariáveis;
- preprocessing;
- hiperparâmetros;
- algoritmo;
- política de ranking.

Somente depois abrir OOT.

## OOT FINAL TEST

Avaliar na onda 6:

- ATE observado;
- Qini;
- AUUC;
- Uplift@10%;
- Uplift@20%;
- estabilidade por segmentos;
- política Random;
- política Top Propensity;
- política Top Uplift;
- impacto financeiro.

Não trocar retrospectivamente o champion caso outro algoritmo pareça superior no OOT.

## COMPARAÇÃO DE POLÍTICAS

Capacidade máxima:

20% da população.

Comparar:

A. Random 20%.

B. Top Propensity 20%.

C. Top Uplift 20%.

Calcular:

- tamanho da audiência;
- conversões;
- incremental conversions;
- incremental revenue;
- incremental margin;
- campaign cost;
- net incremental value;
- ROI.

## POST-TREATMENT BIAS

Investigar explicitamente:

- message_delivered;
- channel;
- campaign_cost;
- post_campaign_product_page_views_7d.

Essas variáveis não podem ser utilizadas como covariáveis pré-tratamento do CATE principal.

## NON-COMPLIANCE

Mensurar delivery rate.

Explicar por que comparação entre delivered e non-delivered não preserva necessariamente randomização.

A análise principal permanece ITT.

## BUSINESS CONCLUSION

Responder:

1. Quem deve ser impactado?
2. Quanto melhor Top Uplift é que Random?
3. Quanto melhor Top Uplift é que Top Propensity?
4. Qual é o ganho incremental esperado?
5. Qual é o ROI?
6. Existem Sleeping Dogs?
7. Como evitar impacto negativo?
8. Como validar em produção?
9. Como monitorar drift de uplift e policy value?

## APRESENTAÇÃO

Estruturar conteúdo para no máximo cinco slides:

1. Business Problem & Experiment Design
2. Data Quality, Randomization & Causal Framing
3. Uplift Modeling & Validation
4. Policy Comparison: Random vs Propensity vs Uplift
5. Incremental Value, Risks & Next Steps

A apresentação deve enfatizar decisão de negócio, não apenas algoritmo.
