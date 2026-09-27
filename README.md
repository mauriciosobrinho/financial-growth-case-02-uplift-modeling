# Financial Growth - Case 2: Uplift Modeling

Case técnico end-to-end de uplift modeling para decidir quais clientes devem receber uma campanha financeira com capacidade limitada. A solução separa propensão de conversão de efeito incremental, preserva o desenho randomizado, usa validação temporal e traduz o CATE estimado em política econômica.

End-to-end uplift modeling case for deciding which customers should receive a capacity-constrained financial campaign. The solution separates conversion propensity from incremental treatment effect, preserves the randomized design, uses temporal validation, and translates estimated CATE into an economic policy.

## Pergunta causal

O objetivo é estimar:

`CATE(x) = E[Y(1) - Y(0) | X=x]`

O tratamento causal principal é `treatment_assigned` e o outcome é `conversion_30d`. A análise principal segue o princípio de Intention-to-Treat. Variáveis observadas depois da atribuição, como entrega, canal, custo realizado e navegação pós-campanha, não entram como covariáveis do modelo causal.

## Protocolo temporal

- Train: ondas 1 a 4
- Validation: onda 5
- OOT Final Test: onda 6

O champion é escolhido e congelado na onda de Validation. A onda OOT não participa da seleção de covariáveis, tuning, escolha do algoritmo ou definição da política.

## Políticas comparadas

- Random 20%
- Top Propensity 20%
- Top Uplift 20%

As políticas serão avaliadas por uplift, Qini, AUUC, conversões incrementais, margem incremental, custo, policy value e ROI.

## Modelos

- Propensity response benchmark entre clientes tratados
- S-Learner
- T-Learner / Two-model
- X-Learner com nuisance models em cross-fitting

Causal Forest e Uplift Tree foram avaliados quanto à disponibilidade, mas não executados porque `econml` e `causalml` não estavam instalados no ambiente. O notebook registra essa limitação sem atribuir esses nomes a aproximações metodologicamente diferentes.

## Resultados principais

- Randomização: maior Standardized Mean Difference no Train de 0,009.
- Delivery rate entre tratados: 92,4%.
- Champion: S-Learner, escolhido exclusivamente na onda 5.
- Validation: Qini 0,00352, AUUC 0,02290 e Uplift@20% de 12,22 p.p.
- OOT: ITT de 7,70 p.p. (IC95%: 6,91 a 8,48 p.p.).
- OOT: Qini 0,00297, AUUC 0,02216 e Uplift@20% de 10,83 p.p.
- Top Uplift 20%: aproximadamente 1.083 conversões incrementais e valor líquido incremental de 74.178 unidades monetárias.
- Top Propensity 20%: aproximadamente 889 conversões incrementais e valor líquido incremental de 73.311 unidades monetárias.
- Random 20%: aproximadamente 762 conversões incrementais e valor líquido incremental médio de 54.511 unidades monetárias em 200 sorteios.

Top Uplift melhora claramente as conversões incrementais. A vantagem de valor sobre Top Propensity é positiva, mas estreita, indicando que a próxima evolução deve modelar margem incremental líquida diretamente.

## Estrutura

```text
data/raw/          base pública de modelagem e dicionário
notebooks/         notebook executado de ponta a ponta
outputs/figures/   gráficos analíticos
outputs/tables/    tabelas de diagnóstico e negócio
outputs/metrics/   métricas de validação e OOT
outputs/models/    objetos do champion congelado
presentation/      apresentação executiva de cinco slides
src/               funções reutilizáveis essenciais
tests/             testes das métricas e políticas
```

O arquivo privado com o Data Generating Process não integra este repositório e não é acessado pelo pipeline de modelagem.

## Execução

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m unittest -v tests.test_uplift_utils
python -m src.run_case
jupyter nbconvert --to notebook --execute notebooks/01_end_to_end_case.ipynb --output 01_end_to_end_case.ipynb --ExecutePreprocessor.timeout=3600
```

## Limitação central

Resultados de uplift dependem das hipóteses do desenho causal, da estabilidade entre ondas e da política operacional. A implantação deve ser validada por novo experimento randomizado com holdout persistente.

**Propensity prioriza clientes, mas ainda não demonstra causalidade.** Uplift modeling usa a variação experimental para estimar efeito incremental, mas a política continua exigindo validação prospectiva.
