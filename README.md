# Lab 07 - Fine-Tuning de LLM com LoRA e QLoRA

## Descrição
Pipeline completo de fine-tuning do modelo Llama 2 7B utilizando técnicas de
eficiência de parâmetros (PEFT/LoRA) e quantização (QLoRA).

## Objetivo
Viabilizar o treinamento de um modelo de linguagem fundacional em hardware
limitado, utilizando quantização 4-bits e adaptadores LoRA para reduzir o
consumo de memória sem comprometer a qualidade do aprendizado.

## Estrutura do Projeto
- `lab07.ipynb` — notebook principal com todo o pipeline
- `treino.jsonl` — dataset sintético de treinamento (45 pares)
- `teste.jsonl` — dataset de teste (5 pares)

## Decisões Técnicas e Adaptações

### Dataset Sintético
O documento original especificava o uso da API da OpenAI (GPT-3.5/GPT-4)
para geração do dataset. Durante a execução, foram encontradas duas limitações:

- A API da OpenAI exige créditos pagos e o saldo estava zerado
- A API do Google Gemini (alternativa gratuita) atingiu o limite de cota do tier gratuito

Solução adotada: o dataset de 50 pares de instrução e resposta sobre
inteligência artificial e machine learning foi construído manualmente em Python,
respeitando o formato exigido (.jsonl) e a divisão 90/10 entre treino e teste.
O conteúdo cobre os tópicos centrais do curso, como BPE, WordPiece, LoRA,
transformers, MLM e CLM.

### Configuração do Modelo
- Modelo base: meta-llama/Llama-2-7b-hf (acesso aprovado pela Meta via Hugging Face)
- Quantização: 4-bits com NormalFloat4 (nf4) via bitsandbytes
- Compute dtype: bfloat16 (ajustado de float16 para compatibilidade com a GPU T4 do Colab)

### Configuração LoRA (conforme especificado)
- Rank (r): 64
- Alpha: 16
- Dropout: 0.1
- Task type: CAUSAL_LM

### Otimizador e Scheduler (conforme especificado)
- Otimizador: paged_adamw_32bit
- LR Scheduler: cosine
- Warmup steps: 2

### Ajustes na API do SFTTrainer
A versão instalada da biblioteca trl apresentou incompatibilidades com os
parâmetros `dataset_text_field` e `warmup_ratio`, que foram removidos em
versões mais recentes. O dataset foi pré-formatado antes de ser passado ao
trainer, e `warmup_ratio` foi substituído por `warmup_steps`.

## Como Executar
1. Acesse o Google Colab e ative uma GPU T4
2. Solicite acesso ao Llama 2 em huggingface.co/meta-llama/Llama-2-7b-hf
3. Gere um token de acesso no Hugging Face (settings > tokens)
4. Execute as células do notebook em ordem
5. Instale as dependências:
   pip install bitsandbytes transformers peft trl accelerate datasets

## Dependências
- transformers
- peft
- trl
- bitsandbytes
- accelerate
- datasets
- torch

## Uso de IA
Partes geradas/complementadas com IA, revisadas por Alcimar Rosal Benvindo Filho.

Especificamente, o Claude (Anthropic) foi utilizado para:
- Estruturar o pipeline de fine-tuning (passos 2, 3 e 4)
- Depurar erros de compatibilidade entre versões das bibliotecas
- Gerar o dataset sintético de 50 pares quando as APIs externas falharam

Todo o código foi revisado, compreendido e adaptado pelo autor antes da submissão.

## Versionamento
Tag v1.0 — versão final entregue para avaliação acadêmica