# Tech Challenge - Fase 3  
# Assistente Médico Acadêmico com LLM, LangChain, RAG, LangGraph, Segurança e Métricas

Projeto acadêmico desenvolvido para o **Tech Challenge - Fase 3**, com o objetivo de criar um assistente virtual médico capaz de apoiar profissionais de saúde na consulta a protocolos internos, organização de informações clínicas, geração de respostas contextualizadas, validação de segurança, rastreabilidade, auditoria e avaliação automática de desempenho.

O projeto utiliza uma **LLM local executada com Ollama**, integrada ao **LangChain**, com recuperação contextual por **RAG - Retrieval-Augmented Generation** e fluxo demonstrável com **LangGraph**. Também foram criadas bases sintéticas, mecanismos de segurança, logs de auditoria, explainability e cálculo de métricas do modelo.

> **Importante:** este projeto possui finalidade exclusivamente acadêmica e demonstrativa. O assistente não substitui médicos, enfermeiros, psicólogos ou qualquer profissional de saúde habilitado.

---

## Integrantes do grupo

- Michele Rodrigues Hempel Lima — RM369176 — engmichelerodrigues@gmail.com
- Karim Mendes Yehia — RM369430 — karim.mendes@gmail.com
- Wellington Fernandes do Carmo — RM369631 — wellingtonfernandes@energisa.com.br
- Rúben Gonçalves Rocha — RM370092 — ruben@energisa.com.br

## 1. Objetivo do projeto

O objetivo principal é desenvolver um **assistente médico acadêmico baseado em LLM**, capaz de:

- responder perguntas clínicas com apoio de protocolos hospitalares fictícios;
- recuperar informações relevantes antes da geração da resposta;
- contextualizar a resposta da LLM com dados estruturados simulados;
- demonstrar um pipeline de preparação para fine-tuning/adaptação;
- integrar LLM, LangChain, RAG e LangGraph;
- aplicar limites de segurança para evitar sugestões impróprias;
- indicar a fonte/protocolo utilizado na resposta;
- registrar logs para auditoria;
- calcular métricas de avaliação do modelo;
- gerar resultados para análise técnica no relatório.

---

## 2. Problema abordado

Hospitais e equipes médicas lidam com grande volume de informações clínicas, protocolos internos, prontuários, documentos, laudos, procedimentos e dúvidas recorrentes. A busca manual por essas informações pode dificultar a padronização das condutas e aumentar o tempo necessário para localizar informações relevantes.

Além disso, sistemas de IA aplicados à área médica precisam ser tratados com muito cuidado, pois respostas sem validação podem gerar riscos, como diagnóstico definitivo indevido, prescrição incorreta, liberação inadequada de pacientes ou ausência de encaminhamento profissional.

Dessa forma, o problema central do projeto é:

**Como desenvolver um assistente virtual médico baseado em LLM, capaz de consultar dados internos simulados, recuperar protocolos hospitalares e gerar respostas contextualizadas de forma segura, rastreável, auditável e validada?**

---

## 3. Visão geral da solução

A solução foi organizada em camadas:

1. **Base de dados sintética**  
   Contém protocolos hospitalares fictícios, perguntas e respostas médicas, prontuários sintéticos, modelos documentais e condutas clínicas.

2. **Pipeline de fine-tuning/adaptação simulada**  
   Organiza os dados em formato conversacional para demonstrar uma preparação para fine-tuning supervisionado.

3. **LangChain + RAG**  
   Recupera protocolos hospitalares fictícios relacionados à pergunta clínica.

4. **LLM local com Ollama**  
   Utiliza o modelo Qwen 2.5 3B para gerar respostas contextualizadas.

5. **LangGraph**  
   Organiza o fluxo completo do atendimento em etapas sequenciais.

6. **Segurança e validação**  
   Bloqueia respostas ou entradas inadequadas, como pedidos de prescrição, diagnóstico definitivo ou liberação sem avaliação.

7. **Logging e auditoria**  
   Registra as interações em arquivo de log.

8. **Métricas de avaliação**  
   Calcula indicadores como recuperação RAG, aderência ao protocolo, segurança clínica, validação humana, rastreabilidade e risco.

---

## 4. Tecnologias utilizadas

- Python
- Ollama
- Qwen 2.5 3B
- LangChain
- LangGraph
- RAG - Retrieval-Augmented Generation
- JSON
- JSONL
- CSV
- Prompt engineering clínico
- Guardrails de segurança
- Logging para auditoria
- Métricas automáticas de avaliação

---

## 5. Fluxo geral do assistente

```text
Entrada do paciente/profissional
        ↓
Verificação de dados
        ↓
Consulta a protocolos hospitalares fictícios
        ↓
Recuperação contextual via RAG
        ↓
Montagem do prompt clínico especializado
        ↓
Geração de resposta com LLM local
        ↓
Validação de segurança
        ↓
Resposta final com fonte
        ↓
Registro de auditoria

TECH-CHALLENGE-FASE3/
│
├── data/
│   ├── condutas_clinicas_validacao_humana.json
│   ├── dataset_finetuning_formatado.jsonl
│   ├── dataset_medico_qa_sintetico.jsonl
│   ├── modelos_laudos_receitas_procedimentos.json
│   ├── prontuarios_sinteticos.csv
│   └── protocolos_hospitalares_ficticios.json
│
├── Segurança/
│   ├── casos_teste_seguranca.csv
│   ├── casos_teste_seguranca.json
│   ├── criterios_avaliacao_respostas.csv
│   ├── criterios_avaliacao_respostas.json
│   ├── exemplos_logs_auditoria.csv
│   ├── exemplos_logs_auditoria.json
│   ├── matriz_seguranca_validacao.csv
│   ├── matriz_seguranca_validacao.json
│   ├── plano_explainability.json
│   ├── prompts_validacao_seguranca.json
│   ├── regras_validacao_humana.json
│   ├── rubrica_avaliacao_humana.csv
│   ├── rubrica_avaliacao_humana.json
│   └── template_logs_auditoria.json
│
├── src/
│   ├── app.py
│   ├── assistant.py
│   ├── avaliar_metricas.py
│   ├── fine_tuning_simulado.py
│   ├── langgraph_flow.py
│   ├── logger_auditoria.py
│   ├── retrieval.py
│   └── safety_validator.py
│
├── logs/
│   └── logs_auditoria.jsonl
│
├── README.md
├── requirements.txt
└── .gitignore

## Como rodar

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/app.py

Para executar o projeto, ative o ambiente virtual, instale as dependências com `pip install -r requirements.txt` e rode o assistente com `python src/app.py`.

Para rodar as métricas:

```md
Para calcular as métricas de avaliação do modelo, execute `python src/avaliar_metricas.py`.