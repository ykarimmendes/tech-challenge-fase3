# Resultados do Participante 3 — Segurança, Validação, Explainability e Avaliação

## Objetivo
Garantir que o assistente médico atue apenas como ferramenta de apoio à decisão, com regras de segurança, validação humana, rastreabilidade, explainability, logs e critérios de avaliação.

## Insumos utilizados
Foram utilizados os arquivos gerados pelo Participante 2: condutas, protocolos, dataset QA, modelos de documentos, prontuários sintéticos e documentação de preprocessing/anonimização.

## Entregáveis gerados
- Matriz de segurança e validação.
- Regras de validação humana.
- Critérios e rubrica de avaliação.
- Casos de teste positivos e negativos.
- Template e exemplos de logs/auditoria.
- Plano de explainability.
- Prompts de validação e reescrita segura.

## Princípios de segurança
1. Não prescrever medicamentos.
2. Não fechar diagnóstico definitivo.
3. Não liberar paciente.
4. Não assinar ou finalizar documentos clínicos.
5. Citar protocolo/fonte quando houver orientação clínica.
6. Indicar validação humana obrigatória.
7. Solicitar dados faltantes quando necessário.
8. Registrar interações para auditoria.

## Uso no projeto
O Participante 1 pode usar os arquivos de regras, prompts e casos de teste na implementação. O Participante 4 pode usar os casos de teste e o plano de explainability no fluxo LangGraph e no vídeo. O relatório técnico pode aproveitar diretamente este documento.

Data de geração: 25/05/2026 14:48
