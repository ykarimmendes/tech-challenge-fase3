from typing import Optional, TypedDict, Dict, Any

from langgraph.graph import StateGraph, START, END

from assistant import llm, montar_prompt
from retrieval import buscar_protocolo
from safety_validator import validar_resposta_segura
from logger_auditoria import registrar_log


class EstadoAtendimento(TypedDict, total=False):
    entrada_paciente: str
    dados_validos: bool
    motivo_validacao: str

    score_protocolo: Optional[int]
    protocolo: Optional[Dict[str, Any]]

    prompt: str
    resposta_llm: str

    validacao_seguranca: Dict[str, Any]
    resposta_final: str


def verificar_dados(state: EstadoAtendimento) -> EstadoAtendimento:
    """
    Nó 1: verifica se a entrada possui informação mínima.
    """

    entrada = state.get("entrada_paciente", "").strip()

    if not entrada:
        return {
            "dados_validos": False,
            "motivo_validacao": "Nenhuma entrada foi informada pelo usuário.",
        }

    if len(entrada) < 10:
        return {
            "dados_validos": False,
            "motivo_validacao": "A entrada é muito curta para consulta segura ao protocolo.",
        }

    return {
        "dados_validos": True,
        "motivo_validacao": "Entrada válida para consulta aos protocolos.",
    }


def consultar_protocolos(state: EstadoAtendimento) -> EstadoAtendimento:
    """
    Nó 2: consulta a base de protocolos por meio do retrieval.py.
    """

    if not state.get("dados_validos"):
        return {
            "score_protocolo": None,
            "protocolo": None,
        }

    resultado = buscar_protocolo(state["entrada_paciente"])

    if resultado is None:
        return {
            "score_protocolo": None,
            "protocolo": None,
        }

    score, protocolo = resultado

    return {
        "score_protocolo": score,
        "protocolo": protocolo,
    }


def gerar_resposta(state: EstadoAtendimento) -> EstadoAtendimento:
    """
    Nó 3: gera resposta com a LLM usando o prompt clínico.
    """

    protocolo = state.get("protocolo")

    if not state.get("dados_validos"):
        return {
            "resposta_llm": (
                "Não foi possível gerar resposta clínica, pois os dados informados "
                "são insuficientes. Recomenda-se fornecer mais informações e buscar "
                "avaliação profissional."
            )
        }

    if protocolo is None:
        return {
            "resposta_llm": (
                "Nenhum protocolo relacionado foi encontrado para a entrada informada. "
                "Não é seguro gerar uma orientação clínica sem contexto adequado. "
                "Recomenda-se avaliação por profissional de saúde."
            )
        }

    prompt = montar_prompt(state["entrada_paciente"], protocolo)
    resposta = llm.invoke(prompt)

    return {
        "prompt": prompt,
        "resposta_llm": resposta,
    }


def validar_seguranca(state: EstadoAtendimento) -> EstadoAtendimento:
    """
    Nó 4: valida a entrada e a resposta da LLM.
    """

    resposta = state.get("resposta_llm", "")
    entrada = state.get("entrada_paciente", "")

    validacao = validar_resposta_segura(
        resposta=resposta,
        entrada_usuario=entrada,
    )

    return {
        "validacao_seguranca": validacao,
    }


def montar_resposta_final(state: EstadoAtendimento) -> EstadoAtendimento:
    """
    Nó 5: monta a resposta final com fonte, protocolo e validação.
    """

    protocolo = state.get("protocolo")
    validacao = state.get("validacao_seguranca", {})
    resposta_llm = state.get("resposta_llm", "")

    if protocolo is None:
        resposta_final = f"""
=== FLUXO LANGGRAPH ===
Entrada verificada: {state.get("motivo_validacao")}

=== PROTOCOLO RECUPERADO ===
Nenhum protocolo foi recuperado.

=== VALIDAÇÃO DE SEGURANÇA ===
Status: RESPOSTA CONSERVADORA
Motivo: ausência de protocolo relacionado.

=== RESPOSTA FINAL ===
{resposta_llm}
"""
    elif not validacao.get("segura", False):
        resposta_final = f"""
=== FLUXO LANGGRAPH ===
1. Entrada recebida
2. Dados verificados
3. Protocolo consultado
4. Resposta gerada pela LLM
5. Validação de segurança executada
6. Resposta insegura bloqueada

=== PROTOCOLO RECUPERADO ===
Score: {state.get("score_protocolo")}
ID: {protocolo.get("id_protocolo")}
Título: {protocolo.get("titulo")}
Área: {protocolo.get("area_clinica")}
Risco: {protocolo.get("nivel_risco_padrao")}
Fonte: {protocolo.get("fonte_simulada")}

=== VALIDAÇÃO DE SEGURANÇA ===
Status: REPROVADA
Motivo: {validacao.get("motivo")}
Ocorrências encontradas: {validacao.get("ocorrencias")}
Origem da reprovação: {validacao.get("origem")}

=== RESPOSTA FINAL SEGURA ===
A resposta foi bloqueada pelo validador de segurança, pois pode conter solicitação ou orientação clínica inadequada.
O assistente não deve prescrever medicamentos, fechar diagnóstico definitivo ou dispensar avaliação profissional.

Recomenda-se avaliação por profissional de saúde habilitado.

=== FONTE UTILIZADA ===
{protocolo.get("fonte_simulada")}
"""
    else:
        resposta_final = f"""
=== FLUXO LANGGRAPH ===
1. Entrada recebida
2. Dados verificados
3. Protocolo consultado
4. Resposta gerada pela LLM
5. Segurança validada
6. Resposta final com fonte

=== PROTOCOLO RECUPERADO ===
Score: {state.get("score_protocolo")}
ID: {protocolo.get("id_protocolo")}
Título: {protocolo.get("titulo")}
Área: {protocolo.get("area_clinica")}
Risco: {protocolo.get("nivel_risco_padrao")}
Fonte: {protocolo.get("fonte_simulada")}

=== VALIDAÇÃO DE SEGURANÇA ===
Status: APROVADA
Motivo: {validacao.get("motivo")}

=== RESPOSTA FINAL ===
{resposta_llm}

=== FONTE UTILIZADA ===
{protocolo.get("fonte_simulada")}
"""

    return {
        "resposta_final": resposta_final,
    }


def registrar_auditoria(state: EstadoAtendimento) -> EstadoAtendimento:
    """
    Nó 6: registra auditoria da execução.
    """

    protocolo = state.get("protocolo") or {}

    registrar_log(
        {
            "entrada_paciente": state.get("entrada_paciente"),
            "dados_validos": state.get("dados_validos"),
            "motivo_validacao": state.get("motivo_validacao"),
            "score_protocolo": state.get("score_protocolo"),
            "id_protocolo": protocolo.get("id_protocolo"),
            "titulo_protocolo": protocolo.get("titulo"),
            "area_clinica": protocolo.get("area_clinica"),
            "nivel_risco": protocolo.get("nivel_risco_padrao"),
            "fonte": protocolo.get("fonte_simulada"),
            "validacao_seguranca": state.get("validacao_seguranca"),
            "resposta_final": state.get("resposta_final"),
        }
    )

    return state


def criar_fluxo_langgraph():
    """
    Cria e compila o fluxo LangGraph.
    """

    workflow = StateGraph(EstadoAtendimento)

    workflow.add_node("verificar_dados", verificar_dados)
    workflow.add_node("consultar_protocolos", consultar_protocolos)
    workflow.add_node("gerar_resposta", gerar_resposta)
    workflow.add_node("validar_seguranca", validar_seguranca)
    workflow.add_node("montar_resposta_final", montar_resposta_final)
    workflow.add_node("registrar_auditoria", registrar_auditoria)

    workflow.add_edge(START, "verificar_dados")
    workflow.add_edge("verificar_dados", "consultar_protocolos")
    workflow.add_edge("consultar_protocolos", "gerar_resposta")
    workflow.add_edge("gerar_resposta", "validar_seguranca")
    workflow.add_edge("validar_seguranca", "montar_resposta_final")
    workflow.add_edge("montar_resposta_final", "registrar_auditoria")
    workflow.add_edge("registrar_auditoria", END)

    return workflow.compile()


def executar_fluxo(entrada_paciente: str) -> EstadoAtendimento:
    """
    Executa o fluxo completo.
    """

    app = criar_fluxo_langgraph()

    resultado = app.invoke(
        {
            "entrada_paciente": entrada_paciente,
        }
    )

    return resultado