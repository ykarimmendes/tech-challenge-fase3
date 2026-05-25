import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_QA_DATASET = BASE_DIR / "data" / "dataset_medico_qa_sintetico.jsonl"

INPUT_MODELOS_DOCUMENTOS = (
    BASE_DIR
    / "data"
    / "modelos_laudos_receitas_procedimentos.json"
)

OUTPUT_DATASET = BASE_DIR / "data" / "dataset_finetuning_formatado.jsonl"


def limpar_texto(texto):
    if not texto:
        return ""

    texto = texto.strip()
    texto = re.sub(r"\s+", " ", texto)

    return texto


def anonimizar_texto(texto):
    if not texto:
        return ""

    texto = re.sub(
        r"\b\d{3}\.\d{3}\.\d{3}\-\d{2}\b",
        "[CPF_REMOVIDO]",
        texto
    )

    texto = re.sub(
        r"\b\d{11}\b",
        "[CPF_REMOVIDO]",
        texto
    )

    texto = re.sub(
        r"\b\S+@\S+\.\S+\b",
        "[EMAIL_REMOVIDO]",
        texto
    )

    texto = re.sub(
        r"\b\d{2}/\d{2}/\d{4}\b",
        "[DATA_REMOVIDA]",
        texto
    )

    return texto


def preparar_texto(texto):
    texto = limpar_texto(texto)
    texto = anonimizar_texto(texto)

    return texto


def validar_registro(pergunta, resposta):
    if not pergunta:
        return False

    if not resposta:
        return False

    if len(pergunta) < 10:
        return False

    if len(resposta) < 10:
        return False

    return True


def carregar_dataset_qa():
    registros = []

    with open(INPUT_QA_DATASET, "r", encoding="utf-8") as file:
        for linha in file:
            if linha.strip():
                registros.append(json.loads(linha))

    return registros


def carregar_modelos_documentos():
    with open(INPUT_MODELOS_DOCUMENTOS, "r", encoding="utf-8") as file:
        return json.load(file)


def extrair_pergunta_qa(registro):
    entrada = registro.get("entrada", {})

    if isinstance(entrada, dict):
        return entrada.get("pergunta_usuario", "")

    return ""


def converter_qa_para_treinamento(registro):
    pergunta = preparar_texto(
        extrair_pergunta_qa(registro)
    )

    resposta = preparar_texto(
        registro.get("saida_esperada", "")
    )

    categoria = registro.get("area_clinica", "geral")
    protocolo = registro.get("protocolo_referencia", "")
    fonte = registro.get("fonte_simulada", "")
    nivel_risco = registro.get("nivel_risco", "")

    if not validar_registro(pergunta, resposta):
        return None

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é um assistente acadêmico especializado "
                    "em apoio clínico. Responda com segurança, "
                    "sem prescrever medicamentos e sem substituir "
                    "avaliação de profissional de saúde."
                )
            },
            {
                "role": "user",
                "content": pergunta
            },
            {
                "role": "assistant",
                "content": resposta
            }
        ],
        "metadata": {
            "origem": "qa_medico_sintetico",
            "area_clinica": categoria,
            "protocolo_referencia": protocolo,
            "fonte_simulada": fonte,
            "nivel_risco": nivel_risco
        }
    }


def converter_modelo_documento_para_treinamento(registro):
    tipo_documento = registro.get("tipo_documento", "documento")
    titulo = registro.get("titulo", "")
    descricao = registro.get("descricao", "")
    texto_para_rag = registro.get("texto_para_rag", "")
    fonte = registro.get("fonte_simulada", "")
    id_documento = registro.get("id_documento_modelo", "")

    campos_sugeridos = registro.get("campos_sugeridos", [])
    limites_de_uso = registro.get("limites_de_uso", [])

    pergunta = preparar_texto(
        f"Organize um modelo interno do tipo {tipo_documento}: {titulo}."
    )

    resposta = preparar_texto(
        f"""
Modelo interno: {titulo}
Tipo de documento: {tipo_documento}
Descrição: {descricao}

Campos sugeridos:
{", ".join(campos_sugeridos)}

Orientações e limites de uso:
{", ".join(limites_de_uso)}

Texto de referência:
{texto_para_rag}

Fonte:
{fonte}

Observação:
Este conteúdo é acadêmico, fictício e exige revisão humana antes de qualquer uso clínico.
"""
    )

    if not validar_registro(pergunta, resposta):
        return None

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é um assistente acadêmico especializado "
                    "em organização de documentos clínicos internos. "
                    "Não assine documentos, não prescreva medicamentos "
                    "e sempre exija revisão humana."
                )
            },
            {
                "role": "user",
                "content": pergunta
            },
            {
                "role": "assistant",
                "content": resposta
            }
        ],
        "metadata": {
            "origem": "modelo_documento_interno",
            "id_documento_modelo": id_documento,
            "tipo_documento": tipo_documento,
            "fonte_simulada": fonte
        }
    }


def gerar_dataset_formatado():
    registros_formatados = []

    registros_qa = carregar_dataset_qa()
    modelos_documentos = carregar_modelos_documentos()

    for registro in registros_qa:
        convertido = converter_qa_para_treinamento(registro)

        if convertido:
            registros_formatados.append(convertido)

    for registro in modelos_documentos:
        convertido = converter_modelo_documento_para_treinamento(registro)

        if convertido:
            registros_formatados.append(convertido)

    with open(OUTPUT_DATASET, "w", encoding="utf-8") as file:
        for item in registros_formatados:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
            )
            file.write("\n")

    print("Dataset formatado gerado em:")
    print(OUTPUT_DATASET)
    print()
    print("Registros QA carregados:")
    print(len(registros_qa))
    print()
    print("Modelos internos carregados:")
    print(len(modelos_documentos))
    print()
    print("Total de registros formatados:")
    print(len(registros_formatados))


if __name__ == "__main__":
    gerar_dataset_formatado()