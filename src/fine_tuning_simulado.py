import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DATASET = (
    BASE_DIR
    / "data"
    / "dataset_medico_qa_sintetico.jsonl"
)

OUTPUT_DATASET = (
    BASE_DIR
    / "data"
    / "dataset_finetuning_formatado.jsonl"
)


def carregar_dataset():
    registros = []

    with open(INPUT_DATASET, "r", encoding="utf-8") as file:
        for linha in file:
            if linha.strip():
                registros.append(json.loads(linha))

    return registros


def extrair_pergunta(registro):
    entrada = registro.get("entrada", {})

    if isinstance(entrada, dict):
        return entrada.get("pergunta_usuario", "")

    return ""


def converter_para_formato_treinamento(registro):
    pergunta = extrair_pergunta(registro)

    resposta = registro.get("saida_esperada", "")

    categoria = registro.get("area_clinica", "geral")

    protocolo = registro.get("protocolo_referencia", "")

    fonte = registro.get("fonte_simulada", "")

    nivel_risco = registro.get("nivel_risco", "")

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
            "area_clinica": categoria,
            "protocolo_referencia": protocolo,
            "fonte_simulada": fonte,
            "nivel_risco": nivel_risco
        }
    }


def gerar_dataset_formatado():
    dataset_original = carregar_dataset()

    dataset_convertido = []

    for registro in dataset_original:
        convertido = converter_para_formato_treinamento(registro)

        pergunta = convertido["messages"][1]["content"]
        resposta = convertido["messages"][2]["content"]

        if not pergunta or not resposta:
            continue

        dataset_convertido.append(convertido)

    with open(OUTPUT_DATASET, "w", encoding="utf-8") as file:
        for item in dataset_convertido:
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
    print("Total de registros originais:")
    print(len(dataset_original))
    print()
    print("Total de registros convertidos:")
    print(len(dataset_convertido))


if __name__ == "__main__":
    gerar_dataset_formatado()