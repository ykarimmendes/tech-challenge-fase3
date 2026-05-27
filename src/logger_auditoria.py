import json
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "logs_auditoria.jsonl"


def registrar_log(dados: dict) -> None:
    """
    Registra cada execução em arquivo JSONL para auditoria.
    """

    LOG_DIR.mkdir(exist_ok=True)

    registro = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **dados,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")