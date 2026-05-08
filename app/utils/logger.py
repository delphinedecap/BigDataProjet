import logging
from pathlib import Path


def setup_logger(log_file: str = "runs/app.log") -> logging.Logger:
    """
    Initialise un logger qui écrit dans un fichier et dans la console.

    La fonction réinitialise les handlers à chaque appel pour permettre
    d'exécuter plusieurs runs dans un même processus avec des fichiers
    de logs différents.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("eloquent")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger