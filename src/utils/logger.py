"""Configuração central de logging dos pontos de entrada."""

import logging

from src.config import LOG_FILE_PATH, LOG_LEVEL


def configure_logging(console_level: int | str = LOG_LEVEL) -> None:
    """Grava registros a partir de INFO e configura o nível do console."""
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    console = logging.StreamHandler()
    console.setLevel(console_level)

    log_file = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    log_file.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console, log_file],
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
