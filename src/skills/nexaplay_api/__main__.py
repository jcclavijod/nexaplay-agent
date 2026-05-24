"""Punto de entrada para lanzar el server nexaplay_api como módulo: ``python -m src.skills.nexaplay_api``."""
from .server import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
