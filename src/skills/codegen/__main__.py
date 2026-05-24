"""Punto de entrada para lanzar el server codegen como módulo: ``python -m src.skills.codegen``."""
import asyncio

from .server import main

if __name__ == "__main__":
    asyncio.run(main())
