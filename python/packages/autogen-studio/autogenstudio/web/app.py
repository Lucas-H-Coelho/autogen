# api/app.py
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from ..version import VERSION
from .auth import authroutes
from .auth.middleware import AuthMiddleware
from .config import settings
from .deps import cleanup_managers, init_auth_manager, init_managers, register_auth_dependencies
from .initialization import AppInitializer
from .routes import gallery, runs, sessions, settingsroute, teams, validation, ws

# Tentativa de importar crewai_routes e debug
try:
    from .routes import crewai_routes
    print("DEBUG: crewai_routes importado com SUCESSO.")
except ImportError as e:
    print(f"DEBUG: FALHA ao importar crewai_routes: {e}")
    crewai_routes = None # Define como None se a importação falhar

# Initialize application
app_file_path = os.path.dirname(os.path.abspath(__file__))
initializer = AppInitializer(settings, app_file_path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Iniciando lifespan do servidor...")
    try:
        await init_managers(initializer.database_uri, initializer.config_dir, initializer.app_root)
        await register_auth_dependencies(app, auth_manager)
        logger.info(
            f"Application startup complete. Navigate to http://{os.environ.get('AUTOGENSTUDIO_HOST', '127.0.0.1')}:{os.environ.get('AUTOGENSTUDIO_PORT', '8081')}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    yield
    logger.info("Iniciando cleanup do servidor...")
    try:
        await cleanup_managers()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


auth_manager = init_auth_manager(initializer.config_dir)
app = FastAPI(lifespan=lifespan, debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8001", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware, auth_manager=auth_manager)

api = FastAPI(
    root_path="/api",
    title="AutoGen Studio API",
    version=VERSION,
    description="AutoGen Studio is a low-code tool for building and testing multi-agent workflows.",
    docs_url="/docs" if settings.API_DOCS else None,
)

# Endpoint de teste simples diretamente no app.py
@api.get("/test-debug")
async def test_debug_endpoint():
    print("DEBUG: Endpoint /api/test-debug ATINGIDO!")
    return {"message": "Debug endpoint no app.py funciona!"}

print("DEBUG: A registar routers existentes...")
api.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
api.include_router(runs.router, prefix="/runs", tags=["runs"])
api.include_router(teams.router, prefix="/teams", tags=["teams"])
api.include_router(ws.router, prefix="/ws", tags=["websocket"])
api.include_router(validation.router, prefix="/validate", tags=["validation"])
api.include_router(settingsroute.router, prefix="/settings", tags=["settings"])
api.include_router(gallery.router, prefix="/gallery", tags=["gallery"])
api.include_router(authroutes.router, prefix="/auth", tags=["auth"])
print("DEBUG: Routers existentes registados.")

if crewai_routes and hasattr(crewai_routes, 'router'):
    print(f"DEBUG: A tentar incluir crewai_routes.router com prefixo /crewai. Objecto Router: {crewai_routes.router}")
    api.include_router(
        crewai_routes.router,
        prefix="/crewai",
        tags=["CrewAI"],
        responses={404: {"description": "Not found"}}
    )
    print("DEBUG: crewai_routes.router INCLUÍDO.")
elif crewai_routes:
    print(f"DEBUG: crewai_routes foi importado, mas não tem um atributo 'router'. Conteúdo: {dir(crewai_routes)}")
else:
    print("DEBUG: crewai_routes NÃO foi importado, não se pode incluir o router.")

@api.get("/version")
async def get_version():
    return {"status": True, "message": "Version retrieved successfully", "data": {"version": VERSION}}

@api.get("/health")
async def health_check():
    return {"status": True, "message": "Service is healthy"}

app.mount("/api", api)
app.mount("/files", StaticFiles(directory=initializer.static_root, html=True), name="files")
app.mount("/", StaticFiles(directory=initializer.ui_root, html=True), name="ui")

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {str(exc)}")
    return {"status": False, "message": "Internal server error", "detail": str(exc) if settings.API_DOCS else "Internal server error"}

def create_app() -> FastAPI:
    return app
