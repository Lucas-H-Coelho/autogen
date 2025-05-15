from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import asyncio

from ..crewai_engine.executor import CrewAIExecutor
# Se precisar de dependências de autenticação ou configurações do AutoGen Studio:
# from ..web.deps import get_current_user, User # Exemplo
# from ..utils.utils import get_app_settings # Exemplo

router = APIRouter()

class CrewTaskRequest(BaseModel):
    task_description: str
    # Pode adicionar mais campos aqui, como configurações de LLM específicas da tarefa
    # llm_api_key: str | None = None
    # llm_model_name: str | None = None
    # llm_api_base: str | None = None

class CrewTaskResponse(BaseModel):
    result: str
    error: str | None = None

@router.post("/crewai/run-task", response_model=CrewTaskResponse, tags=["CrewAI"])
async def run_crewai_task_endpoint(request_data: CrewTaskRequest):
    """
    Endpoint para executar uma tarefa usando o motor CrewAI.
    """
    try:
        # Aqui, pode querer ir buscar configurações de LLM do AutoGen Studio
        # ou permitir que sejam passadas na request_data.
        # Por agora, o executor tentará usar variáveis de ambiente se não for configurado.
        
        # llm_config = {}
        # if request_data.llm_api_key: llm_config["api_key"] = request_data.llm_api_key
        # if request_data.llm_model_name: llm_config["model_name"] = request_data.llm_model_name
        # if request_data.llm_api_base: llm_config["api_base"] = request_data.llm_api_base
        
        # executor = CrewAIExecutor(llm_config=llm_config if llm_config else None)
        executor = CrewAIExecutor() # Usará env vars ou defaults se não passar llm_config

        # CrewAI pode ser síncrono; execute-o num thread pool se for o caso
        # para não bloquear o loop de eventos do FastAPI.
        # No entanto, a função kickoff do CrewAI já é tipicamente assíncrona ou não bloqueadora
        # de forma significativa para o FastAPI se as operações internas de LLM forem assíncronas.
        # Se o seu `executor.run_task` for puramente síncrono e longo, use `run_in_threadpool`.
        # from fastapi.concurrency import run_in_threadpool
        # result = await run_in_threadpool(executor.run_task, request_data.task_description)
        
        # Assumindo que executor.run_task não é excessivamente bloqueador ou é assíncrono internamente:
        result = executor.run_task(request_data.task_description)

        if result.startswith("Erro:"):
            return CrewTaskResponse(result="", error=result)
        return CrewTaskResponse(result=result)

    except Exception as e:
        # logger.error(f"Erro no endpoint CrewAI: {e}")
        raise HTTPException(status_code=500, detail=str(e))
