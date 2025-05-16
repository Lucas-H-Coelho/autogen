import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

dotenv_path_from_cwd = os.path.join(os.getcwd(), '.env')
dotenv_loaded_successfully = False

if os.path.exists(dotenv_path_from_cwd):
    if load_dotenv(dotenv_path=dotenv_path_from_cwd, override=True):
        print(f"DEBUG (crewai_routes): .env carregado de CWD: {dotenv_path_from_cwd}")
        dotenv_loaded_successfully = True
    else:
        print(f"ALERTA (crewai_routes): Encontrado .env em {dotenv_path_from_cwd} mas falhou ao carregar.")
elif load_dotenv(override=True):
    print("DEBUG (crewai_routes): .env carregado pelo dotenv usando pesquisa padrão.")
    dotenv_loaded_successfully = True 
else:
    print("ALERTA (crewai_routes): Nenhum ficheiro .env encontrado ou carregado.")

if dotenv_loaded_successfully:
    print(f"DEBUG (crewai_routes): Após load_dotenv - GOOGLE_API_KEY presente: {'Sim' if os.getenv('GOOGLE_API_KEY') else 'Não'}")
    print(f"DEBUG (crewai_routes): Após load_dotenv - GEMINI_MODEL_NAME: {os.getenv('GEMINI_MODEL_NAME')}")
    print(f"DEBUG (crewai_routes): Após load_dotenv - GOOGLE_CLOUD_PROJECT_ID: {os.getenv('GOOGLE_CLOUD_PROJECT_ID')}") # Novo log
else:
    print("ALERTA (crewai_routes): dotenv não conseguiu carregar .env. A execução dependerá de vars de ambiente já existentes.")

from autogenstudio.crewai_engine.executor import CrewAIExecutor

router = APIRouter()

class CrewTaskRequest(BaseModel):
    task_description: str

class CrewTaskResponse(BaseModel):
    result: str
    error: str | None = None

@router.post("/run-task", response_model=CrewTaskResponse, tags=["CrewAI"])
async def run_crewai_task_endpoint(request_data: CrewTaskRequest):
    print("DEBUG (crewai_routes): Endpoint /api/crewai/run-task atingido.")
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash") 
    google_cloud_project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID") # Ler project_id

    print(f"INFO (crewai_routes): A usar GEMINI_MODEL_NAME='{gemini_model_name}', GOOGLE_CLOUD_PROJECT_ID='{google_cloud_project_id}'")

    if not google_api_key:
        print("ERRO (crewai_routes): GOOGLE_API_KEY não foi encontrada.")
        raise HTTPException(status_code=500, detail="Erro de Configuração: GOOGLE_API_KEY não encontrada.")
    
    # if not google_cloud_project_id: # Opcional: tornar obrigatório
    #     print("ERRO (crewai_routes): GOOGLE_CLOUD_PROJECT_ID não foi encontrado.")
    #     raise HTTPException(status_code=500, detail="Erro de Configuração: GOOGLE_CLOUD_PROJECT_ID não encontrado.")

    llm_config_from_router = {
        "api_key": google_api_key,
        "model_name": gemini_model_name,
        "project_id": google_cloud_project_id # Passar project_id
    }
    print(f"DEBUG (crewai_routes): Passando para CrewAIExecutor: {llm_config_from_router}")

    try:
        executor = CrewAIExecutor(llm_config=llm_config_from_router)
        
        if not executor.llm:
            error_detail = f"Erro: CrewAIExecutor não conseguiu inicializar o LLM. Modelo: '{executor.model_name_from_config if hasattr(executor, 'model_name_from_config') else 'N/A'}'. API Key fornecida: {'Sim' if executor.google_api_key else 'Não'}. Project ID: '{executor.project_id if hasattr(executor, 'project_id') else 'N/A'}'."
            return CrewTaskResponse(result="", error=error_detail)

        result = executor.run_task(request_data.task_description)
        
        print(f"DEBUG (crewai_routes): Resultado do CrewAIExecutor: {result[:200]}...")
        if result.startswith("Erro:"):
            return CrewTaskResponse(result="", error=result)
        return CrewTaskResponse(result=result, error=None)

    except HTTPException as http_exc: 
        raise http_exc
    except Exception as e:
        print(f"ERRO (crewai_routes): Exceção inesperada ao chamar CrewAIExecutor: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor ao processar tarefa CrewAI: {str(e)}")
