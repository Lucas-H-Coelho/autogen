import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Tentativa de carregar .env
dotenv_path_from_cwd = os.path.join(os.getcwd(), '.env')
dotenv_loaded_successfully = False

if os.path.exists(dotenv_path_from_cwd):
    if load_dotenv(dotenv_path=dotenv_path_from_cwd, override=True):
        print(f"DEBUG (crewai_routes): .env carregado de CWD: {dotenv_path_from_cwd}")
        dotenv_loaded_successfully = True
    else:
        print(f"ALERTA (crewai_routes): Encontrado .env em {dotenv_path_from_cwd} mas falhou ao carregar.")
elif load_dotenv(override=True): # Tenta caminhos padrão do dotenv
    print("DEBUG (crewai_routes): .env carregado pelo dotenv usando pesquisa padrão (ex: diretório atual ou subindo).")
    dotenv_loaded_successfully = True # Assume sucesso se load_dotenv() retornar True
else:
    print("ALERTA (crewai_routes): Nenhum ficheiro .env encontrado ou carregado pelo dotenv automaticamente.")

if dotenv_loaded_successfully:
    print(f"DEBUG (crewai_routes): Após load_dotenv - GOOGLE_API_KEY presente: {'Sim' if os.getenv('GOOGLE_API_KEY') else 'Não'}")
    print(f"DEBUG (crewai_routes): Após load_dotenv - GEMINI_MODEL_NAME: {os.getenv('GEMINI_MODEL_NAME')}")
else:
    # Se não conseguiu carregar, pode ser um problema para a configuração do LLM
    print("ALERTA (crewai_routes): dotenv não conseguiu carregar um ficheiro .env. A execução dependerá de variáveis de ambiente já existentes.")

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
    # Se GEMINI_MODEL_NAME não estiver no .env ou ambiente, usa um default mais seguro
    gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash") 

    # Adiciona um log para ver o que foi efetivamente lido ou usado como default
    print(f"INFO (crewai_routes): A usar GEMINI_MODEL_NAME='{gemini_model_name}'")

    if not google_api_key:
        print("ERRO (crewai_routes): GOOGLE_API_KEY não foi encontrada no ambiente. Verifique o seu ficheiro .env e os logs de arranque do servidor.")
        # Retorna um erro claro se a chave não estiver aqui, antes de chamar o executor
        # que pode ser confuso se a chave estiver em falta.
        raise HTTPException(status_code=500, detail="Erro de Configuração: GOOGLE_API_KEY não encontrada no servidor. Verifique o ficheiro .env.")

    llm_config_from_router = {
        "api_key": google_api_key,
        "model_name": gemini_model_name
    }
    print(f"DEBUG (crewai_routes): Passando para CrewAIExecutor: model_name='{gemini_model_name}', api_key_presente={'Sim' if google_api_key else 'Não'}")

    try:
        executor = CrewAIExecutor(llm_config=llm_config_from_router)
        
        if not executor.llm:
            error_detail = "Erro: CrewAIExecutor não conseguiu inicializar o LLM."
            if not executor.google_api_key: # Esta verificação é redundante se a exceção acima for levantada
                error_detail += " GOOGLE_API_KEY não foi fornecida na configuração passada."
            else:
                error_detail += f" Falha ao inicializar com o modelo '{executor.gemini_model_name}'. Verifique a validade da chave API e do nome do modelo nos logs."
            # Não levanta HTTP Exception aqui, pois o executor já deve ter logado o problema específico
            # Apenas retorna a mensagem de erro que o executor teria preparado (ou uma genérica)
            return CrewTaskResponse(result="", error=error_detail)

        result = executor.run_task(request_data.task_description)
        
        print(f"DEBUG (crewai_routes): Resultado do CrewAIExecutor: {result[:200]}...")
        if result.startswith("Erro:"):
            return CrewTaskResponse(result="", error=result) 
        return CrewTaskResponse(result=result, error=None)

    except HTTPException as http_exc: 
        raise http_exc # Re-levanta exceções HTTP para que o FastAPI as trate
    except Exception as e:
        print(f"ERRO (crewai_routes): Exceção inesperada ao chamar CrewAIExecutor: {e}")
        # Captura outras exceções para retornar uma resposta 500 mais informativa
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor ao processar tarefa CrewAI: {str(e)}")
