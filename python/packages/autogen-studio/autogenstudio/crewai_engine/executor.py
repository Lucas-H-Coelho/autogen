import os
from crewai import Agent, Task, Crew, Process
# Atualizado para importar o wrapper que usa google.generativeai diretamente
from .custom_gemini_llm import CustomGoogleGeminiLLM, CustomGoogleGeminiError 
import traceback

class CrewAIExecutor:
    def __init__(self, llm_config: dict = None):
        self.google_api_key = None
        # GEMINI_MODEL_NAME no .env deve ser o nome base, ex: "gemini-1.5-flash"
        self.model_name = "gemini-1.5-flash" # Default
        self.project_id = None # Adicionado para guardar o project_id
        self.llm = None

        if llm_config:
            self.google_api_key = llm_config.get("api_key")
            self.model_name = llm_config.get("model_name", self.model_name)
            self.project_id = llm_config.get("project_id") # Guardar project_id
        else:
            # Fallback para teste local do executor.py
            print("AVISO (CrewAIExecutor): llm_config não fornecido. Tentando carregar do ambiente.")
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            self.model_name = os.getenv("GEMINI_MODEL_NAME", self.model_name)
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")

        print(f"DEBUG (CrewAIExecutor): __init__ - Chave API: {'Sim' if self.google_api_key else 'Não'}, Modelo: '{self.model_name}', Projeto: '{self.project_id}'")

        if not self.google_api_key:
            print("ERRO (CrewAIExecutor): GOOGLE_API_KEY não configurada.")
            return
        if not self.model_name:
            print("ERRO (CrewAIExecutor): Nome do modelo (GEMINI_MODEL_NAME) não configurado.")
            return
        # O project_id é opcional para a API Gemini direta, mas bom para logar.

        try:
            self.llm = CustomGoogleGeminiLLM(
                model_name=self.model_name, 
                google_api_key=self.google_api_key,
                project_id=self.project_id # Passa para o custom LLM (embora ele possa não usar para API Gemini direta)
            )
            print(f"DEBUG (CrewAIExecutor): CustomGoogleGeminiLLM (direto google.generativeai) inicializado.")
        except CustomGoogleGeminiError as e_custom:
            print(f"ERRO (CrewAIExecutor): Falha ao inicializar CustomGoogleGeminiLLM: {e_custom}")
            self.llm = None
        except Exception as e:
            print(f"ERRO INESPERADO (CrewAIExecutor): Falha ao inicializar CustomGoogleGeminiLLM: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            self.llm = None

    def run_task(self, task_description: str) -> str:
        if not self.llm:
            return f"Erro: LLM (CustomGoogleGeminiLLM) não inicializado. Verifique a config e logs."

        try:
            researcher = Agent(
                role='Investigador de IA',
                goal=f'Conduzir pesquisa sobre {task_description}',
                backstory='Investigador experiente.',
                llm=self.llm,
                verbose=True,
                allow_delegation=False
            )
            task = Task(description=f'Resumo sobre {task_description}', expected_output='Resumo conciso.', agent=researcher)
            crew = Crew(agents=[researcher], tasks=[task], verbose=True)
            
            print(f"DEBUG (CrewAIExecutor): A executar crew.kickoff() com CustomGoogleGeminiLLM...")
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            print(f"ERRO (CrewAIExecutor) na execução da tarefa: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            return f"Erro ao executar tarefa CrewAI: {type(e).__name__}: {str(e)}"

if __name__ == '__main__':
    print("--- Teste local do CrewAIExecutor com CustomGoogleGeminiLLM (direto) ---")
    from dotenv import load_dotenv
    if load_dotenv(override=True):
        print("INFO (Executor Teste): .env carregado.")
    else:
        print("AVISO (Executor Teste): .env não encontrado.")

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash") # Espera nome base
    project = os.getenv("GOOGLE_CLOUD_PROJECT_ID")

    if not api_key or not model_name:
        print("ERRO (Executor Teste): GOOGLE_API_KEY ou GEMINI_MODEL_NAME não definidos.")
    else:
        print(f"INFO (Executor Teste): API Key: Sim, Modelo: {model_name}, Projeto: {project}")
        executor = CrewAIExecutor(llm_config={"api_key": api_key, "model_name": model_name, "project_id": project})
        if executor.llm:
            output = executor.run_task("os desafios éticos da IA generativa")
            print("--- Resultado Teste Local ---")
            print(output)
        else:
            print("ERRO (Executor Teste): LLM não inicializado.")
