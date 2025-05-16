import os
from crewai import Agent, Task, Crew, Process
from .custom_gemini_llm import CustomGoogleGeminiLLM, CustomGoogleGeminiError # Importa o wrapper customizado
import traceback

class CrewAIExecutor:
    def __init__(self, llm_config: dict = None):
        self.google_api_key = None
        # GEMINI_MODEL_NAME no .env deve ser o nome base, ex: "gemini-1.5-flash"
        self.model_name = "gemini-1.5-flash" # Default
        self.llm = None

        if llm_config:
            self.google_api_key = llm_config.get("api_key")
            self.model_name = llm_config.get("model_name", self.model_name)
        else:
            print("AVISO (CrewAIExecutor): llm_config não fornecido. Tentando carregar do ambiente (para teste local).")
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            self.model_name = os.getenv("GEMINI_MODEL_NAME", self.model_name)

        print(f"DEBUG (CrewAIExecutor): __init__ - Chave API presente: {'Sim' if self.google_api_key else 'Não'}")
        print(f"DEBUG (CrewAIExecutor): __init__ - Nome do modelo base a usar: '{self.model_name}'")

        if not self.google_api_key:
            print("ERRO (CrewAIExecutor): GOOGLE_API_KEY não configurada.")
            return
        if not self.model_name:
            print("ERRO (CrewAIExecutor): Nome do modelo (GEMINI_MODEL_NAME) não configurado.")
            return

        try:
            self.llm = CustomGoogleGeminiLLM(
                model_name=self.model_name, 
                google_api_key=self.google_api_key
                # Pode adicionar outros parâmetros como temperature aqui se o wrapper os aceitar no __init__
            )
            print(f"DEBUG (CrewAIExecutor): CustomGoogleGeminiLLM inicializado com sucesso. Modelo: '{self.model_name}'")
        except CustomGoogleGeminiError as e_custom:
            print(f"ERRO (CrewAIExecutor): Falha ao inicializar CustomGoogleGeminiLLM. Erro: {e_custom}")
            # O traceback já foi impresso dentro do CustomGoogleGeminiLLM
            self.llm = None
        except Exception as e:
            print(f"ERRO INESPERADO (CrewAIExecutor): Falha ao inicializar CustomGoogleGeminiLLM. Modelo: '{self.model_name}'. Erro: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            self.llm = None

    def run_task(self, task_description: str) -> str:
        if not self.llm:
            return f"Erro: LLM (CustomGoogleGeminiLLM) não inicializado. Verifique a configuração (modelo: {self.model_name}, API Key fornecida: {'Sim' if self.google_api_key else 'Não'}) e os logs do servidor."

        try:
            # Agentes e Tarefas simplificados para focar na chamada LLM
            researcher = Agent(
                role='Investigador de IA',
                goal=f'Conduzir uma pesquisa aprofundada sobre {task_description}',
                backstory='Você é um investigador de IA de renome com um olhar atento para os detalhes e uma capacidade de encontrar informação relevante rapidamente.',
                llm=self.llm,
                verbose=True,
                allow_delegation=False # Simplificando
            )
            task = Task(
                description=f'Criar um breve resumo sobre {task_description}. O resumo deve ser conciso e informativo.',
                expected_output='Um resumo de 2-3 parágrafos sobre o tópico.',
                agent=researcher
            )
            crew = Crew(agents=[researcher], tasks=[task], verbose=True)
            
            print(f"DEBUG (CrewAIExecutor): A executar crew.kickoff() com CustomGoogleGeminiLLM...")
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            print(f"ERRO (CrewAIExecutor) na execução da tarefa com CustomGoogleGeminiLLM: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            return f"Erro ao executar tarefa CrewAI: {type(e).__name__}: {str(e)}"

if __name__ == '__main__':
    # Para teste local do executor com o wrapper customizado
    print("--- Teste local do CrewAIExecutor com CustomGoogleGeminiLLM ---")
    from dotenv import load_dotenv
    if load_dotenv(override=True):
        print("INFO (Executor Teste Local): .env carregado.")
    else:
        print("AVISO (Executor Teste Local): .env não encontrado.")

    api_key = os.getenv("GOOGLE_API_KEY")
    # GEMINI_MODEL_NAME no .env deve ser o nome base, ex: "gemini-1.5-flash"
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash") 

    if not api_key or not model_name:
        print("ERRO (Executor Teste Local): GOOGLE_API_KEY ou GEMINI_MODEL_NAME não definidos.")
    else:
        print(f"INFO (Executor Teste Local): API Key: Sim, Modelo: {model_name}")
        executor = CrewAIExecutor(llm_config={"api_key": api_key, "model_name": model_name})
        if executor.llm:
            output = executor.run_task("os desafios éticos da inteligência artificial generativa")
            print("--- Resultado Teste Local (Custom LLM) ---")
            print(output)
        else:
            print("ERRO (Executor Teste Local): LLM não inicializado no executor.")
