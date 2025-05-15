import os
from crewai import Agent, Task, Crew, Process

# --- Configuração do LLM (Exemplo com OpenAI) ---
# Certifique-se de que a sua chave API está definida como uma variável de ambiente
# ou configure o LLM diretamente nos agentes.
# os.environ["OPENAI_API_KEY"] = "SUA_CHAVE_API_AQUI"
# os.environ["OPENAI_MODEL_NAME"] = "gpt-4" # ou gpt-3.5-turbo ou outro

# Se estiver a usar Ollama ou outro LLM local com LiteLLM, configure aqui:
# os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1" # Exemplo para Ollama via LiteLLM
# os.environ["OPENAI_MODEL_NAME"] = "ollama/mistral" # Exemplo para Ollama/Mistral

class CrewAIExecutor:
    def __init__(self, llm_config: dict = None):
        """
        Inicializa o executor.
        llm_config (opcional): Dicionário com configurações de LLM
                                (ex: api_key, model_name, api_base)
                                Se não fornecido, espera variáveis de ambiente.
        """
        if llm_config:
            if llm_config.get("api_key"):
                os.environ["OPENAI_API_KEY"] = llm_config["api_key"]
            if llm_config.get("model_name"):
                os.environ["OPENAI_MODEL_NAME"] = llm_config["model_name"]
            if llm_config.get("api_base"): # Para LLMs locais como Ollama via LiteLLM
                os.environ["OPENAI_API_BASE"] = llm_config["api_base"]


    def run_task(self, task_description: str) -> str:
        """
        Define e executa uma Crew simples para uma dada descrição de tarefa.
        """
        try:
            # Verificar se a chave API ou base da API está configurada se necessário
            if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_BASE"):
                return "Erro: Configuração do LLM não encontrada (OPENAI_API_KEY ou OPENAI_API_BASE)."

            planner = Agent(
                role='Planeador de Conteúdo Estratégico',
                goal=f'Planear um conteúdo envolvente e factualmente correto sobre {task_description}',
                backstory='Você é um planeador de conteúdo renomado, conhecido pela sua visão estratégica e atenção aos detalhes. Tem um talento especial para transformar ideias complexas em planos de conteúdo cativantes.',
                verbose=True,
                allow_delegation=False # Pode ser True se tiver mais agentes
            )

            writer = Agent(
                role='Escritor de Conteúdo Profissional',
                goal=f'Escrever um artigo perspicaz e envolvente sobre {task_description}, com base no plano fornecido.',
                backstory="""Você é um escritor de conteúdo excecional, famoso pela sua capacidade de criar narrativas cativantes e material bem pesquisado.
                Consegue transformar até os tópicos mais secos em leituras fascinantes.""",
                verbose=True,
                allow_delegation=False
            )

            plan_task = Task(
                description=f'1. Criar um plano de tópicos para um artigo sobre {task_description}. O plano deve cobrir os aspetos chave e garantir um fluxo lógico.
2. O plano deve ser acionável para um escritor.',
                expected_output=f'Um plano de conteúdo abrangente para um artigo sobre {task_description}, formatado como uma lista de tópicos ou secções principais.',
                agent=planner
            )

            write_task = Task(
                description=f'Escrever um artigo completo com base no plano de conteúdo fornecido para {task_description}. O artigo deve ser informativo, envolvente e bem estruturado.',
                expected_output=f'Um artigo bem escrito sobre {task_description}, com pelo menos 3 parágrafos.',
                agent=writer,
                context=[plan_task] # Passa o output da tarefa de planeamento para a tarefa de escrita
            )

            crew = Crew(
                agents=[planner, writer],
                tasks=[plan_task, write_task],
                process=Process.sequential,
                verbose=2
            )

            result = crew.kickoff()
            return str(result)

        except Exception as e:
            # logger.error(f"Erro ao executar a tarefa CrewAI: {e}")
            return f"Erro ao executar a tarefa CrewAI: {str(e)}"

if __name__ == '__main__':
    # Para testar este módulo isoladamente
    # Defina as suas variáveis de ambiente para OPENAI_API_KEY e OPENAI_MODEL_NAME
    # ou configure o llm_config no construtor.
    
    # Exemplo de configuração para Ollama (se estiver a correr localmente e LiteLLM estiver configurado):
    # config_ollama = {
    # "api_base": "http://localhost:11434/v1",
    # "model_name": "ollama/mistral" # ou o seu modelo Ollama preferido
    # }
    # executor = CrewAIExecutor(llm_config=config_ollama)
    
    # Exemplo para OpenAI (requer OPENAI_API_KEY e OPENAI_MODEL_NAME como env vars)
    if not os.getenv("OPENAI_API_KEY"):
        print("AVISO: OPENAI_API_KEY não está definida. O teste pode falhar ou usar defaults do LiteLLM se configurado.")

    executor = CrewAIExecutor()
    
    sample_task_desc = "o futuro da computação quântica e o seu impacto na criptografia"
    print(f"A executar tarefa CrewAI de teste: {sample_task_desc}")
    output = executor.run_task(sample_task_desc)
    print("
--- Resultado do Teste CrewAI Executor ---")
    print(output)
    print("--- Fim do Teste CrewAI Executor ---")
