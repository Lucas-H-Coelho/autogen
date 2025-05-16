import os
import google.generativeai as genai
from typing import Any, List, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
import traceback

class CustomGoogleGeminiError(Exception):
    pass

class CustomGoogleGeminiLLM(LLM):
    model_name: str = "gemini-1.5-flash"
    google_api_key: Optional[str] = None
    client: Optional[genai.GenerativeModel] = None
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    # project_id: Optional[str] = None # Não usado diretamente por genai.GenerativeModel para API Gemini direta

    def __init__(self, model_name: str, google_api_key: str, project_id: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.google_api_key = google_api_key
        # self.project_id = project_id # Guardar se quisermos logar ou usar para debug

        print(f"DEBUG (CustomGoogleGeminiLLM): __init__ - Modelo: '{self.model_name}', API Key Fornecida: {'Sim' if self.google_api_key else 'Não'}, Project ID Recebido: '{project_id}'")

        if not self.google_api_key:
            raise CustomGoogleGeminiError("GOOGLE_API_KEY não fornecida.")
        try:
            genai.configure(api_key=self.google_api_key)
            # Para a API Gemini direta (não Vertex), o project_id não é usado aqui.
            self.client = genai.GenerativeModel(self.model_name)
            print(f"DEBUG (CustomGoogleGeminiLLM): Cliente Gemini (google.generativeai) inicializado com sucesso para o modelo: {self.model_name}")
        except Exception as e:
            print(f"ERRO (CustomGoogleGeminiLLM): Falha ao configurar ou inicializar o cliente Gemini (google.generativeai). Modelo: {self.model_name}. Erro: {e}")
            print(traceback.format_exc())
            raise CustomGoogleGeminiError(f"Falha na inicialização do cliente Gemini: {e}") from e

    @property
    def _llm_type(self) -> str:
        return "custom_google_gemini_direct_api"

    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> str:
        if self.client is None:
            raise ValueError("Cliente Gemini (google.generativeai) não inicializado.")

        generation_config_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }
        generation_config_dict = {k: v for k, v in generation_config_params.items() if v is not None}
        final_generation_config = genai.types.GenerationConfig(**{**generation_config_dict, **kwargs})

        print(f"DEBUG (CustomGoogleGeminiLLM): Enviando prompt para Gemini (google.generativeai): '{prompt[:100]}...'")
        print(f"DEBUG (CustomGoogleGeminiLLM): Usando config de geração: {final_generation_config}")

        try:
            response = self.client.generate_content(
                prompt, 
                generation_config=final_generation_config
            )
            
            if response.parts:
                generated_text = response.text
                print(f"DEBUG (CustomGoogleGeminiLLM): Texto gerado: '{generated_text[:100]}...'")
                return generated_text
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                block_message = getattr(response.prompt_feedback, 'block_reason_message', "Sem mensagem adicional.")
                print(f"ERRO (CustomGoogleGeminiLLM): Geração bloqueada. Razão: {block_reason}, Mensagem: {block_message}")
                raise ValueError(f"Geração bloqueada pela API Gemini. Razão: {block_reason}. Mensagem: {block_message}")
            else:
                print(f"ERRO (CustomGoogleGeminiLLM): Resposta do Gemini (google.generativeai) não continha partes de texto.")
                print(f"DEBUG (CustomGoogleGeminiLLM): Resposta completa: {response}")
                raise ValueError("Resposta vazia ou inesperada da API Gemini (google.generativeai).")
        except Exception as e:
            print(f"ERRO (CustomGoogleGeminiLLM): Erro durante generate_content: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            raise
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature, # ... etc.
               }

if __name__ == '__main__':
    print("--- Teste Local do CustomGoogleGeminiLLM (direto google.generativeai) ---")
    from dotenv import load_dotenv
    if load_dotenv(override=True):
        print("INFO (Custom LLM Teste): .env carregado.")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        gemini_model = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash") # Espera nome base
        project = os.getenv("GOOGLE_CLOUD_PROJECT_ID") # Carrega para logar
        if google_api_key and gemini_model:
            print(f"INFO (Custom LLM Teste): Usando API Key (presente), Modelo: {gemini_model}, Projeto (para log): {project}")
            try:
                llm = CustomGoogleGeminiLLM(model_name=gemini_model, google_api_key=google_api_key, project_id=project)
                response = llm.invoke("Olá, como você está?")
                print(f"Resposta do LLM: {response}")
            except Exception as e:
                print(f"Erro no teste: {e}")
                print(traceback.format_exc())
        else:
            print("ERRO (Custom LLM Teste): GOOGLE_API_KEY ou GEMINI_MODEL_NAME não encontrados no .env")
    else:
        print("AVISO (Custom LLM Teste): .env não encontrado.")
