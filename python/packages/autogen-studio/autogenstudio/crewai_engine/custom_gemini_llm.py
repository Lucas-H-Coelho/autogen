import os
import google.generativeai as genai
from typing import Any, List, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
import traceback # Adicionado para traceback no teste local

class CustomGoogleGeminiError(Exception):
    """Custom exception for errors in CustomGoogleGeminiLLM."""
    pass

class CustomGoogleGeminiLLM(LLM):
    """Custom LangChain LLM wrapper for Google Gemini using google-generativeai."""
    model_name: str = "gemini-1.5-flash"
    google_api_key: Optional[str] = None
    client: Optional[genai.GenerativeModel] = None
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None

    def __init__(self, model_name: str, google_api_key: str, **kwargs: Any):
        super().__init__(**kwargs) # Passa quaisquer kwargs para a classe base LLM
        self.model_name = model_name
        self.google_api_key = google_api_key
        try:
            genai.configure(api_key=self.google_api_key)
            self.client = genai.GenerativeModel(self.model_name)
            print(f"DEBUG (CustomGoogleGeminiLLM): Cliente Gemini inicializado com sucesso para o modelo: {self.model_name}")
        except Exception as e:
            print(f"ERRO (CustomGoogleGeminiLLM): Falha ao configurar ou inicializar o cliente Gemini. Modelo: {self.model_name}. Erro: {e}")
            raise CustomGoogleGeminiError(f"Falha na inicialização do cliente Gemini: {e}") from e

    @property
    def _llm_type(self) -> str:
        return "custom_google_gemini"

    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None, # <--- CORRIGIDO AQUI
        **kwargs: Any
    ) -> str:
        if self.client is None:
            raise ValueError("Cliente Gemini não inicializado. Verifique a API Key e o nome do modelo.")

        generation_config_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }
        generation_config_dict = {k: v for k, v in generation_config_params.items() if v is not None}
        
        final_generation_config = genai.types.GenerationConfig(**{**generation_config_dict, **kwargs}) # Combina configs

        print(f"DEBUG (CustomGoogleGeminiLLM): A enviar prompt para Gemini: '{prompt[:100]}...'")
        print(f"DEBUG (CustomGoogleGeminiLLM): Usando config de geração: {final_generation_config}")

        try:
            response = self.client.generate_content(
                prompt, 
                generation_config=final_generation_config,
                # O SDK do Gemini trata stop sequences dentro do generation_config ou através de um argumento separado,
                # dependendo da versão. Para este wrapper, vamos assumir que `stop` é tratado por LangChain/CrewAI 
                # ou que não é criticamente necessário para o primeiro teste.
                # Se `stop` for necessário, precisaria ser mapeado para `stop_sequences` em `GenerationConfig`.
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
                print("ERRO (CustomGoogleGeminiLLM): Resposta do Gemini não continha partes de texto ou feedback de bloqueio claro.")
                try:
                    print(f"DEBUG (CustomGoogleGeminiLLM): Resposta completa do Gemini: {response}")
                except Exception as print_e:
                    print(f"DEBUG (CustomGoogleGeminiLLM): Não foi possível imprimir a resposta completa do Gemini: {print_e}")
                raise ValueError("Resposta vazia ou inesperada da API Gemini.")

        except Exception as e:
            print(f"ERRO (CustomGoogleGeminiLLM): Erro durante a chamada generate_content: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            raise
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }

if __name__ == '__main__':
    print("--- Teste Local do CustomGoogleGeminiLLM ---")
    from dotenv import load_dotenv
    if load_dotenv(override=True):
        print("INFO (Custom LLM Teste Local): .env carregado.")
    else:
        print("AVISO (Custom LLM Teste Local): .env não encontrado.")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash") 

    if not google_api_key:
        print("ERRO (Custom LLM Teste Local): GOOGLE_API_KEY não definida no ambiente.")
    else:
        print(f"INFO (Custom LLM Teste Local): A usar API Key (presente) e Modelo: {gemini_model}")
        try:
            llm = CustomGoogleGeminiLLM(model_name=gemini_model, google_api_key=google_api_key, temperature=0.5)
            prompt_text = "Descreva o ciclo da água em três frases."
            response_text = llm._call(prompt=prompt_text)
            print(f"Prompt: {prompt_text}")
            print(f"Resposta: {response_text}")
        except Exception as e:
            print(f"ERRO no teste local do CustomGoogleGeminiLLM: {e}")
            print(traceback.format_exc())
