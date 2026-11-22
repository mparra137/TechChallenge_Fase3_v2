# Agente médico conversacional · Tech Challenge FIAP, Fase 3

O projeto reúne duas tarefas em uma conversa de terminal. O Llama 3.1 8B Instruct ajustado com QLoRA responde a perguntas biomédicas gerais na GPU local. Para perguntas sobre um paciente, o agente consulta um acervo hospitalar sintético e apresenta os documentos que sustentam a resposta. O LangGraph coordena o roteamento e a memória da sessão. Os componentes LangChain integram os modelos, os documentos e o recuperador.

Por exemplo, pergunte “Quais são os últimos 3 exames da paciente Ana Lopes Cunha?” e depois “Qual a última anamnese?”. O agente mantém Ana como paciente ativa e aponta a anamnese `ATD-PCT-000086-03` como fonte da segunda resposta.

O ajuste do Llama usou perguntas do MedQuAD, evidências do PubMedQA e protocolos hospitalares fictícios. Os prontuários consultados durante a conversa formam uma base separada. Para experimentar o agente, bastam o adaptador publicado e a base RAG incluída no repositório. O projeto é acadêmico e os resultados clínicos são fictícios.

## Instalação

Execute os blocos abaixo no **PowerShell**. O ambiente de referência utiliza **Python 3.12.10**, GPU NVIDIA com driver compatível com CUDA 12.8 e PyTorch 2.10.0. O Llama é carregado em 4 bits na inicialização da CLI. Os embeddings do RAG são calculados na CPU.

### 1. Clonar e criar o ambiente

```powershell
git clone https://github.com/mparra137/TechChallenge_Fase3_v2.git
cd TechChallenge_Fase3_v2
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar as dependências e verificar a GPU

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/verify_torch.py
```

Confira se a saída mostra `CUDA disponível: True`, o nome da GPU e a operação de teste em `cuda:0`. Os comandos seguintes pressupõem a raiz do clone e a `.venv` ativa.

### 3. Configurar credenciais e baixar os pesos

Em um clone novo, execute `Copy-Item .env.example .env`. Se já tiver um `.env`, edite o arquivo existente. Preencha as credenciais na raiz do projeto e mantenha seus valores fora do Git.

```dotenv
HUGGINGFACE_API_KEY=seu_token_de_leitura_do_hugging_face
OPENAI_API_KEY=sua_chave_da_openai
```

- **Hugging Face.** Configure `HUGGINGFACE_API_KEY` para obter o adaptador oficial e o modelo-base com acesso autenticado. O pipeline de fine-tuning usa a mesma credencial para baixar MedQuAD e PubMedQA quando você optar por reproduzir o treinamento. [Adaptador mparra137/llama-lora-medical-tc3-300](https://huggingface.co/mparra137/llama-lora-medical-tc3-300)
- **OpenAI.** Configure `OPENAI_API_KEY` para a síntese e a interpretação de documentos do paciente. A conta precisa ter acesso ao modelo configurado, por padrão `gpt-4o-mini`. O nome `OPEN_AI_API_KEY` do arquivo de exemplo também é aceito. [Configuração do projeto](src/config.py)

Na primeira instalação, execute o bloco abaixo com acesso à internet. Ele lê a credencial do projeto, baixa o adaptador e identifica o modelo-base pelo arquivo de configuração PEFT. Execute a CLI com o mesmo usuário e cache. Com `MEDICAL_LOCAL_FILES_ONLY=True`, o agente carrega esses arquivos do disco e faz a inferência localmente.

```powershell
@'
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from src import config

adapter = snapshot_download(config.MEDICAL_ADAPTER_ID, token=config.HUGGINGFACE_API_KEY)
metadata = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
snapshot_download(metadata["base_model_name_or_path"], token=config.HUGGINGFACE_API_KEY)
'@ | python -
```

### 4. Preparar a busca nos prontuários

Os JSONL de 200 pacientes e 1.787 documentos já acompanham o clone. A primeira ingestão pode baixar o MiniLM multilíngue e cria o índice local em `database/chroma/`.

```powershell
python scripts/hospital_rag.py ingest
python scripts/hospital_rag.py stats
```

Confira `vector_count` nas estatísticas. Para a base distribuída, o valor esperado é **1787**, um vetor por documento.

### 5. Abrir a conversa

```powershell
python -m scripts.medical_agent
```

Quando aparecer `Você>`, experimente estas perguntas em sequência:

```text
Você> Quais são os últimos 3 exames da paciente Ana Lopes Cunha?
Você> Qual a última anamnese?
Você> O que é hemoglobina glicada?
Você> Qual o resultado do exame de ultrassom abdominal?
Você> Quero saber o cadastro da paciente Paula Porto.
```

A segunda pergunta usa o prontuário de Ana. A terceira segue para o Llama local e a quarta volta ao prontuário de Ana. A última seleciona Paula para as próximas consultas. Também é possível iniciar a busca com um código como `PCT-000086`. Diante de um nome ambíguo ou desconhecido, o agente pede esclarecimento.

## Como o agente responde

| Pedido | Execução | Como ler a resposta |
|---|---|---|
| Pergunta médica geral (`GENERAL_MEDICAL`) | O Llama local recebe a pergunta e o histórico geral recente. | As referências e URLs que ele gerar aparecem no texto. Essa rota não verifica a bibliografia. |
| Consulta exata ao prontuário (`PATIENT_RECORD`) | O agente identifica o paciente. Consultas estruturadas selecionam, por exemplo, os três exames mais recentes ou a última anamnese por data. | A resposta local inclui os registros encontrados e suas fontes. |
| Síntese documental (`PATIENT_RECORD`) | O agente recupera evidências do paciente e envia à OpenAI a pergunta e os trechos mascarados, com identificadores temporários. | O agente confere os identificadores citados e exibe ID, data e origem dos documentos correspondentes. Se a evidência for insuficiente ou a síntese falhar na validação, a CLI informa o motivo geral. |

A memória de conversa vive em RAM. O paciente continua ativo para as consultas documentais até uma troca de identidade, um pedido de esclarecimento ou a limpeza da sessão. O filtro por paciente é demonstrativo. Use apenas a base sintética para os exercícios.

| Comando | Efeito |
|---|---|
| `/ajuda` | Lista os comandos disponíveis. |
| `/limpar` | Apaga o histórico e o paciente ativo e mantém a sessão atual. |
| `/nova` | Descarta o contexto anterior e abre uma sessão com outro identificador. |
| `/sair` | Encerra a CLI e descarta a memória. `Ctrl+C` ou fim da entrada também encerram a conversa. |

## Diagnóstico e testes

| Sintoma | O que conferir |
|---|---|
| A CLI encerra com falha ao carregar o Llama | Execute `python scripts/verify_torch.py`. Confira a distribuição CUDA do PyTorch, a memória disponível e os dois downloads do passo 3. |
| A CLI abre e informa que os prontuários estão indisponíveis | Execute `ingest` e `stats` do passo 4. Confira o MiniLM em cache e a coleção `hospital_patients`. Perguntas gerais continuam disponíveis. |
| Consultas exatas funcionam, mas a síntese documental está indisponível | Confira a chave OpenAI, o acesso da conta ao modelo e os avisos no terminal. Uma resposta fora do esquema ou com fontes inválidas também impede a exibição da síntese. |

Execute `python -m pytest -q` para verificar o projeto. Para acompanhar tempos e rotas, use `python -m scripts.medical_agent --log-level INFO`.

## Guias do projeto

- [Fine-tuning do Llama](README_finetuning.md) — fontes, pré-processamento, notebook no Kaggle e resultados dos 300 passos.
- [Serviço RAG](README_RAG.md) — ingestão, consultas estruturadas e vetoriais, síntese e citações.
- [Base hospitalar sintética](README_RAG_DATA.md) — arquivos versionados, geração reprodutível e validação.
