# RAG Hospitalar Sintético

Este módulo implementa recuperação aumentada por geração sobre a base hospitalar artificial do projeto TCv2. Ele serve exclusivamente a testes acadêmicos e de engenharia; não é destinado a atendimento, diagnóstico, prescrição ou uso com dados reais.

## Arquitetura

```text
JSONL sintético → HospitalDocumentLoader → ChromaDB
                                      ↓
Pergunta → PatientResolver → RetrieverService → ContextBuilder → LLM → resposta + fontes
```

- `src/ingestion/document_loader.py`: leitura, validação relacional e conversão de cada documento JSONL em um `Document` LangChain.
- `src/embeddings/embedding_service.py`: instancia uma vez por processo o modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, em CPU e com vetores normalizados.
- `src/vectorstore/chroma_service.py`: persistência idempotente no Chroma, busca e estatísticas.
- `src/retrieval/patient_resolver.py`: busca determinística por código, nome completo ou trecho de nome inequívoco.
- `src/retrieval/retriever_service.py`: filtros por paciente/tipo e recuperação semântica.
- `src/rag/context_builder.py`: limita e identifica o contexto encaminhado ao LLM.
- `src/llm/llm_factory.py`: factory desacoplada do provedor; OpenAI é o provedor atual.
- `src/rag/rag_service.py`: coordena recuperação, contexto, geração e fontes estruturadas.

Os mesmos componentes de recuperação também atendem ao agente conversacional. O comando `query` e a síntese documental da CLI usam a geração OpenAI configurada. Na CLI do agente, o Llama local atende às perguntas médicas gerais e auxilia no roteamento.

## Agente local

Use Python 3.12 com PyTorch CUDA. No Windows, o perfil de referência é Torch 2.10.0 com CUDA 12.8. Em um ambiente novo, instale primeiro a distribuição CUDA e depois as dependências do projeto.

```powershell
& .\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m scripts.medical_agent
```

O agente espera o adaptador configurado, o respectivo modelo-base e o MiniLM no cache do Hugging Face. A inferência geral usa CUDA e a recuperação usa CPU. Para perguntas de síntese ou interpretação documental, configure `OPENAI_API_KEY` no ambiente ou em `.env` e o modelo em `src/config.py`. A API recebe a pergunta mascarada e os documentos sintéticos selecionados, com aliases temporários. A aplicação valida os aliases e resolve as fontes no ambiente local. A rota geral permanece disponível quando a OpenAI ou o índice está indisponível. As consultas exatas continuam disponíveis quando falta apenas o acesso à OpenAI.

- `/ajuda` lista os comandos.
- `/limpar` apaga o contexto e os checkpoints da conversa atual.
- `/nova` encerra a sessão anterior e cria uma conversa isolada.
- `/sair`, EOF ou `Ctrl+C` encerram a CLI.

Exemplo de conversa sintética: “Qual foi a última creatinina do paciente PCT-000001?” e “E a glicemia dele?”. A CLI confirma o nome selecionado, exibe fatos documentados e referencia as fontes por ID, data e origem. A memória vive em RAM. O modo `--log-level INFO` registra tempos e métricas técnicas de inferência. Consulte o [plano técnico do agente](docs/agent_impl.md) para detalhes de segurança e avaliação.

Pedidos diretos pela última anamnese exibem o documento mais recente. Pedidos como “Quais os exames realizados pelo paciente PCT-000086?” listam os exames laboratoriais e de imagem registrados para o paciente. “Quais os últimos 3 exames?” ordena os registros por data e aplica o limite solicitado. A resposta e as fontes podem mostrar o código do prontuário consultado. O mascaramento continua aplicado aos demais identificadores protegidos.

Após identificar um paciente, a CLI usa o mesmo prontuário nas consultas documentais seguintes que omitirem nome e código. Perguntas médicas gerais continuam no Llama local e preservam o paciente ativo. Uma nova identidade confirmada muda o prontuário. Identidades ambíguas ou não encontradas suspendem a seleção até o esclarecimento. Pedidos de resultado de ultrassom abdominal usam o laudo mais recente, ou o laudo da data informada.

Nas respostas médicas gerais, a CLI exibe as referências e os links gerados pelo Llama. A verificação de proveniência dos documentos recuperados se aplica às fontes do RAG.

## Configuração

Use `.env.example` como referência para colocar a chave OpenAI em `.env` (`OPEN_AI_API_KEY` ou `OPENAI_API_KEY`). A chave Hugging Face é opcional para o modelo público. Configure modelo e endpoint OpenAI, provedor, embeddings, coleção, caminhos, K, orçamento de contexto e log diretamente em `src/config.py`. Para a síntese na CLI, use um modelo habilitado na conta e compatível com [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Instalação

No PowerShell, com o ambiente virtual do projeto:

```powershell
& .\.venv\Scripts\python.exe -m pip install "langchain-openai==1.8.0" "langchain-core==1.8.0"
```

O primeiro uso dos embeddings pode baixar o modelo multilíngue. A execução de recuperação usa CPU para preservar a GPU do Llama.

O comando acima complementa a instalação existente do TCv2. Com o MiniLM completo em cache, `$env:HF_HUB_OFFLINE = '1'` no PowerShell permite reutilizá-lo localmente. Essa opção do Hugging Face vale para a sessão do terminal.

## Ingestão persistente

Os JSONL existentes não são regenerados. Cada documento clínico original é indexado como um vetor, usando `document_id` como identificador estável.

```powershell
& .\.venv\Scripts\python.exe scripts\hospital_rag.py ingest
& .\.venv\Scripts\python.exe scripts\hospital_rag.py stats
```

A base atual deve apresentar 200 pacientes, 1.787 documentos e 1.787 vetores após a primeira ingestão. Rodar a ingestão novamente mantém a coleção idempotente: documentos inalterados são ignorados e documentos alterados são atualizados. O índice é persistido em `database/chroma/`, que é ignorado pelo Git.

## Consultas

Consulta completa, com recuperação e resposta OpenAI:

```powershell
& .\.venv\Scripts\python.exe scripts\hospital_rag.py query "Quais foram os exames realizados?" --patient-id PCT-000001
& .\.venv\Scripts\python.exe scripts\hospital_rag.py query "Quais sintomas foram relatados?" --patient-name "Emanuel Cunha Cordeiro" --document-type anamnese
```

Consulta de recuperação sem LLM, útil para validação ou quando não houver chave OpenAI:

```powershell
& .\.venv\Scripts\python.exe scripts\hospital_rag.py query "Qual foi a hemoglobina glicada?" --patient-id PCT-000001 --document-type exame_laboratorial --retrieve-only
```

Para abrir a interface interativa em português:

```powershell
& .\.venv\Scripts\python.exe scripts\hospital_rag.py menu
```

O resolvedor não escolhe silenciosamente entre homônimos. Um código de prontuário tem prioridade e consultas sem paciente específico continuam permitidas. Em cenário real, o escopo autorizado do paciente deve ser derivado da identidade autenticada no servidor, não de um valor enviado livremente pelo cliente.

## Fontes e limites

Cada resposta retorna `sources` e `retrieved_documents` com `document_id`, paciente, tipo, data e origem. O serviço não chama o LLM quando não recupera contexto suficiente, evitando inventar dados de paciente.

Busca vetorial não substitui consultas estruturadas para números, datas, agregações ou cálculos longitudinais. Dados e resultados do projeto são sintéticos e não clinicamente validados.

O caminho estruturado percorre os JSONL para responder, por exemplo, “Quantos pacientes há na base?”, “Qual foi a maior glicemia de jejum na base?” e “Quais pacientes possuem registros de glicemia elevada?”. A última consulta compara cada medida com o intervalo informado no próprio documento. Para um paciente, o serviço também seleciona registros por data ISO ou brasileira e pela data mais recente. Agregações fora dessas formas precisam de uma consulta específica.

Cada coleção registra seu modelo de embeddings e a estratégia de um vetor por documento. Para mudar qualquer um deles, configure um novo nome de coleção e execute a ingestão. O serviço confere os IDs persistidos. Os modos de geração com Llama local e Groq ficam reservados na factory. Os componentes de resolução, recuperação, contexto e geração podem ser usados separadamente pelo futuro grafo.

## Testes

Os testes unitários não exigem chave OpenAI, internet ou download de embeddings:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

Eles cobrem leitura e integridade JSONL, conversão de metadados, resolução de pacientes, contexto, resposta com LLM simulado e idempotência do Chroma com embeddings falsos.

Para incluir a medição de dimensão e norma do MiniLM real, defina `$env:TEST_REAL_MINILM = '1'` antes de executar o pytest. O contexto respeita limites de caracteres e de tokens configurados em `src/config.py`. O limite em bytes UTF-8 fornece uma estimativa conservadora do teto de tokens de conteúdo do BPE da OpenAI.

### Validação de 19/11/2026

A suíte Python 3.12 passou nos 42 testes, incluindo o MiniLM real em CPU. A primeira ingestão inseriu os 1.787 documentos dos 200 pacientes. Em outro processo, a segunda ingestão conferiu os IDs e ignorou os 1.787 registros inalterados. A consulta por nome e tipo recuperou apenas `ATD-PCT-000001-01` e `ATD-PCT-000001-02`. O teste OpenAI exercitou o cliente real com transporte HTTP simulado. A chamada ao serviço externo continua pendente de autorização para transmitir o documento sintético e usar a conta.
