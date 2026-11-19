# IMPLEMENTAÇÃO DO RAG HOSPITALAR — PROJETO TCv2

## 1. SEU PAPEL E OBJETIVO

Atue como engenheiro de software sênior, especialista em Python, Machine Learning, LLMs, RAG, LangChain, ChromaDB e LangGraph.

Você deverá implementar um sistema RAG (Retrieval-Augmented Generation) integrado ao projeto Python existente, cujo objetivo é consultar uma base hospitalar sintética.

O sistema deverá recuperar informações de pacientes, prontuários, anamneses, exames laboratoriais, exames de imagem e evoluções clínicas.

Inicialmente, utilizaremos:

- Python 3.12.
- LangChain.
- HuggingFaceEmbeddings.
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
- ChromaDB.
- OpenAI API como LLM gerador inicial.

A arquitetura deverá ser modular e desacoplada do provedor de LLM, permitindo posteriormente utilizar:

1. O modelo Llama 3.1 8B Instruct com fine-tuning LoRA/QLoRA, já desenvolvido e executado localmente neste projeto.
2. Modelos disponibilizados pela Groq Cloud.

O objetivo final será integrar o RAG a um agente inteligente utilizando LangGraph.

IMPORTANTE: nesta implementação, desenvolver somente o RAG. Não implementar ainda a orquestração completa com LangGraph.

---

## 2. REGRA FUNDAMENTAL: PRESERVAR A ESTRUTURA EXISTENTE

O projeto já possui uma estrutura de diretórios estabelecida.

Não criar outro projeto independente.

Não recriar a estrutura raiz.

Não mover ou renomear arquivos existentes sem autorização explícita.

Não sobrescrever documentos, datasets, scripts ou implementações existentes.

Antes de desenvolver, inspecione os arquivos e identifique as funcionalidades que já estão implementadas.

### Estrutura atual

```text
TCv2/
│
├── .pytest_cache/
├── .venv/
│
├── data/
│   ├── processed/
│   ├── rag_data/
│   │   ├── medical_documents.jsonl
│   │   ├── patients.jsonl
│   │   └── validation_report.json
│   │
│   └── raw/
│
├── inference/
│   ├── inference.py
│   └── model_loader.py
│
├── logs/
│
├── notebooks/
│   └── colab_finetuning.ipynb
│
├── scripts/
│   ├── create_repo_hf.py
│   ├── hospital_rag.py
│   ├── model_hf_test.py
│   ├── Process_dataset.py
│   ├── test_bitsandbytes.py
│   └── verify_torch.py
│
├── src/
│
├── tests/
│
├── .env
├── .env.example
├── .gitignore
├── agents.md
├── ALTERACOES.md
├── README.md
├── README_RAG_DATA.md
├── README_RAG.md
└── requirements.txt
```

### Regras de preservação

1. Manter todos os diretórios existentes.
2. Manter a pasta `.venv` e o ambiente Python atual.
3. Não modificar os notebooks de fine-tuning.
4. Não modificar `inference/inference.py` ou `inference/model_loader.py` nesta etapa.
5. Não modificar a implementação de treinamento do modelo.
6. Preservar os arquivos JSONL originais.
7. Não regenerar a base hospitalar.
8. Não sobrescrever os arquivos README existentes sem antes analisar seu conteúdo.
9. Não substituir integralmente o requirements.txt.
10. Não remover dependências que já sejam utilizadas por outras partes do projeto.

O RAG deverá ser desenvolvido preferencialmente dentro da pasta `src/`.

Os scripts de execução deverão permanecer dentro de `scripts/`.

Os testes deverão permanecer dentro de `tests/`.

A persistência do ChromaDB deverá utilizar um novo diretório `database/chroma/`, na raiz do projeto.

Se o projeto já possuir alguma implementação funcional de RAG, avaliar seu reaproveitamento antes de criar módulos redundantes.

---

## 3. ANÁLISE OBRIGATÓRIA ANTES DA IMPLEMENTAÇÃO

Antes de escrever código, execute uma análise do projeto.

Inspecione:

- Estrutura de diretórios.
- requirements.txt.
- .env.example.
- .gitignore.
- README.md.
- README_RAG.md.
- README_RAG_DATA.md.
- scripts/hospital_rag.py.
- Diretório src/.
- Diretório tests/.
- inference/inference.py.
- inference/model_loader.py.
- Arquivos da pasta data/rag_data/.

Atenção especial aos seguintes documentos:

`data/rag_data/medical_documents.jsonl`

`data/rag_data/patients.jsonl`

`data/rag_data/validation_report.json`

O projeto possui uma base hospitalar sintética com:

- **200 pacientes sintéticos.**
- **1.787 documentos hospitalares.**
- Documentos armazenados em formato JSONL.
- Identificação dos pacientes por nome e código de prontuário.
- Relacionamento entre pacientes e documentos clínicos.

Arquivos principais:

- `data/rag_data/patients.jsonl`
- `data/rag_data/medical_documents.jsonl`
- `data/rag_data/validation_report.json`

Antes de iniciar a implementação, validar:

1. Se o arquivo `patients.jsonl` contém exatamente 200 pacientes.
2. Se o arquivo `medical_documents.jsonl` contém exatamente 1.787 documentos.
3. Se todos os documentos possuem identificadores únicos.
4. Se todos os documentos estão associados a pacientes existentes.
5. Se existem pacientes sem documentos associados.
6. Se existem nomes duplicados ou homônimos.

Caso as quantidades encontradas sejam diferentes das esperadas, apresentar um alerta e investigar a inconsistência antes de executar a ingestão.

Não modificar nem regenerar os arquivos JSONL originais.

Verifique o formato real dos JSONL, os campos disponíveis, a quantidade de registros, os identificadores e os relacionamentos entre pacientes e documentos.

Não presuma que os campos possuem nomes ou estruturas específicos sem inspecionar os registros.

O arquivo validation_report.json deverá ser utilizado para compreender a qualidade e a integridade da base, quando aplicável.

Antes de iniciar a implementação, apresente um breve diagnóstico da estrutura encontrada e um plano de alterações.

Depois, prossiga com a implementação sem aguardar uma nova autorização, desde que não seja necessário modificar ou excluir arquivos existentes de maneira destrutiva.

---

## 4. ESTRUTURA DE DIRETÓRIOS A SER IMPLEMENTADA

Preservar a estrutura atual e expandir o diretório src/.

Estrutura desejada:

```text
TCv2/
│
├── data/
│   ├── processed/
│   ├── rag_data/
│   │   ├── medical_documents.jsonl
│   │   ├── patients.jsonl
│   │   └── validation_report.json
│   └── raw/
│
├── database/
│   └── chroma/
│
├── inference/
│   ├── inference.py
│   └── model_loader.py
│
├── logs/
│
├── notebooks/
│
├── scripts/
│   ├── hospital_rag.py
│   └── [demais scripts existentes]
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_service.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── document_loader.py
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chroma_service.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── patient_resolver.py
│   │   └── retriever_service.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── llm_factory.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   ├── context_builder.py
│   │   └── rag_service.py
│   │
│   └── models/
│       ├── __init__.py
│       └── schemas.py
│
├── tests/
│   ├── test_document_loader.py
│   ├── test_embeddings.py
│   ├── test_vectorstore.py
│   ├── test_retrieval.py
│   ├── test_patient_resolver.py
│   └── test_rag.py
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── README_RAG.md
└── README_RAG_DATA.md
```

A estrutura é uma referência arquitetural.

Caso alguns módulos ou diretórios já existam, reutilize-os.

Não crie arquivos duplicados.

---

## 5. CONFIGURAÇÕES CENTRALIZADAS

Criar ou adaptar:

`src/config/settings.py`

Centralizar configurações por variáveis de ambiente.

Utilizar python-dotenv.

Variáveis:

```env
# LLM
LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=

# Embeddings
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./database/chroma
CHROMA_COLLECTION_NAME=hospital_patients

# Retrieval
RETRIEVER_K=5

# Dataset
PATIENTS_FILE=./data/rag_data/patients.jsonl
DOCUMENTS_FILE=./data/rag_data/medical_documents.jsonl

# Logging
LOG_LEVEL=INFO
```

Não inventar um identificador de modelo OpenAI.

O modelo deverá ser configurado por OPENAI_MODEL, utilizando um identificador efetivamente disponibilizado na API.

Não colocar chaves ou credenciais diretamente no código.

Preservar o conteúdo do .env existente, adicionando somente configurações necessárias.

Atualizar .env.example sem inserir credenciais reais.

Validar as configurações na inicialização.

Utilizar caminhos relativos à raiz do projeto, evitando problemas de execução quando o script for iniciado em outro diretório.

---

## 6. CARREGAMENTO E TRANSFORMAÇÃO DOS DADOS

Implementar:

`src/ingestion/document_loader.py`

Responsabilidades:

1. Carregar patients.jsonl.
2. Carregar medical_documents.jsonl.
3. Validar os registros.
4. Preservar a estrutura original dos documentos.
5. Relacionar documentos aos respectivos pacientes.
6. Converter os documentos para LangChain Document.
7. Criar identificadores estáveis.
8. Registrar erros de processamento.

O conteúdo clínico deverá ser utilizado como page_content.

Os metadados deverão incluir, quando disponíveis:

- document_id
- patient_id
- patient_name
- document_type
- document_date
- source

Adaptar a implementação aos campos reais encontrados nos JSONL.

Não descartar informações clínicas.

Não concatenar todos os documentos de um paciente em um único documento.

Manter os documentos separados por finalidade.

Exemplos:

- patient_profile
- anamnesis
- laboratory
- imaging
- clinical_evolution

Preservar os documentos originais em português.

Tratar corretamente acentuação e caracteres especiais.

Verificar os limites de tipos aceitos nos metadados do ChromaDB.

Se existirem metadados complexos, convertê-los adequadamente sem perder informações relevantes do conteúdo.

### Validação de integridade da base hospitalar

A implementação deverá validar a integridade dos relacionamentos entre os 200 pacientes e os 1.787 documentos.

Implementar verificações para:

- Quantidade total de pacientes.
- Quantidade total de documentos.
- Quantidade de documentos por paciente.
- Quantidade de documentos por tipo.
- Documentos com `patient_id` inexistente.
- Documentos com identificadores duplicados.
- Pacientes sem documentos associados.

As quantidades esperadas deverão ser utilizadas como referência de validação inicial, mas o código não deverá possuir limites fixos de 200 pacientes ou 1.787 documentos.

O sistema deverá permitir o crescimento da base sem necessidade de alterações no código.

Registrar os resultados das validações nos logs.

---

## 7. EMBEDDINGS

Implementar:

`src/embeddings/embedding_service.py`

Utilizar:

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

Modelo:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Configuração:

```python
model_kwargs={
    "device": "cpu"
}

encode_kwargs={
    "normalize_embeddings": True
}
```

Requisitos:

- Instanciar o modelo uma única vez por processo.
- Permitir reutilização.
- Evitar recarregamento a cada consulta.
- Registrar o modelo carregado.
- Tratar erros de inicialização.
- Permitir alteração do modelo via configuração.

A dimensionalidade esperada é 384.

Não implementar treinamento ou fine-tuning de embeddings.

O embedding será utilizado exclusivamente para indexação e recuperação semântica.

Inicialmente utilizar CPU para preservar recursos da GPU utilizados pelo Llama.

---

## 8. CHROMADB

Implementar:

`src/vectorstore/chroma_service.py`

Utilizar:

```python
from langchain_chroma import Chroma
```

Configuração:

- Collection: hospital_patients.
- Diretório: database/chroma/.
- Persistência local.
- Embedding function: embedding_service.

Implementar:

- create_or_load_vectorstore()
- add_documents()
- update_documents()
- search_documents()
- search_by_patient()
- search_by_document_type()
- get_collection_stats()
- clear_collection()

### Persistência e idempotência

Utilizar document_id como identificador estável.

A ingestão não deverá gerar duplicidades.

Implementar upsert ou mecanismo equivalente de atualização.

Não recriar a coleção automaticamente.

Não apagar documentos durante a inicialização.

A função clear_collection deverá exigir confirmação explícita.

Registrar estatísticas de ingestão:

- Total de documentos lidos.
- Total inserido.
- Total atualizado.
- Total ignorado.
- Total rejeitado.

Garantir que a base possa ser reutilizada após reinicialização do programa.

### Validação da indexação vetorial

Após a primeira ingestão, verificar se todos os documentos válidos foram indexados no ChromaDB.

A base atual possui 1.787 documentos hospitalares.

Se cada documento JSONL for representado por um único LangChain Document, sem divisão em chunks, a quantidade esperada de registros vetoriais será de 1.787, desde que todos sejam válidos.

Se for utilizada uma estratégia de chunking, a quantidade de vetores poderá ser maior.

O sistema deverá apresentar:

- Quantidade de documentos originais processados.
- Quantidade de documentos convertidos.
- Quantidade de chunks gerados, se aplicável.
- Quantidade de vetores armazenados.
- Quantidade de documentos rejeitados.
- Quantidade de documentos atualizados.

Executar a ingestão novamente não deverá aumentar a quantidade de vetores quando os documentos originais não tiverem sido modificados.

Caso o modelo de embedding ou a estratégia de chunking seja alterada, verificar a necessidade de reconstruir a coleção ou criar uma nova coleção versionada.

Não misturar embeddings de modelos diferentes na mesma coleção.

---

## 9. IDENTIFICAÇÃO DOS PACIENTES

Implementar:

`src/retrieval/patient_resolver.py`

A identificação correta dos pacientes é requisito prioritário.

O sistema deverá permitir localizar pacientes por:

- Código do prontuário.
- Nome completo.
- Nome parcial, quando inequívoco.

Implementar normalização textual:

- Ignorar diferenças entre maiúsculas e minúsculas.
- Normalizar espaços.
- Tratar acentuação.
- Evitar correspondências incorretas.

Ordem de prioridade:

1. Correspondência exata do código.
2. Correspondência exata do nome.
3. Correspondência parcial inequívoca.
4. Busca semântica como estratégia complementar.

Quando houver homônimos, não escolher arbitrariamente.

Solicitar esclarecimento ao usuário.

Após identificar o paciente, utilizar patient_id como filtro de metadados nas consultas ao ChromaDB.

Não misturar documentos de pacientes diferentes.

Quando a pergunta não mencionar paciente específico, permitir recuperação sem filtro de paciente.

---

## 10. RETRIEVER

Implementar:

`src/retrieval/retriever_service.py`

Responsabilidades:

- Busca por similaridade semântica.
- Busca com filtros de metadados.
- Busca por paciente.
- Busca por tipo documental.
- Recuperação dos K documentos mais relevantes.
- Retorno de documentos e metadados.
- Exposição das fontes recuperadas.

Utilizar inicialmente K=5.

Permitir alteração via configuração.

Avaliar posteriormente mecanismos de reranking, MMR e busca híbrida, mas não adicionar complexidade desnecessária à primeira versão.

### Consultas exatas e agregações

Não depender exclusivamente de busca vetorial para:

- Localizar códigos.
- Identificar nomes.
- Consultar datas exatas.
- Contar pacientes.
- Comparar resultados entre todos os pacientes.
- Calcular médias ou máximos.

Quando necessário, utilizar consultas estruturadas aos dados JSONL.

Não afirmar que uma busca vetorial encontrou todos os pacientes que atendem a determinada condição.

---

## 11. FACTORY DE LLM

Implementar:

`src/llm/llm_factory.py`

A arquitetura deverá desacoplar o RAG do provedor de LLM.

### Provedor inicial: OpenAI

Utilizar:

```python
from langchain_openai import ChatOpenAI
```

Configurações:

- OPENAI_API_KEY
- OPENAI_MODEL
- LLM_PROVIDER=openai

Criar função:

```python
get_llm()
```

Ela deverá retornar uma instância compatível com a interface de chat do LangChain.

Não fixar o modelo diretamente no código.

Não assumir que o nome do modelo é válido sem configuração e validação.

### Preparação para Llama local

O projeto já possui:

- inference/inference.py
- inference/model_loader.py

Inspecionar esses arquivos para compreender como o Llama fine-tuned é carregado e executado.

Não modificar esses arquivos nesta etapa.

Projetar a factory para aceitar futuramente um adaptador LangChain que utilize a inferência local existente.

Se futuramente houver uma API local compatível com OpenAI, permitir sua utilização através de ChatOpenAI com base_url configurável.

Não presumir que essa API já existe.

### Preparação para Groq

Prever a implementação futura utilizando ChatGroq.

O provedor deverá ser selecionado através de configuração:

LLM_PROVIDER=openai
LLM_PROVIDER=local
LLM_PROVIDER=groq

Não é necessário implementar local e Groq agora.

O objetivo é permitir adicionar esses provedores sem modificar o Retriever, ChromaDB, embeddings ou Context Builder.

---

## 12. SERVIÇO RAG

Implementar:

- src/rag/prompts.py
- src/rag/context_builder.py
- src/rag/rag_service.py

Separar as responsabilidades.

### Fluxo

1. Receber a pergunta.
2. Resolver a identificação do paciente.
3. Aplicar filtros necessários.
4. Recuperar os documentos.
5. Construir o contexto.
6. Enviar pergunta e contexto ao LLM.
7. Gerar resposta.
8. Retornar resposta e fontes.

O Context Builder deverá preservar a identificação das fontes e evitar ultrapassar desnecessariamente a janela de contexto do modelo.

### Prompt do sistema

Utilizar instruções semelhantes a:

"Você é um assistente de informações hospitalares destinado a apoiar profissionais de saúde.

Responda às perguntas utilizando os documentos hospitalares fornecidos como contexto.

REGRAS:

1. Utilize as informações recuperadas como fonte principal.
2. Não invente informações clínicas.
3. Não invente resultados de exames.
4. Não presuma informações ausentes.
5. Se não houver informação suficiente, informe claramente.
6. Diferencie informações documentadas de interpretações.
7. Preserve datas, unidades e valores originais.
8. Não misture informações de pacientes diferentes.
9. Identifique as fontes utilizadas.
10. Informe divergências entre documentos.
11. Não produza diagnósticos definitivos.
12. Não substitua a avaliação de profissionais de saúde.

Responda em português brasileiro, de maneira clara e objetiva."

### Contexto vazio

Se nenhum documento relevante for recuperado, o sistema deverá informar que não localizou informações suficientes.

Não utilizar conhecimento geral do LLM para inventar dados do paciente.

### Resposta estruturada

Criar schemas Pydantic.

Retornar:

- success
- question
- answer
- patient_id
- patient_name
- sources
- retrieved_documents
- error

Cada fonte deverá conter, quando disponível:

- document_id
- patient_id
- patient_name
- document_type
- document_date
- source

Não inventar fontes ou documentos.

Não apresentar como recuperados documentos que não foram efetivamente consultados.

---

## 13. PREPARAÇÃO PARA LANGGRAPH

Não implementar o agente completo agora.

Entretanto, o RAG deverá ter componentes independentes que possam futuramente ser utilizados como nós de um StateGraph.

Separar:

- Patient Resolver.
- Retriever.
- Context Builder.
- LLM Factory.
- Answer Generator.

Preparar um estado conceitual semelhante a:

```python
class RAGState:
    question: str
    patient_id: str | None
    patient_name: str | None
    retrieved_documents: list
    context: str
    answer: str | None
    sources: list
    error: str | None
```

A implementação definitiva poderá utilizar TypedDict ou outro formato compatível com LangGraph.

No futuro, o agente terá:

- Nó de classificação de intenção.
- Nó de consulta RAG.
- Nó de inferência Llama.
- Nó de consolidação de respostas.
- Roteamento condicional.
- Tratamento de erros.

O RAG deverá poder ser invocado independentemente pelo agente.

---

## 14. INTERFACE DE TESTE

Utilizar ou adaptar:

`scripts/hospital_rag.py`

Primeiro inspecionar se já existe uma implementação que possa ser reaproveitada.

O script deverá oferecer um menu interativo:

1. Executar ingestão da base.
2. Consultar informações de pacientes.
3. Consultar exames laboratoriais.
4. Consultar anamneses.
5. Consultar por código de prontuário.
6. Consultar por nome.
7. Exibir estatísticas do ChromaDB.
8. Sair.

Exemplos:

- Qual é a idade do paciente João da Silva?
- Quais foram os resultados do último hemograma do paciente João?
- Qual foi o resultado de glicose do paciente PRT0001?
- Quais sintomas foram relatados na anamnese de Maria Oliveira?
- Quais exames foram realizados pelo paciente PRT0005?
- Quais pacientes possuem registros de glicemia elevada?

O programa deverá funcionar em português brasileiro.

Apresentar a resposta e as fontes documentais.

Não utilizar o LLM para substituir consultas estruturadas quando a operação exigir exatidão ou agregação.

---

## 15. TESTES AUTOMATIZADOS

Expandir a pasta tests/ existente.

Criar testes para:

### Documentos
- Leitura dos JSONL.
- Conversão para Document.
- Validação dos metadados.
- Documentos inválidos.
- Relacionamento entre pacientes e documentos.

### Embeddings
- Inicialização.
- Geração de vetores.
- Dimensionalidade 384.
- Normalização.

### ChromaDB
- Inserção.
- Atualização.
- Persistência.
- Recuperação.
- Filtros.
- Idempotência.

### Identificação de pacientes
- Código exato.
- Nome exato.
- Nome parcial.
- Acentuação.
- Homônimos.
- Paciente não encontrado.

### RAG
- Recuperação.
- Construção do contexto.
- Resposta com LLM simulado.
- Fontes.
- Contexto vazio.
- Erros de execução.

Utilizar pytest.

Não exigir chave real da OpenAI para executar testes unitários.

Utilizar fake LLM ou modelo simulado.

Separar testes de integração que necessitem de internet, download de modelos ou API externa.

---

## 16. DEPENDÊNCIAS

O projeto já possui requirements.txt.

Não substituí-lo integralmente.

Primeiro verificar as dependências existentes e suas versões.

Adicionar somente os pacotes ausentes e necessários.

Considerar:

- langchain
- langchain-core
- langchain-community
- langchain-huggingface
- langchain-chroma
- langchain-openai
- chromadb
- sentence-transformers
- python-dotenv
- pydantic
- pytest

Evitar versões antigas e APIs depreciadas.

Verificar compatibilidade com Python 3.12 e com as dependências de PyTorch e Transformers já instaladas.

Não reinstalar ou atualizar indiscriminadamente PyTorch, CUDA, Transformers ou bitsandbytes.

Essas bibliotecas já são utilizadas pelo módulo de inferência do Llama.

Preservar o ambiente `.venv` existente.

---

## 17. LOGS

Utilizar a pasta existente:

`logs/`

Registrar:

- Inicialização.
- Carregamento do embedding.
- Ingestão.
- Quantidade de documentos.
- Consultas.
- Quantidade de documentos recuperados.
- Tempo de recuperação.
- Tempo de geração.
- Erros da API.

Não registrar chaves de API.

Evitar registrar conteúdo clínico completo nos logs.

---

## 18. DOCUMENTAÇÃO

Preservar os arquivos de documentação existentes.

Inspecionar:

- README.md
- README_RAG.md
- README_RAG_DATA.md

Atualizar preferencialmente README_RAG.md para documentar a implementação do RAG.

Não apagar informações existentes sobre a geração da base sintética.

Documentar:

1. Arquitetura.
2. Dependências.
3. Configuração.
4. OpenAI API.
5. Embeddings.
6. ChromaDB.
7. Ingestão.
8. Consultas.
9. Identificação de pacientes.
10. Testes.
11. Persistência.
12. Integração futura com LangGraph.
13. Preparação para Llama local.
14. Preparação para Groq.

Incluir exemplos de comandos para Windows PowerShell.

---

## 19. CRITÉRIOS DE ACEITAÇÃO

A implementação será considerada concluída quando:

1. Os JSONL de data/rag_data forem carregados corretamente.
2. Os documentos forem convertidos para LangChain Document.
3. O MiniLM gerar embeddings de 384 dimensões.
4. O ChromaDB armazenar os vetores localmente.
5. A persistência funcionar após reinicialização.
6. A ingestão for idempotente.
7. A identificação de pacientes funcionar por nome e código.
8. Os filtros de metadados funcionarem.
9. O LLM OpenAI responder com base no contexto recuperado.
10. As fontes forem retornadas corretamente.
11. O sistema tratar ausência de informações.
12. Os testes automatizados forem executados.
13. A arquitetura estiver desacoplada do provedor de LLM.
14. A integração futura com LangGraph for viável.
15. A implementação atual de inferência do Llama permanecer intacta.
16. A estrutura e os arquivos preexistentes forem preservados.
### Critérios adicionais de aceitação

17. A implementação deverá identificar corretamente os 200 pacientes da base sintética.
18. A implementação deverá processar os 1.787 documentos hospitalares.
19. Todos os documentos válidos deverão ser associados aos pacientes correspondentes.
20. A quantidade de registros indexados deverá ser consistente com a estratégia de indexação utilizada.
21. Uma segunda execução da ingestão não deverá gerar documentos ou vetores duplicados.
22. O sistema deverá apresentar estatísticas de documentos por paciente e por tipo documental.
23. O sistema deverá permitir adicionar novos pacientes e documentos sem modificar o código-fonte.

---

## 20. ORDEM DE EXECUÇÃO

Executar sequencialmente:

FASE 1 — Diagnóstico do projeto existente.

FASE 2 — Análise dos JSONL e dos scripts existentes.

FASE 3 — Organização dos módulos em src/.

FASE 4 — Configurações e dependências.

FASE 5 — Implementação dos embeddings.

FASE 6 — Implementação do ChromaDB.

FASE 7 — Ingestão e transformação dos documentos.

FASE 8 — Recuperação e identificação de pacientes.

FASE 9 — Integração com OpenAI.

FASE 10 — Serviço RAG.

FASE 11 — Interface em scripts/hospital_rag.py.

FASE 12 — Testes automatizados.

FASE 13 — Documentação.

Ao final, executar os testes e corrigir os problemas encontrados.

Não declarar testes aprovados sem executá-los.

Apresentar um relatório final contendo:

- Arquivos criados.
- Arquivos modificados.
- Dependências adicionadas.
- Funcionalidades implementadas.
- Testes executados e resultados.
- Comandos para executar a aplicação.
- Exemplos de consultas reais à base.
- Limitações identificadas.

## 21. DIRETRIZ FINAL

A prioridade é desenvolver um RAG independente, modular e confiável, respeitando integralmente a estrutura atual do projeto TCv2.

Não reconstruir componentes que já funcionam.

Não alterar o fine-tuning ou a inferência do Llama.

Não criar um projeto paralelo.

O RAG deverá utilizar inicialmente a OpenAI API, mas sua arquitetura deverá permitir substituir o LLM por inferência local do Llama fine-tuned ou pela Groq Cloud.

O resultado deverá estar preparado para ser incorporado futuramente a um agente LangGraph, responsável por decidir quando consultar os dados hospitalares, quando utilizar o conhecimento biomédico do Llama e quando combinar os dois.

Implemente efetivamente o código, não entregue somente sugestões ou exemplos.