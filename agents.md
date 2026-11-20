# Diretrizes de Desenvolvimento e Escopo do Projeto — TCv2

Este documento é a referência de escopo, arquitetura, qualidade e segurança do TCv2. A especificação do RAG em `Codex.md` foi incorporada para a fase de geração com OpenAI. A convenção local de configuração está na seção 10.

---

## 1. Visão geral

### 1.1 Objetivo final

Construir, para fins acadêmicos, um agente de apoio à informação biomédica que:

1. responda questões clínicas gerais com um modelo Llama 3.1 8B Instruct ajustado por QLoRA;
2. recupere informações relevantes de uma base hospitalar **inteiramente sintética** usando RAG (*Retrieval-Augmented Generation*);
3. fundamente a resposta nos documentos recuperados, identificando as fontes utilizadas;
4. declare limitações, não invente evidências e não substitua a avaliação de profissional habilitado.

O projeto não é um dispositivo médico, não diagnostica, não prescreve e não deve ser usado para atendimento real ou decisão clínica autônoma.

### 1.2 Estado atual do projeto

As etapas abaixo já foram implementadas:

- Pipeline de dados multi-fonte para fine-tuning.
- Integração com MedQuAD (`lavita/MedQuAD`) e PubMedQA (`qiaojin/PubMedQA`).
- Geração de protocolos e casos hospitalares fictícios, anonimização por regras e curadoria.
- Exportação de treino, validação, manifesto, descartes auditáveis e conjunto de avaliação PubMedQA.
- Notebook de fine-tuning com Unsloth/QLoRA, executado no Kaggle após tentativa no Google Colab.
- Upload do adaptador LoRA ao Hugging Face e código de carregamento/inferência local com GPU.
- Geração de pacientes e documentos clínicos sintéticos para RAG.
- Ingestão local no Chroma e busca vetorial com filtros por paciente e tipo de documento.

O serviço RAG usa o adaptador OpenAI e devolve fontes documentais. A integração do modelo fine-tuned com o agente é uma etapa futura. Consulte `README_RAG.md` para execução e limites.

### 1.3 Fontes e finalidade dos dados

| Fonte | Uso | Situação |
|---|---|---|
| MedQuAD (`lavita/MedQuAD`) | Perguntas e respostas médicas de fontes NIH | Fine-tuning |
| PubMedQA `pqa_artificial` | Perguntas clínicas com contexto de artigos | Fine-tuning, por amostragem configurável |
| PubMedQA `pqa_labeled` | Perguntas com decisão `yes`/`no`/`maybe` | Avaliação; não entra no treino |
| Protocolos hospitalares sintéticos | Protocolos, FAQ, casos e exemplos de segurança | Fine-tuning |
| Pacientes e documentos sintéticos | Cadastros, anamnese, exames, laudos e evoluções | RAG |

Na execução registrada em `data/processed/dataset_summary.json`, o dataset curado contém 26.784 exemplos: 16.122 MedQuAD, 10.000 PubMedQA e 662 sintéticos. Há 24.105 exemplos de treino, 2.679 de validação e 1.000 exemplos de avaliação PubMedQA.

Todos os dados internos, pacientes, resultados, protocolos e prontuários criados pelo projeto são artificiais. Eles não representam pessoas reais nem informação clínica validada.

---

## 2. Arquitetura e estrutura do repositório

```text
TCv2/
├── data/
│   ├── raw/
│   │   └── synthetic_hospital_raw.jsonl     # PII fictícia, somente demonstração de anonimização
│   ├── processed/
│   │   ├── finetuning_train.jsonl           # Gerado; ignorado pelo Git
│   │   ├── finetuning_val.jsonl             # Gerado; ignorado pelo Git
│   │   ├── eval_pubmedqa_labeled.jsonl      # Gerado; ignorado pelo Git
│   │   ├── dataset_summary.json
│   │   ├── dataset_manifest.csv
│   │   └── discarded*.json                  # Auditoria dos descartes
│   └── rag_data/
│       ├── patients.jsonl                   # Pacientes sintéticos
│       ├── medical_documents.jsonl          # Documentos sintéticos para RAG
│       └── validation_report.json
├── database/chroma/                         # Coleção persistida do serviço RAG
├── inference/
│   ├── model_loader.py                      # Carregamento do modelo-base e adaptador LoRA
│   └── inference.py                         # Inferência local experimental e métricas de VRAM
├── logs/
│   └── execucao.log
├── notebooks/
│   └── colab_finetuning.ipynb               # Fine-tuning QLoRA/Unsloth
├── scripts/
│   ├── Process_dataset.py                   # Orquestrador do pipeline multi-fonte
│   ├── hospital_rag.py                      # Geração, ingestão e recuperação RAG atuais
│   ├── model_hf_test.py                     # Testes de consumo do modelo publicado
│   └── create_repo_hf.py
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── anonymizer.py
│   ├── curation.py
│   ├── synthetic_generator.py
│   ├── jsonl_converter.py
│   └── utils/logger.py
├── tests/
├── .env                                     # Segredos; nunca versionar
├── .env.example
├── agents.md
├── ALTERACOES.md                            # Histórico das alterações do pipeline
├── README.md
├── README_RAG_DATA.md
└── requirements.txt
```

Arquivos `.jsonl` e `.json` de `data/processed/` são artefatos reproduzíveis e estão ignorados pelo Git. Após um clone, devem ser regenerados pelo pipeline ou obtidos pelo canal de armazenamento definido para o projeto; sua presença não deve ser presumida.

### 2.1 Responsabilidades atuais

- `src/config.py`: configurações, caminhos com `pathlib.Path`, tokens obtidos do ambiente e parâmetros de execução.
- `src/data_loader.py`: acesso e limpeza das fontes remotas de perguntas e respostas.
- `src/anonymizer.py`: mascaramento de PII por regras e auditoria de substituições.
- `src/curation.py`: normalização textual, filtros de qualidade, deduplicação e motivo de descarte.
- `src/synthetic_generator.py`: geração determinística de dados hospitalares fictícios para fine-tuning.
- `src/jsonl_converter.py`: schema de mensagens, divisão treino/validação, serialização e validação JSONL.
- `scripts/Process_dataset.py`: apenas coordenação do pipeline de preparação.
- `scripts/hospital_rag.py`: geração/validação dos dados RAG, ingestão no Chroma e recuperação vetorial atual.
- `inference/model_loader.py`: obtenção do modelo-base e aplicação do adaptador PEFT/LoRA.
- `inference/inference.py`: geração local e instrumentação de memória durante experimentos.

---

## 3. Dados de fine-tuning

### 3.1 Formato alvo

Cada linha dos arquivos de fine-tuning deve conter estritamente três mensagens, nesta ordem:

```json
{
  "messages": [
    {"role": "system", "content": "<system prompt configurado>"},
    {"role": "user", "content": "<pergunta>"},
    {"role": "assistant", "content": "<resposta com fonte quando aplicável>"}
  ]
}
```

O `SYSTEM_PROMPT` centralizado em `src/config.py` deve instruir o modelo a usar evidência, distinguir achados de interpretação, responder no idioma da pergunta, citar fonte, declarar limitações e exigir validação profissional para tratamento ou prescrição.

### 3.2 Pipeline obrigatório

O fluxo de dados é:

`carregar → anonimizar → curar → converter → dividir → exportar → validar → registrar resumo e manifesto`

- Remover ou auditar perguntas/respostas nulas, vazias ou fora dos limites definidos.
- Preservar a proveniência (`source`, `source_ref`, categoria) até a geração do manifesto.
- Aplicar anonimização antes de qualquer conteúdo entrar no conjunto de fine-tuning.
- Usar UTF-8 e `ensure_ascii=False` na exportação.
- Validar todas as linhas com `json.loads()` e validar papéis, ordem e conteúdo não vazio.
- A divisão atual é aleatória e reprodutível por semente; não deve ser chamada de estratificada enquanto essa estratégia não for implementada e testada.
- Manter `pqa_labeled` fora de treino e validação para reduzir vazamento de avaliação.

### 3.3 Privacidade e rastreabilidade

- O arquivo em `data/raw/` possui somente PII fictícia e existe para demonstrar o antes/depois da anonimização.
- Nunca adicionar dados pessoais ou prontuários reais ao repositório, aos datasets, aos logs ou ao índice vetorial.
- A anonimização atual baseada em regex é educacional. Em um cenário real, exigiria detecção de entidades nomeadas, revisão humana, controles de acesso, retenção e governança de dados.
- Todo descarte relevante deve registrar a razão de exclusão em arquivo auditável.

---

## 4. Fine-tuning e inferência local

O notebook `notebooks/colab_finetuning.ipynb` contém a preparação para fine-tuning supervisionado com Unsloth, quantização de 4 bits e QLoRA sobre Llama 3.1 8B Instruct. A execução de treinamento foi realizada na plataforma Kaggle; os datasets processados foram transferidos para o ambiente de entrada utilizado no treinamento.

O adaptador publicado é consumido pelos arquivos em `inference/`. O carregamento deve:

- obter o token exclusivamente de variável de ambiente;
- descobrir o modelo-base a partir da configuração do adaptador PEFT;
- manter o modelo em modo de avaliação;
- ser explícito sobre GPU/CUDA, quantização e requisitos de memória;
- nunca embutir tokens, chaves ou dados privados no código.

O identificador do adaptador, parâmetros de geração e limites de hardware devem evoluir para configuração centralizada ou variáveis de ambiente, e não permanecer como valores rígidos em módulos de execução.

---

## 5. RAG: estado atual e arquitetura alvo

### 5.1 O que já existe

`scripts/hospital_rag.py` já implementa:

- geração determinística de pacientes e documentos clínicos artificiais;
- validação de IDs, vínculos paciente-documento, datas, campos obrigatórios e densidade documental;
- indexação atual com um vetor por documento e embeddings multilíngues no Chroma;
- busca por similaridade com filtros por `patient_id`, `patient_name` e `document_type`;
- desambiguação de nomes, sem escolher silenciosamente entre múltiplos pacientes.

O relatório atual registra 200 pacientes e 1.787 documentos válidos, distribuídos entre cadastro, anamnese, exames laboratoriais, laudos de imagem e evoluções.

### 5.2 O que falta para o RAG integrado

A busca vetorial isolada não constitui a entrega final. Devem ser implementados e testados:

1. recuperação programática sem acoplamento à impressão de CLI;
2. construção de contexto com limite de tokens, ordenação e deduplicação de trechos;
3. template de conversa que apresente ao modelo somente o contexto recuperado pertinente;
4. chamada ao Llama fine-tuned e pós-processamento da resposta;
5. citações verificáveis contendo, no mínimo, `document_id`, data e fonte dos documentos realmente usados;
6. resposta explícita de insuficiência de evidência quando não houver base recuperada;
7. avaliação de recuperação e de geração, incluindo casos de ausência de documentos e conflito de evidências.

### 5.3 Fluxo RAG obrigatório

```text
Pergunta → autenticação/autorização → escopo de paciente autorizado
→ recuperação filtrada no índice → seleção e montagem de contexto
→ Llama fine-tuned → validação de segurança e citações → resposta
```

Em produção, `patient_id` não pode ser aceito como autorização fornecida livremente pelo cliente. O escopo de paciente/tenant deve ser derivado de identidade autenticada e políticas executadas no servidor. A filtragem demonstrativa atual é somente uma prova de conceito acadêmica.

Busca vetorial não substitui acesso estruturado para números, intervalos de data, tendências laboratoriais ou cálculos longitudinais. Quando necessário, a solução deve combinar recuperação semântica com uma camada estruturada/SQL ou função determinística apropriada.

### 5.4 Organização recomendada para a próxima etapa

Ao integrar o RAG, separar responsabilidades em módulos, por exemplo:

- `src/rag/ingestion.py`: conversão de documentos e indexação;
- `src/rag/retriever.py`: filtros, resolução de paciente e recuperação;
- `src/rag/context_builder.py`: seleção, truncamento e formatação dos trechos;
- `src/rag/service.py`: coordenação do RAG com o modelo;
- `src/rag/citations.py`: geração e validação de referências;
- `src/rag/schemas.py`: dataclasses ou modelos de dados para consulta, contexto e resposta.

Esses caminhos são uma arquitetura alvo; sua criação deve acompanhar testes e migração gradual das responsabilidades hoje presentes em `scripts/hospital_rag.py`.

---

## 6. Roadmap

- [x] Configuração, schema de mensagens, carregamento MedQuAD e exportação JSONL.
- [x] Integração PubMedQA, base sintética, anonimização, curadoria e auditoria de descartes.
- [x] Dataset de treino/validação, conjunto de avaliação e manifesto de proveniência.
- [x] Notebook de fine-tuning QLoRA e execução do fine-tuning no Kaggle.
- [x] Carregamento local do adapter LoRA e inferência experimental em GPU.
- [x] Base RAG sintética, validação, indexação Chroma e recuperação vetorial filtrada.
- [ ] Integrar retriever, construtor de contexto, modelo fine-tuned e citações em um serviço RAG.
- [ ] Criar testes unitários e de integração específicos para recuperação, isolamento de pacientes e resposta fundamentada.
- [ ] Criar protocolo de avaliação comparando modelo sem RAG, modelo fine-tuned e modelo fine-tuned com RAG.
- [ ] Refatorar scripts experimentais para logging, tipagem, configuração injetável e interfaces de serviço adequadas.

---

## 7. Princípios SOLID

### 7.1 SRP — responsabilidade única

Cada função, classe e módulo deve ter uma única razão para mudar. Separar aquisição de dados, anonimização, curadoria, indexação, recuperação, construção de contexto, inferência, citações e apresentação CLI.

### 7.2 OCP — aberto para extensão, fechado para modificação

Novas fontes, bancos vetoriais, modelos de embeddings, modelos geradores ou formatos de saída devem ser adicionados por interfaces/adaptadores, sem reescrever o núcleo do pipeline ou do serviço RAG.

### 7.3 LSP — substituição de Liskov

Implementações alternativas de loader, repositório vetorial, cliente de LLM ou recuperador devem preservar contratos, tipos de retorno, exceções documentadas e garantias de segurança.

### 7.4 ISP — interfaces pequenas

Preferir protocolos específicos, por exemplo `DocumentRetriever`, `EmbeddingProvider`, `GenerationClient` e `CitationFormatter`, em vez de dependências amplas entre módulos.

### 7.5 DIP — inversão de dependência

Serviços de alto nível dependem de abstrações. Caminhos, credenciais, modelos, limites de contexto, retriever e gerador devem ser injetáveis e configuráveis, facilitando testes com fakes/mocks.

---

## 8. Clean Code e padrões Python

- Usar `snake_case` para variáveis e funções, `PascalCase` para classes e `UPPER_SNAKE_CASE` para constantes.
- Aplicar type hints a funções e métodos públicos, incluindo retorno.
- Escrever docstrings Google Style em módulos, classes e APIs públicas.
- Preferir funções pequenas, puras quando possível, com no máximo três argumentos simples; usar dataclasses/configurações para parâmetros relacionados.
- Usar `pathlib.Path` em I/O de arquivos. Não embutir caminhos absolutos.
- Usar `logging` nos módulos de aplicação e biblioteca. `print()` pode existir apenas na borda de uma CLI para exibir resultado já produzido por serviços; não deve conter regra de negócio.
- Capturar exceções específicas, encadeá-las com `raise ... from exc` quando necessário e nunca silenciar falhas com `except:` ou `except Exception:` genérico.
- Remover imports não utilizados, código morto e valores de configuração duplicados ao tocar um módulo.
- Evitar efeitos colaterais na importação de módulos; criação de diretórios, downloads, indexação e carregamento de modelo devem ocorrer em pontos explícitos de execução.
- Não registrar segredos, textos clínicos sensíveis ou tokens de autenticação.

---

## 9. Testes, qualidade e critérios de aceite

Usar `pytest` com testes determinísticos, mocks para serviços externos e dados sintéticos mínimos. Alterações devem preservar os testes existentes e incluir cobertura para o comportamento novo.

### 9.1 Testes mínimos do pipeline

- carregamento e limpeza de fontes com mocks;
- anonimização de identificadores esperados;
- curadoria, deduplicação e motivo de descarte;
- formato JSONL, parsing linha a linha e três papéis obrigatórios;
- reprodutibilidade da divisão por semente;
- ausência de sobreposição entre dados de treino e avaliação PubMedQA.

### 9.2 Testes mínimos do RAG

- schema e validação dos arquivos `patients.jsonl` e `medical_documents.jsonl`;
- ingestão idempotente ou procedimento explícito de reconstrução de índice;
- filtros por paciente e tipo documental;
- desambiguação de nome;
- garantia de que uma consulta autorizada a um paciente não recupera documentos de outro;
- montagem de contexto respeitando o limite definido;
- citações presentes e correspondentes a documentos recuperados;
- comportamento seguro quando a recuperação não retornar evidência suficiente.

### 9.3 Critérios para considerar o RAG integrado concluído

1. Uma pergunta retorna resposta do modelo e as fontes dos trechos realmente usados.
2. A resposta não inventa `document_id`, data ou fonte.
3. Não há vazamento de contexto entre pacientes em testes negativos.
4. O sistema declara limitação ou ausência de evidência quando apropriado.
5. A base e o índice podem ser regenerados de modo reproduzível.
6. Todos os testes automatizados passam no ambiente documentado.

---

## 10. Configuração, execução e documentação

- Credenciais residem em `.env` ou no ambiente. O arquivo `.env.example` exemplifica as chaves externas. Parâmetros operacionais do RAG residem em `src/config.py`.
- A documentação de execução deve preferir o interpretador do ambiente virtual do projeto quando necessário, por exemplo no PowerShell: `& .\.venv\Scripts\python.exe -m pytest -q`.
- O pipeline de dados é executado por `scripts/Process_dataset.py`; o conjunto de avaliação pode ser regenerado com `--only-eval`.
- Os comandos de geração, ingestão e consulta da base RAG estão em `scripts/hospital_rag.py` e descritos em `README_RAG_DATA.md`.
- Ao regenerar dados RAG, reconstruir o índice ou usar diretório novo de persistência para não misturar chunks antigos e novos.
- `README.md`, `README_RAG_DATA.md`, `ALTERACOES.md` e este arquivo devem permanecer consistentes. Quando houver divergência, o código executável e os artefatos de validação são a fonte factual; a documentação deve ser corrigida na mesma alteração.

---

## 11. Checklist antes de concluir uma alteração

1. O escopo da alteração está claro e respeita SRP/SOLID?
2. Há type hints, docstrings e tratamento específico de erros nas APIs novas?
3. Caminhos e segredos são configuráveis e seguros?
4. Logs são úteis sem expor tokens ou informações sensíveis?
5. Dados novos são sintéticos, anonimizados e rastreáveis?
6. A saída JSON/JSONL ou a resposta RAG foi validada estruturalmente?
7. A resposta RAG está fundamentada somente em contexto recuperado e cita fontes verificáveis?
8. Foram incluídos testes proporcionais ao risco, inclusive isolamento de paciente quando aplicável?
9. A documentação afetada foi atualizada?
