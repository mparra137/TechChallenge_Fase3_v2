# Base hospitalar sintética para RAG

## Conteúdo
- `scripts/hospital_rag.py`: gerador, validação, exportação JSONL e pipeline LangChain + Chroma.
- `requirements.txt`: dependências da ingestão vetorial.
- Os dados são criados na pasta `data/rag_data/` ao executar o gerador (configurado em `src/config.py`).

## Instalação
Recomendado: Python 3.10 ou superior, em ambiente virtual.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Gerar 200 pacientes
```bash
python scripts/hospital_rag.py generate
```

Saídas:
- `data/rag_data/patients.jsonl`
- `data/rag_data/medical_documents.jsonl`
- `data/rag_data/validation_report.json`

A densidade média gera de 6 a 12 documentos por paciente, contando o cadastro (total aproximado de 1.200 a 2.400 documentos clínicos sintéticos).

## Ingerir no Chroma
```bash
python scripts/hospital_rag.py ingest
```

O modelo de embeddings multilíngue é baixado na primeira execução. A execução padrão usa CPU.

## Consultar
```bash
python scripts/hospital_rag.py query "Qual foi a hemoglobina glicada?" --patient-id PCT-000001
python scripts/hospital_rag.py query "Quais foram os últimos exames?" --patient-name "Alexia Vieira Menezes"
python scripts/hospital_rag.py query "Quais exames foram realizados?" --patient-id PCT-000001 --document-type exame_laboratorial
```

## Observações importantes
- Todo o conteúdo é artificial e serve somente para testes acadêmicos e de engenharia de RAG.
- Nomes fictícios podem coincidir incidentalmente com nomes de pessoas reais; não há intenção de representar indivíduos.
- Resultados, faixas de referência, hipóteses e laudos são ilustrativos, não clinicamente validados.
- O filtro por paciente é demonstrativo. Em aplicações reais, implemente autorização no servidor e isolamento de dados por usuário/tenant; não confie em um `patient_id` enviado livremente pelo cliente.
- Para respostas com citações, preserve `document_id`, `patient_id`, `document_date` e `source` nos metadados recuperados.
- A busca vetorial não substitui consultas SQL para valores numéricos, intervalos de datas ou cálculos longitudinais. Considere uma camada híbrida (SQL + busca vetorial).


## Pesquisa por nome do paciente
Todos os documentos carregam `patient_name` nos metadados. A opção
`--patient-name` resolve o nome no `patients.jsonl` e filtra os chunks pelo
`patient_id` correspondente. A busca tenta primeiro correspondência exata
(normalizada sem distinção de acentos/caixa) e depois parcial. Se houver
mais de um candidato, o sistema não escolhe automaticamente: solicita
desambiguação. A geração garante nomes completos únicos nesta base.

Exemplo:
```bash
python hospital_rag.py query "Quais foram os resultados laboratoriais?" --patient-name "Alexia Vieira Menezes"
```

Depois de regenerar os JSONL, execute novamente `ingest`. Para evitar misturar
chunks antigos com novos, utilize uma pasta Chroma vazia ou um novo diretório
de persistência.
