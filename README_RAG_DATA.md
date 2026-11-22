# Base hospitalar sintética para RAG

O clone inclui os **200 pacientes e 1.787 documentos** usados na demonstração. Para consultar essa base, siga direto para a ingestão. A geração interessa a quem quer reproduzir os dados ou criar outra população. Os prontuários são artificiais e destinam-se a exercícios acadêmicos. [Instalação do agente](README.md#instalação) · [Ingestão e consulta](README_RAG.md#ingestão-e-índice)

## Arquivos distribuídos

| Arquivo | Conteúdo |
|---|---|
| [patients.jsonl](data/rag_data/patients.jsonl) | Um cadastro por paciente, identificado por `patient_id` |
| [medical_documents.jsonl](data/rag_data/medical_documents.jsonl) | Registros com ID, paciente, tipo de documento, data, origem, texto e metadados específicos |
| [validation_report.json](data/rag_data/validation_report.json) | Resultado da validação, contagens e distribuição documental |

Cada paciente tem de 6 a 12 documentos, incluindo o cadastro. A distribuição registrada no relatório é:

| Tipo | Documentos |
|---|---:|
| Cadastro (`cadastro`) | 200 |
| Anamnese (`anamnese`) | 484 |
| Exame laboratorial (`exame_laboratorial`) | 484 |
| Laudo de imagem (`laudo_imagem`) | 319 |
| Evolução e plano (`evolucao_plano`) | 300 |
| **Total** | **1.787** |

O vínculo entre arquivos é o `patient_id`. Por exemplo, `ATD-PCT-000001-01` é uma anamnese de `PCT-000001`. O campo `document_date` permite ordenar atendimentos por data e `source` identifica a origem do registro. Os nomes, medidas, faixas e laudos são dados de teste.

## Reproduzir a geração

Use o ambiente Python 3.12 descrito no [README principal](README.md#instalação). A configuração de referência usa 200 pacientes e semente **20261111**. Para comparar uma reprodução com a base versionada, grave em uma pasta separada:

```powershell
python scripts/hospital_rag.py generate --patients 200 --seed 20261111 --output-dir data/rag_reproducao
```

Confira `valid`, `errors`, a distribuição por tipo e os limites por paciente no `validation_report.json` produzido. O gerador verifica IDs duplicados, nomes completos normalizados, vínculo e nome do paciente em cada documento, campos obrigatórios, conteúdo e datas. O comando retorna erro se a validação falhar. [Gerador e validação](scripts/hospital_rag.py)

Para regenerar **no diretório usado pelo agente**, o comando é `python scripts/hospital_rag.py generate`. Ele regrava os arquivos em `data/rag_data/`. Se adotar outra população, configure sua pasta de pacientes e documentos e um diretório Chroma novo em [src/config.py](src/config.py). A ingestão deve usar o mesmo par de arquivos que será consultado pelo agente.

## Conferir a recuperação

Com os JSONL da base de referência e a `.venv` ativa, execute a partir da raiz:

```powershell
python scripts/hospital_rag.py ingest
python scripts/hospital_rag.py stats
python scripts/hospital_rag.py query "Quais sintomas foram relatados?" --patient-name "Emanuel Cunha Cordeiro" --document-type anamnese --retrieve-only
```

Confira se os documentos recuperados pertencem a `PCT-000001` e ao tipo `anamnese`. A busca por nome tenta primeiro a correspondência exata normalizada e depois a parcial. Em caso de ambiguidade, use o código de prontuário. Para entender como esses documentos chegam à resposta do modelo, consulte o [guia de RAG](README_RAG.md).
