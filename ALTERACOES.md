# Alterações no pipeline de dados – Fase 3

## O que mudou

O pipeline agora junta **três fontes**, e não só o MedQuAD, e cobre os itens do enunciado sobre preparação dos dados (preprocessing, anonimização e curadoria).

| Fonte | Volume aproximado | Para quê |
|---|---|---|
| MedQuAD (`lavita/MedQuAD`) | ~16,4 mil após limpeza | Perguntas e respostas de saúde do NIH |
| PubMedQA `pqa_artificial` (`qiaojin/PubMedQA`) | 10 mil (amostra configurável) | Perguntas clínicas com evidência de artigo, dataset sugerido no enunciado |
| Base sintética do hospital | ~660 | Protocolos internos, FAQ de médicos, casos clínicos, modelos de laudo, minutas de receita e exemplos de segurança |
| PubMedQA `pqa_labeled` | ~1 mil | **Só avaliação** (antes/depois do fine-tuning), fora do treino |

Total esperado para treino + validação: **~27 mil exemplos**.

### Arquivos novos
- `src/anonymizer.py`: troca nome, CPF, RG, CNS, prontuário, telefone, e-mail, CEP e datas por marcadores padronizados em inglês (`[NAME]`, `[CPF]`, `[DATE]`, `[TELEPHONE]`, `[MRN]`, `[EMAIL]`, `[ZIP_CODE]`). Também conta as substituições no log, para auditoria.
- `src/curation.py`: normaliza espaços, aplica filtros de tamanho e remove duplicatas exatas. Cada descarte é salvo com o motivo em `discarded_curation.json`.
- `src/synthetic_generator.py`: gera a base sintética do hospital por templates em inglês com dados e PII de pacientes brasileiros (regras da LGPD), **sem LLM e sem gastar créditos**. São 5 protocolos fictícios (sepse, dor torácica/SCA, hipoglicemia, AVC, crise hipertensiva), com casos clínicos cuja resposta é calculada a partir dos dados do caso (qSOFA, janela de trombólise, volume por kg etc.).
- `tests/test_new_modules.py`: 7 testes novos.

### Arquivos alterados
- `src/config.py`: liga/desliga cada fonte, define o tamanho da amostra do PubMedQA, os filtros de curadoria e os caminhos novos. O `SYSTEM_PROMPT` agora pede para citar a fonte, responder no idioma da pergunta e nunca prescrever sem validação médica.
- `src/data_loader.py`: a limpeza foi para uma classe base (`BaseQADataLoader`). O `MedQuadDataLoader` continua igual para os testes antigos e ganhou `include_metadata=True` para trazer a fonte do documento. Também foi adicionado o `PubMedQADataLoader`.
- `scripts/Process_dataset.py`: novo fluxo `carregar → anonimizar → curar → converter → separar → exportar → validar`.

## Explainability e segurança já no treino
- Toda resposta termina com a **fonte**: `Source: MedQuAD (NIH) – <url>`, `Source: PubMed PMID <id>` ou `Source: Internal protocol HOSP-PROT-00X`. Assim o modelo aprende a citar.
- As respostas sintéticas terminam com o aviso de **validação médica obrigatória**.
- Há exemplos em que o assistente **recusa** prescrever, dar alta ou fechar diagnóstico sem um médico.

## Arquivos gerados (`data/`)
- `processed/finetuning_train.jsonl` e `processed/finetuning_val.jsonl`: mesmo formato de antes (system/user/assistant).
- `processed/eval_pubmedqa_labeled.jsonl`: conjunto de avaliação, com `expected_decision` (yes/no/maybe).
- `processed/dataset_summary.json`: contagem por fonte e por split, útil para o relatório.
- `processed/dataset_manifest.csv`: fonte de cada linha do treino e da validação.
- `raw/synthetic_hospital_raw.jsonl`: base sintética **antes** da anonimização, com dados pessoais fictícios, para mostrar o antes e depois no vídeo.
- `processed/discarded*.json`: auditoria dos descartes.

## Como rodar
```bash
pip install datasets pandas python-dotenv pytest
python -m pytest -q
python scripts/Process_dataset.py
```
Para um teste rápido, reduza a amostra. No PowerShell: `$env:PUBMEDQA_TRAIN_SAMPLE_SIZE="1000"`, depois `python scripts/Process_dataset.py`.

Para gerar **só** o conjunto de avaliação (sem refazer o treino): `python scripts/Process_dataset.py --only-eval`.

Os downloads do PubMedQA tentam até 3 vezes, com espera crescente. Se o conjunto de avaliação falhar, o pipeline termina do mesmo jeito, porque os arquivos de treino e validação já foram gerados e validados. Recomenda-se colocar `HF_TOKEN=<seu token>` no `.env`, já que requisições sem autenticação têm limite menor no Hugging Face.

## Pontos de atenção
- Todas as três fontes (`MedQuAD`, `PubMedQA` e a base sintética do hospital) agora estão unificadas em **inglês**, evitando desbalanceamento linguístico no fine-tuning. Os dados de pacientes da base sintética mantêm nomes, CPFs e formatos nacionais brasileiros para validar adequadamente o módulo de anonimização (LGPD).
- Os protocolos sintéticos são **fictícios** e servem só para fins acadêmicos.
- A anonimização por regex não reconhece qualquer nome em texto livre. No relatório, vale citar que em produção se usaria NER (por exemplo, Presidio com spaCy) mais revisão humana.

## Notebook de Fine-Tuning no Google Colab (`notebooks/colab_finetuning.ipynb`)
- Criado notebook completo para execução de **Fine-Tuning com Unsloth (4-bit QLoRA)** no Google Colab.
- Conexão e sincronização direta com o Google Drive (`/Tech-Challenge_dataset/`).
- Treinamento supervisionado com Llama 3.1 8B Instruct e masking de resposta (`train_on_responses_only`).
- Validação contínua (curvas de loss), testes inferenciais clínicos e checagem comparativa com o *ground truth*.
- Salvamento no Drive e upload autenticado para o **Hugging Face Hub** (`llama-lora-medical-tc3`) com guia de consumo via API/Transformers.
