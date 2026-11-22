# PLANEJAMENTO — Agente médico com LangGraph, Llama 3.1 Fine-tuned e RAG hospitalar

## 1. Objetivo

Atue como um arquiteto de software sênior, especializado em Python, LangGraph, LangChain, RAG, Hugging Face Transformers, modelos LLM locais e segurança de aplicações de IA.

Precisamos planejar a implementação de um agente conversacional médico, utilizando LangGraph para orquestrar um modelo Llama 3.1 8B Instruct fine-tuned e um pipeline RAG hospitalar.

**IMPORTANTE: ESTA SOLICITAÇÃO É EXCLUSIVAMENTE DE PLANEJAMENTO.**

Não implemente código, não altere arquivos, não instale dependências e não modifique o ambiente.

Primeiro inspecione a estrutura, os módulos e as implementações já existentes no repositório. Não presuma nomes de classes, interfaces, caminhos ou funcionalidades sem verificar o código.

Ao final, apresente um plano técnico detalhado para minha revisão e aprovação.

---

## 2. Contexto do projeto existente

O projeto acadêmico já possui dois componentes implementados:

### 2.1. Inferência local do Llama 3.1

Existe uma implementação funcional para inferência local de um Llama 3.1 8B Instruct, que passou por fine-tuning com QLoRA utilizando Unsloth.

Características conhecidas:

- Modelo base: Llama 3.1 8B Instruct.
- Fine-tuning com datasets MedQuAD, PubMedQA e dados sintéticos de protocolos hospitalares.
- Treinamento predominantemente em inglês.
- Modelo carregado com quantização 4-bit NF4.
- Adapter LoRA aplicado ao modelo base.
- Inferência com Transformers, PEFT, PyTorch e BitsAndBytes.
- Finalidade: responder perguntas médicas como assistente de suporte a profissionais de saúde.
- Não substituir decisões clínicas ou oferecer diagnósticos definitivos.

A implementação de inferência já existente deve ser inspecionada e, preferencialmente, reutilizada.

Não crie uma segunda implementação de carregamento do modelo sem necessidade técnica justificada.

### 2.2. Pipeline RAG hospitalar

Existe uma implementação funcional de RAG utilizando:

- Python 3.12.
- LangChain.
- ChromaDB.
- Sentence Transformers.
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`.
- Documentos hospitalares sintéticos armazenados em JSONL.
- Base com aproximadamente 1.787 documentos.
- Documentos disponíveis na estrutura `data/rag_data`.
- Pesquisa de pacientes por nome e acesso aos documentos relacionados ao paciente.
- RAG configurado para executar embeddings e recuperação na CPU, sem utilizar GPU.

A implementação atual utiliza a API OpenAI para geração de respostas no pipeline RAG.

Na nova arquitetura, o objetivo é substituir essa geração pelo Llama 3.1 fine-tuned executado localmente, preservando o pipeline de recuperação existente.

Não reimplemente loaders, embeddings, vector stores, retrievers, serviços ou mecanismos de resolução de pacientes se as funcionalidades existentes puderem ser reutilizadas.

Verifique primeiro como o RAG foi organizado e quais interfaces estão disponíveis.

---

## 3. Ambiente de hardware

O agente será executado localmente em Windows 11.

Hardware:

- Processador Intel Core i7-10750H.
- 16 GB RAM.
- NVIDIA RTX 2070 Max-Q.
- 8 GB VRAM.
- CUDA configurado para inferência local.

Nos testes anteriores, o Llama 3.1 quantizado com LoRA apresentou consumo aproximado de 5,5 a 5,8 GB de memória alocada na GPU.

O RAG deve continuar utilizando CPU.

### Análise obrigatória

Avalie a viabilidade de compartilhar uma única instância do Llama para:

1. Classificação e roteamento de perguntas.
2. Respostas médicas gerais.
3. Geração de respostas fundamentadas nos documentos recuperados pelo RAG.

Considere:

- VRAM disponível após carregar modelo e adapter.
- Memória ocupada pelo KV cache.
- Tamanho de contexto.
- Impacto da memória conversacional.
- Quantidade de tokens nos documentos recuperados.
- Latência adicional provocada pelo roteamento.
- Consumo de RAM e CPU.
- Possíveis problemas de execução concorrente.
- Risco de CUDA out-of-memory.
- Compatibilidade com o código existente.

Priorize execução sequencial e reutilização da instância do modelo.

Avalie parâmetros distintos de geração para classificação e respostas médicas.

Caso identifique limitações relevantes, proponha alternativas, mas não introduza outro modelo, API externa ou framework desnecessariamente.

Adendo: Como alternativa, considere utilizar a API da OPENAI para interpretação de RAG somente. A inferência local para do Llamma 3.1 fine-tuned localizado na pasta inference é de extrema importância para resposta de perguntas médicas. 

---

## 4. Orquestração com LangGraph

Proponha uma arquitetura baseada em `StateGraph`, utilizando nós e transições condicionais.

O agente deverá receber uma pergunta e identificar a intenção para determinar qual fluxo executar.

### 4.1. Rota GENERAL_MEDICAL

Utilizada para perguntas sobre conhecimentos médicos gerais, sem necessidade de consultar dados de pacientes.

Exemplos:

- O que é diabetes mellitus tipo 2?
- Quais são as causas de hiperglicemia?
- O que significa TGO elevado?
- Como interpretar os níveis de hemoglobina glicada?
- Quais são os critérios utilizados para avaliação de insuficiência renal?

Comportamento:

- Não consultar o RAG hospitalar desnecessariamente.
- Utilizar o Llama fine-tuned.
- Considerar o histórico conversacional relevante.
- Responder como assistente médico de suporte.
- Não inventar referências científicas ou informações não conhecidas.

### 4.2. Rota PATIENT_RECORD

Utilizada quando a pergunta solicita dados de um paciente específico.

Exemplos:

- Quais foram os últimos exames de Maria Silva?
- Mostre o resultado de creatinina de João Pereira.
- Qual foi a última anamnese registrada para determinado paciente?
- Qual foi a evolução da glicemia desse paciente?
- Considerando os exames anteriores, houve melhora dos resultados?

Comportamento:

1. Identificar que a solicitação depende de dados de prontuário.
2. Resolver a identidade do paciente pelos mecanismos existentes.
3. Tratar homônimos e correspondências ambíguas.
4. Obter o identificador interno correto.
5. Executar a recuperação dos documentos utilizando os mecanismos do RAG.
6. Aplicar restrições de recuperação ao paciente identificado.
7. Enviar o contexto recuperado ao mesmo Llama fine-tuned.
8. Gerar uma resposta fundamentada exclusivamente nos dados recuperados para afirmações específicas sobre o paciente.
9. Permitir explicações médicas complementares, distinguindo claramente conhecimento geral e evidências provenientes do prontuário.
10. Informar ausência de dados quando não houver evidência suficiente.
11. Aplicar guardrails e mascaramento antes da apresentação.

Sempre que tecnicamente possível, incluir referências internas aos documentos utilizados, sem expor identificadores pessoais indevidos.

### 4.3. Rota CLARIFICATION

Utilizada quando o agente não consegue determinar a intenção ou identificar corretamente um paciente.

Exemplos:

- Pergunta com nome de paciente ambíguo.
- Paciente não encontrado.
- Referência a um paciente não identificado.
- Solicitação incompleta ou ambígua.

O agente deve solicitar esclarecimentos, sem assumir identidades ou inventar dados.

### 4.4. Rota SAFETY

Avalie a necessidade de uma rota ou mecanismo transversal de segurança para:

- Solicitações que excedam o papel de assistente médico.
- Emergências médicas.
- Tentativas de revelar informações pessoais.
- Solicitações de diagnóstico ou prescrição definitiva.
- Instruções maliciosas presentes em documentos ou entradas do usuário.
- Situações em que o modelo não possui informações suficientes.

As respostas devem manter o caráter de suporte clínico, com decisão final reservada ao profissional de saúde.

---

## 5. Estratégia de classificação e roteamento

A intenção inicial é utilizar o próprio Llama fine-tuned como classificador.

Avalie a capacidade do modelo para essa função, considerando que ele foi treinado principalmente para responder perguntas médicas, e não especificamente para classificação de intenções.

Compare duas abordagens:

**Abordagem A — Roteamento exclusivamente por LLM**

O modelo recebe a pergunta e produz uma intenção estruturada.

**Abordagem B — Roteamento híbrido**

Regras determinísticas identificam situações inequívocas, enquanto o Llama classifica situações que exigem interpretação semântica.

Minha preferência inicial é a abordagem híbrida, desde que não acrescente complexidade injustificada.

Investigue a possibilidade de usar respostas estruturadas e validadas, por exemplo:

```json
{
  "intent": "PATIENT_RECORD",
  "requires_rag": true,
  "patient_reference": null,
  "needs_clarification": false
}
```

Essa estrutura é apenas ilustrativa. Proponha o schema definitivo de acordo com a arquitetura.

Requisitos:

- Temperatura baixa ou geração determinística para classificação.
- Validação do formato produzido.
- Tratamento de respostas inválidas.
- Estratégia segura de fallback.
- Resolução de referências contextuais.
- Não confiar cegamente em níveis de confiança declarados pelo próprio LLM.
- Evitar chamar o RAG para perguntas gerais.
- Nunca responder com dados de um paciente diferente do solicitado.

**Importante:** não assuma que o wrapper atual do modelo suporta `bind_tools()` ou `with_structured_output()`. Inspecione a implementação e determine como obter saída estruturada de forma compatível.

---

## 6. Memória conversacional

O agente deve funcionar como um chat contínuo.

Exemplo:

Usuário: Quais são as principais causas de diabetes tipo 2?

Agente: Resposta médica geral.

Usuário: E quais exames são utilizados para diagnóstico?

Agente: Deve compreender que a pergunta continua relacionada ao assunto anterior.

Outro exemplo:

Usuário: Quais são os últimos resultados de glicemia do paciente João Silva?

Agente: Recupera os dados do paciente pelo RAG.

Usuário: E como estava a creatinina dele?

Agente: Deve compreender que "dele" se refere ao paciente João Silva anteriormente identificado.

Usuário: Esses resultados indicam piora?

Agente: Deve avaliar a pergunta utilizando os resultados e contexto pertinentes, sem inventar dados e sem apresentar diagnóstico definitivo.

### Requisitos de memória

Avalie os mecanismos atuais do LangGraph para memória conversacional e persistência.

Considere:

- Checkpointers.
- Estado por `thread_id`.
- Histórico de mensagens.
- Paciente atualmente selecionado.
- Última intenção identificada.
- Resumos de conversas quando necessários.
- Limites de tamanho do contexto.
- Estratégias de truncamento ou sumarização.
- Separação de contextos entre conversas.
- Evitar transportar documentos completos em todos os turnos.
- Limpeza explícita de contexto quando o usuário solicitar.

A implementação inicial poderá manter memória apenas durante a execução do processo CLI.

Avalie `InMemorySaver` como primeira opção.

Apresente `SqliteSaver` como possível evolução caso seja necessário recuperar conversas após reinicializações.

Não introduza armazenamento persistente de prontuários ou informações pessoais sem necessidade e justificativa.

O estado deve preservar a identidade interna do paciente apenas quando necessário e evitar vazamento de dados entre sessões.

---

## 7. Guardrails e proteção de dados pessoais

Implemente no planejamento uma estratégia de proteção de dados em múltiplas camadas.

Apesar de a base ser sintética, a arquitetura deve considerar boas práticas de proteção de informações pessoais e médicas.

### 7.1. Categorias de informações sensíveis

Avaliar detecção e mascaramento de:

- Nome completo do paciente.
- CPF e RG.
- Data de nascimento completa.
- Endereço residencial.
- CEP.
- Cidade, bairro e demais informações geográficas identificáveis.
- Telefone.
- E-mail.
- Outros identificadores pessoais que possam existir nos documentos.

A base atual parece conter poucos dados de endereço, possivelmente apenas a cidade, mas não presuma isso sem inspecioná-la.

Investigue os campos, metadados e documentos disponíveis.

### 7.2. Estratégia de mascaramento

O mascaramento não pode impedir o funcionamento da busca por pacientes.

Por exemplo, se o usuário perguntar pelos exames de João Silva, o nome deverá poder ser utilizado pelo mecanismo interno de identificação.

Depois da resolução, utilize preferencialmente o identificador interno do paciente para consultar os documentos.

Avalie pseudonimização e mascaramento na saída, nos logs e nos dados de memória, preservando os dados clínicos necessários à interpretação.

Exemplo de resposta mascarada:

"Paciente com CPF: [CPF]. A última glicemia registrada foi de 112 mg/dL."

Não exponha endereço, CPF ou outros identificadores sem uma necessidade funcional explícita e uma política de acesso apropriada.

Diferencie informações identificáveis de informações clínicas: o objetivo não é remover resultados de exames das respostas.

Não há necessidade de mascarar o nome, já que este deve ser de conhecimento do profissional médico.

### 7.3. Guardrails em múltiplos pontos

Planeje verificações:

1. Na entrada do usuário.
2. Na identificação do paciente.
3. Antes da montagem do contexto para o modelo.
4. Na recuperação dos documentos.
5. Na resposta gerada pelo LLM.
6. Na apresentação da resposta pela CLI.
7. Nos logs, traces e checkpoints da memória.

Avalie recursos existentes do LangChain para PII e detectores personalizados, incluindo padrões brasileiros.

Não confie somente no prompt do LLM para proteger dados pessoais.

A filtragem determinística deve continuar funcionando mesmo se o LLM não obedecer às instruções.

### 7.4. Segurança do RAG

Avalie proteções contra:

- Prompt injection em documentos recuperados.
- Recuperação de documentos de pacientes diferentes.
- Exposição acidental de dados em mensagens de erro.
- Vazamento de informação em logs de depuração.
- Consultas que tentem contornar restrições de acesso.
- Mistura de dados entre diferentes conversas.

Identifique os limites da proteção possível em um protótipo acadêmico e quais mecanismos adicionais seriam necessários em um sistema com pacientes reais, incluindo autenticação e autorização.

---

## 8. Aplicação CLI interativa

O agente será executado inicialmente pelo terminal do Windows/VS Code.

Não haverá interface Web nesta etapa.

### Comportamento desejado

Ao iniciar:

- Carregar configurações necessárias.
- Inicializar o modelo fine-tuned uma única vez.
- Inicializar ou conectar aos serviços RAG existentes.
- Preparar o grafo LangGraph e o sistema de memória.
- Exibir uma mensagem indicando que o assistente está pronto.

Depois:

1. Aguardar uma pergunta.
2. Processar a entrada.
3. Roteá-la.
4. Executar o fluxo correspondente.
5. Apresentar a resposta.
6. Aguardar a próxima pergunta.
7. Preservar o contexto entre perguntas da mesma sessão.

Comandos desejados:

- `/sair`: encerrar a aplicação.
- `/limpar`: limpar memória e contexto atual.
- `/nova`: iniciar uma nova conversa isolada.
- `/ajuda`: apresentar comandos disponíveis.

Avalie também tratamento adequado para `Ctrl+C`, `EOF`, entradas vazias e erros de inferência.

O encerramento deve liberar os recursos utilizados sem deixar processos desnecessários.

Planeje logs técnicos configuráveis, sem exposição de dados sensíveis.

Não implemente uma interface gráfica.

---

## 9. Integração com os componentes existentes

Inspecione os seguintes pontos antes de propor a arquitetura:

### Inferência local

- Como o modelo é carregado.
- Como o adapter LoRA é aplicado.
- Onde o tokenizer é inicializado.
- Como os prompts são construídos.
- Como o chat template é aplicado.
- Como os tokens são gerados e decodificados.
- Quais parâmetros de geração estão disponíveis.
- Como o dispositivo CUDA é selecionado.
- Como a memória da GPU é gerenciada.
- Como o histórico atual é manipulado, caso exista.

Defina a melhor forma de disponibilizar uma interface reutilizável para o LangGraph, preservando a implementação atual.

### RAG

- Como os documentos JSONL são carregados.
- Como o ChromaDB é inicializado.
- Como os embeddings são executados na CPU.
- Como o retriever funciona.
- Como os nomes de pacientes são pesquisados.
- Como os identificadores internos são recuperados.
- Como filtros por paciente são aplicados.
- Como o contexto é montado.
- Como a geração via OpenAI está integrada.
- Como substituir a geração via OpenAI pelo Llama local sem prejudicar a recuperação.
- Como são realizados os testes existentes.

Evite múltiplos carregamentos do modelo, reconstruções desnecessárias do índice vetorial e duplicação de lógica de negócio.

### Estrutura do repositório

Leia os documentos de orientação existentes, inclusive `AGENTS.md` e `Codex.md`, quando presentes.

Inspecione `requirements.txt`, configurações, módulos, scripts, testes e documentação.

Preserve a organização existente sempre que possível.

---

## 10. Avaliação da arquitetura

Compare as alternativas de implementação e justifique as escolhas.

Inclua análise de:

- Grafo explícito com `StateGraph` versus agente com chamadas dinâmicas de ferramentas.
- Roteamento determinístico, por LLM ou híbrido.
- Reutilização de uma única instância do modelo.
- Integração do Llama existente via adaptador ou wrapper.
- Compartilhamento de estado entre os nós.
- Memória em RAM versus persistência local.
- Guardrails determinísticos versus verificações assistidas por LLM.
- Tratamento de perguntas em português com modelo fine-tuned em inglês.
- Estratégia para preservar a performance do hardware disponível.

Priorize a solução mais simples que atenda aos requisitos, sem sacrificar segurança, modularidade e capacidade de manutenção.

---

## 11. Plano obrigatório de testes

O planejamento deve contemplar testes automatizados com `pytest`, preferencialmente reutilizando a estrutura existente.

### Testes de roteamento

Criar uma matriz de casos com resultados esperados para:

- Perguntas médicas gerais.
- Perguntas que mencionam pacientes específicos.
- Perguntas ambíguas.
- Referências a pacientes mencionados anteriormente.
- Mudanças explícitas de paciente durante a conversa.
- Perguntas sobre exames.
- Perguntas hipotéticas que não exigem RAG.
- Perguntas em português e inglês.
- Perguntas que exigem combinar prontuário e explicações médicas gerais.

Proponha métricas de acerto, incluindo falsos negativos de roteamento para RAG.

### Testes de RAG

Validar:

- Identificação correta do paciente.
- Tratamento de homônimos.
- Busca por identificador interno.
- Isolamento entre pacientes.
- Recuperação dos documentos relevantes.
- Tratamento de pacientes inexistentes.
- Respostas sem dados suficientes.
- Fidelidade das respostas aos documentos recuperados.

### Testes de memória

Validar:

- Contexto entre turnos consecutivos.
- Resolução de pronomes e referências.
- Persistência do paciente ativo.
- Mudança de paciente.
- Limpeza da memória.
- Isolamento entre `thread_id`.
- Controle de tamanho do histórico.

### Testes de guardrails

Validar mascaramento de:

- CPF.
- RG.
- E-mail.
- Telefones.
- CEP.
- Endereços.
- Cidades e outros dados pessoais relevantes.

Incluir testes com ataques de prompt injection, dados sensíveis no histórico e informações identificáveis em documentos recuperados.

### Testes de desempenho

Propor medições de:

- Tempo de carregamento do modelo.
- Latência de classificação.
- Latência de recuperação RAG.
- Latência de geração de resposta.
- Tempo total por pergunta.
- Memória de GPU em cada etapa.
- Memória RAM utilizada.
- Comportamento após múltiplos turnos.
- Estabilidade em conversas prolongadas.

Utilizar mocks para testes unitários sempre que possível, reservando a GPU para testes de integração que realmente dependam do modelo.

---

## 12. Entregáveis do planejamento

Ao concluir a análise, apresente o resultado organizado nas seguintes seções:

### A. Diagnóstico do projeto atual

- Estrutura de pastas relevante encontrada.
- Componentes existentes.
- Classes e funções reutilizáveis.
- Interfaces disponíveis.
- Dependências relevantes e versões verificadas.
- Problemas identificados.
- Lacunas que precisam ser implementadas.
- Evidências de cada conclusão, indicando arquivos e funções analisados.

### B. Arquitetura proposta

- Diagrama de fluxo em Mermaid.
- Definição dos nós do LangGraph.
- Definição das transições condicionais.
- Estrutura do estado compartilhado.
- Estratégia de roteamento.
- Integração com o RAG.
- Integração com a inferência local.
- Estratégia de memória.
- Estratégia de proteção de dados.
- Tratamento de erros e situações ambíguas.

### C. Viabilidade técnica

- Capacidade estimada do Llama para roteamento e interpretação do RAG.
- Riscos decorrentes do fine-tuning existente.
- Impacto estimado sobre VRAM, RAM e CPU.
- Possíveis gargalos de performance.
- Alternativas em caso de limitações.
- Testes necessários para confirmar as estimativas.

Diferencie fatos observados no repositório de hipóteses que precisarão de benchmarks.

### D. Plano de implementação por fases

Divida a implementação em fases incrementais e verificáveis.

Para cada fase, apresente:

1. Objetivo.
2. Funcionalidades.
3. Arquivos existentes que serão alterados.
4. Novos arquivos necessários.
5. Componentes reutilizados.
6. Dependências de outras fases.
7. Principais decisões técnicas.
8. Riscos.
9. Testes.
10. Critérios objetivos de aceitação.

Priorize uma sequência que permita validar a integração do modelo local com o LangGraph antes de adicionar funcionalidades mais complexas.

### E. Decisões pendentes

Liste as decisões que precisam da minha aprovação antes da implementação.

Recomende a alternativa que considera mais adequada para cada decisão, com justificativa.

---

## 13. Restrições finais

- Não implemente código nesta etapa.
- Não modifique nenhum arquivo, incluindo documentação.
- Não atualize `requirements.txt`.
- Não instale bibliotecas.
- Não altere o banco ChromaDB nem os datasets.
- Não reconstrua índices vetoriais.
- Não inicie processos de treinamento.
- Não faça mudanças no modelo fine-tuned.
- Não exponha segredos ou dados sensíveis.
- Não substitua componentes existentes sem justificativa técnica.
- Não proponha serviços externos quando a execução local for suficiente.
- Utilize APIs atuais e compatíveis com as versões instaladas.
- Caso precise verificar versões ou documentação externa, diferencie essas informações das características efetivamente observadas no projeto.
- Se alguma informação não puder ser verificada, registre explicitamente a incerteza.

**Resultado esperado: um plano arquitetural detalhado, acompanhado do diagnóstico do código existente e de uma proposta de implementação incremental, aguardando minha aprovação antes de executar qualquer alteração.**