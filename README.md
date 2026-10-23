# Contexto
Uma empresa lançou um teste A/B para avaliar a usabilidade do menu lateral de sua plataforma. O teste compara duas configurações:
1. Acesso direto a uma página específica do site (Nível 0 do menu lateral).
2. Acesso à mesma página, mas sob uma aba mais geral (Nível 1 do menu lateral).
O objetivo do desafio a seguir é determinar qual configuração do menu lateral leva a mais acessos e conversões para a página específica da plataforma . Identificar a configuração mais eficaz ajudará a otimizar a navegação do site para aumentar o engajamento dos usuários.

# Base de Dados
A base de dados anexada a este desafio inclui as seguintes colunas:
-user_id: identificador único do usuário
-device: dispositivo do qual o usuário acessou a plataforma
-country: país do qual o usuário acessou a plataforma
-timestamp: data e horário em que o usuário acessou a plataforma
-group: versão do menu lateral (Nível 0 ou Nível 1)
-converted: conversão do usuário para a página de interesse
-session_duration: tempo (s) em que o usuário permaneceu ativo na página de interesse

# Questões
1. Qual configuração do menu lateral levou mais usuários a acessarem a página específica?
2. Qual o tempo médio de sessão para cada configuração do menu?
3. Qual é a taxa de conversão para cada configuração do menu?
4. A diferença nos acessos entre as configurações é estatisticamente significativa?
5. Quais insights podem ser derivados dos dados para otimizar a navegação do site e aumentar o engajamento dos usuários?
6. Existe alguma tendência temporal (comportamento ao longo dia) nos acessos ou conversões para cada configuração do menu?
7. Quais recomendações você faria com base nos dados analisados para otimizar a navegação do site?
