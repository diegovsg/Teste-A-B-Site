import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega a base de dados em excel para um DataFrame."""
    return pd.read_excel(caminho_arquivo)

def limpar_dados_sessao(df: pd.DataFrame) -> pd.DataFrame:
    """Remove tempos nulos ou zerados que distorcem o tempo de permanência."""
    df_limpo = df.dropna(subset=['session_duration']).copy()
    return df_limpo[df_limpo['session_duration'] > 0]

def calcular_acessos(df: pd.DataFrame) -> pd.Series:
    """Q1: Retorna a quantidade total de acessos por grupo."""
    return df['group'].value_counts()

def calcular_estatisticas_tempo(df: pd.DataFrame) -> pd.DataFrame:
    """Q2: Calcula N, Média, Mediana e Desvio Padrão do tempo de sessão."""
    return df.groupby('group')['session_duration'].agg(
        N_Acessos='count', Media='mean', Mediana='median', Desvio_Padrao='std'
    ).round(2)

def calcular_taxa_conversao(df: pd.DataFrame) -> pd.Series:
    """Q3: Calcula a taxa de conversão (%) por grupo."""
    return (df.groupby('group')['converted'].mean() * 100).round(2)

def testar_significancia_conversao(df: pd.DataFrame) -> dict:
    """
    Q4: Teste Qui-Quadrado para significância da conversão entre grupos.
    
    Hipótese Nula (H0): Não há diferença na taxa de conversão entre Nível 0 e Nível 1.
    Hipótese Alternativa (H1): Há diferença significativa na taxa de conversão entre os grupos.
    """
    tabela = pd.crosstab(df['group'], df['converted'])
    chi2, p_valor, _, _ = chi2_contingency(tabela)
    significativo = p_valor < 0.05
    return {
        'Estatistica_Chi2': round(chi2, 4), 
        'P_valor': p_valor, 
        'Significativo': significativo,
        'Mensagem': "A diferença É estatisticamente significativa (Rejeita-se H0)." if significativo else "A diferença NÃO É estatisticamente significativa (Falha em rejeitar H0)."
    }

def analisar_por_dispositivo(df: pd.DataFrame) -> pd.DataFrame:
    """Q5/Q7: Analisa acessos e taxa de conversão cruzando Device e Grupo."""
    stats = df.groupby(['device', 'group']).agg(
        Acessos=('user_id', 'count'),
        Taxa_Conversao_Pct=('converted', lambda x: x.mean() * 100)
    ).round(2).reset_index()
    return stats

def analisar_por_pais(df: pd.DataFrame) -> pd.DataFrame:
    """Q5/Q7: Analisa acessos e taxa de conversão cruzando Country e Grupo."""
    stats = df.groupby(['country', 'group']).agg(
        Acessos=('user_id', 'count'),
        Taxa_Conversao_Pct=('converted', lambda x: x.mean() * 100)
    ).round(2).reset_index()
    return stats

def imprimir_insights_e_recomendacoes():
    """Q5 e Q7: Imprime os insights e recomendações de negócio derivados da análise."""
    print("\n--- Q5: Insights Derivados ---")
    print("- Fricção destrói a conversão: A diferença gritante de desempenho prova que esconder ferramentas ou páginas cruciais dentro de abas gerais (Nível 1) cria uma barreira de navegação. Apontar de forma direta (Nível 0) facilita o clique e retém o usuário na página.")
    print("- Comportamento por Dispositivo: A usabilidade do menu não foi afetada pelo tamanho da tela, pois o Nível 0 venceu de forma unânime no Desktop (45%), Tablet (42,9%) e Smartphone (47,5%). O insight principal aqui é que o Smartphone é o dispositivo que gera a maior taxa de conversão em ambas as configurações.")
    print("- Problemas Regionais Severos: Ao cruzar os grupos com os países, nota-se que o Nível 0 lidera globalmente. Contudo, o Nível 1 teve um desempenho catastrófico na Alemanha, com apenas 3,57% de conversão. Isso é um forte indício de que o rótulo/texto usado na \"aba geral\" do Nível 1 pode ter uma tradução ruim ou confusa no idioma alemão, impedindo os usuários de encontrar a página.")
    print("- O \"Horário Nobre\" da Plataforma: Existe uma janela clara de alto engajamento em horário comercial matinal (07h às 11h), que é onde a plataforma tem maior facilidade de atrair o clique se o acesso for direto.")
    
    print("\n--- Q7: Recomendações para Otimização ---")
    print("- Implementação Imediata do Nível 0: Descontinuar a configuração de Nível 1. A página específica deve ter acesso direto (Nível 0) no menu lateral para 100% da base de usuários, dado seu desempenho estatisticamente superior na conversão.")
    print("- Design focado no Mobile: Como os Smartphones apresentam as melhores taxas de conversão (47,55%), a equipe de UI/UX deve priorizar a responsividade do menu lateral de acesso direto para telas menores, garantindo botões de fácil clique (área de toque adequada).")
    print("- Auditoria de Localização (Traduções): A equipe deve revisar como as categorias do menu estão sendo traduzidas, com atenção redobrada ao mercado alemão, para garantir que os termos de navegação façam sentido para a cultura local.")
    print("- Ações de Marketing Sincronizadas: Utilizar a tendência temporal a favor do negócio. Campanhas, notificações push ou mensagens importantes no sistema devem ser disparadas prioritariamente entre as 09h e 11h da manhã, horário em que os usuários do Nível 0 estão mais propensos a clicar e converter.")

def plotar_tendencias_temporais(df: pd.DataFrame):
    """Q6: Comportamento de acessos e conversões ao longo do dia com eixos independentes."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    
    acessos_hora = df.groupby(['hour', 'group']).size().reset_index(name='acessos')
    conv_hora = df.groupby(['hour', 'group'])['converted'].mean().reset_index(name='taxa_conversao')
    conv_hora['taxa_conversao'] *= 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    sns.lineplot(data=acessos_hora, x='hour', y='acessos', hue='group', marker='o', palette='Set2', ax=ax1)
    ax1.set_title('Tendência Temporal: Acessos por Hora do Dia')
    ax1.set_ylabel('Quantidade de Acessos')
    ax1.set_xlabel('Hora do Dia (0-23)')
    ax1.set_xticks(range(0, 24))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    sns.lineplot(data=conv_hora, x='hour', y='taxa_conversao', hue='group', marker='o', palette='Set2', ax=ax2)
    ax2.set_title('Tendência Temporal: Taxa de Conversão por Hora do Dia (%)')
    ax2.set_xlabel('Hora do Dia (0-23)')
    ax2.set_ylabel('Taxa de Conversão (%)')
    ax2.set_xticks(range(0, 24))
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # Carregamento
    df_bruto = carregar_dados('ab_test_data 1.xlsx')
    df_limpo = limpar_dados_sessao(df_bruto)
    
    # Execuções
    print("--- Q1: Acessos ---")
    print(calcular_acessos(df_bruto))
    
    print("\n--- Q2: Tempo de Sessão (Base Limpa) ---")
    print(calcular_estatisticas_tempo(df_limpo))
    
    print("\n--- Q3: Conversão (%) ---")
    print(calcular_taxa_conversao(df_bruto))
    
    print("\n--- Q4: Teste Estatístico (Conversão) ---")
    res_estat = testar_significancia_conversao(df_bruto)
    print(f"Hipótese Nula (H0): As taxas de conversão são iguais entre os grupos.")
    print(f"Estatística de Teste (Chi2): {res_estat['Estatistica_Chi2']}")
    print(f"P-valor: {res_estat['P_valor']}")
    print(f"Conclusão: {res_estat['Mensagem']}")
    
    print("\n--- Analise por Dispositivo ---")
    print(analisar_por_dispositivo(df_bruto))
    
    print("\n--- Analise por País ---")
    print(analisar_por_pais(df_bruto))
    
    # Imprimir as repostas para Q5 e Q7
    imprimir_insights_e_recomendacoes()
    
    # Exibir gráficos finais da Q6
    plotar_tendencias_temporais(df_bruto)