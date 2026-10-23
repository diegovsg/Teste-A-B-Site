import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega a base de dados em excel e retorna um DataFrame do Pandas."""
    return pd.read_excel(caminho_arquivo)

def limpar_dados_sessao(df: pd.DataFrame) -> pd.DataFrame:
    """Remove registros nulos ou com tempo zero na duração da sessão."""
    df_limpo = df.dropna(subset=['session_duration']).copy()
    return df_limpo[df_limpo['session_duration'] > 0]

def calcular_acessos(df: pd.DataFrame) -> pd.Series:
    """Q1: Calcula o total de acessos por grupo."""
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
    Q4: Realiza o Teste Qui-Quadrado para validar se a diferença na 
    conversão entre os grupos é estatisticamente significativa.
    """
    tabela = pd.crosstab(df['group'], df['converted'])
    chi2, p_valor, _, _ = chi2_contingency(tabela)
    significativo = p_valor < 0.05
    return {
        'Estatistica_Chi2': round(chi2, 4), 
        'P_valor': p_valor, 
        'Significativo': significativo,
        'Mensagem': "A diferença É significativa (Rejeita H0)." if significativo else "A diferença NÃO É significativa (Falha ao rejeitar H0)."
    }

def analisar_por_dispositivo(df: pd.DataFrame) -> pd.DataFrame:
    """Q5/Q7: Agrupa acessos e conversão por dispositivo e grupo."""
    return df.groupby(['device', 'group']).agg(
        Acessos=('user_id', 'count'), Taxa_Conversao_Pct=('converted', lambda x: x.mean() * 100)
    ).round(2).reset_index()

def analisar_por_pais(df: pd.DataFrame) -> pd.DataFrame:
    """Q5/Q7: Agrupa acessos e conversão por país e grupo."""
    return df.groupby(['country', 'group']).agg(
        Acessos=('user_id', 'count'), Taxa_Conversao_Pct=('converted', lambda x: x.mean() * 100)
    ).round(2).reset_index()

def analisar_casos_vazios_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Audita o comportamento de usuários com tempo de sessão = 0 ou Nulo (Rejeição)."""
    df_zero_null = df[(df['session_duration'].isnull()) | (df['session_duration'] == 0)]
    
    stats = df.groupby('group').agg(Total_Acessos=('user_id', 'count'))
    stats['Acessos_Zero_Nulo'] = df_zero_null.groupby('group')['user_id'].count()
    stats['Taxa_Rejeicao_Pct'] = (stats['Acessos_Zero_Nulo'] / stats['Total_Acessos'] * 100).round(2)
    stats['Conversoes_Zero_Nulo'] = df_zero_null.groupby('group')['converted'].sum()
    
    return stats

def imprimir_insights_e_recomendacoes():
    print("\n--- Q5: Insights Derivados ---")
    print("- A diferença entre as duas opções é perceptivel e é vista com a estatística, quando percebemos que sim existe uma diferença de desempenho entre as duas opções de ferramentas, demonstrando que esconder ferramentas dentro de abas gerais (Nível 1) criou uma obstaculo, logo apontar de a página de forma direta (Nível 0) facilita o clique e retém o usuário na página, defivido a alta taxa de Rejeição no Nível 1, porque analisando os dados vazios e zerados, nota-se que 83,56% dos usuários do Nível 1 abandonam a sessão instantaneamente (0 segundos, contra 57% do Nível 0 demonstrando que o menu aninhado afasta o usuário imediatamente. Acho interessante ressaltar o comportamento por Dispositivo: O Smartphone é o dispositivo que gera a maior taxa de conversão em ambas as configurações (47,5% no Nível 0). Problemas Regionais: O Nível 1 teve um desempenho super abaixo na Alemanha (3,57%). Forte indício de algo atrapalhando a aba geral no idioma alemão, mas que seguindo o nível 0 ainda foi superior em todos os paises. Horário de atividade: Existe uma janela clara de alto engajamento em horário comercial matinal (07h às 11h).")
    
    print("\n--- Q7: Recomendações para Otimização ---")
    print("- Implementação definitiva do Acesso direto a uma página específica do site (Nível 0): Descontinuar o acesso à mesma página, mas sob uma aba mais geral (Nível 1). A página deve ter acesso direto (Nível 0) para 100% da base, dado seu desempenho superior.")
    print("- Auditoria de Tracking (Coleta de Dados): Investigar por que 36 usuários conseguiram registrar uma 'conversão' tendo 0 segundos ou tempo nulo de sessão. Pode ser um erro no disparo do script de tempo ou conversões via abertura de novas guias.")
    print("- Design focado no Mobile: Priorizar a responsividade do menu lateral para telas menores (área de toque adequada). Focado no tipo acesso direto a uma página específica do site")
    print("- Ações de Marketing Sincronizadas: Disparar notificações ou campanhas prioritariamente entre 09h e 11h da manhã.")

def plotar_distribuicao_sessao(df_limpo: pd.DataFrame):
    """Q2 (Visual): Plota o histograma de distribuição do tempo de sessão."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_limpo, x='session_duration', hue='group', kde=True, bins=30, palette='Set2', alpha=0.6)
    plt.title('Distribuição do Tempo de Sessão por Grupo (Nível 0 vs Nível 1)')
    plt.xlabel('Duração da Sessão (segundos)')
    plt.ylabel('Quantidade de Acessos')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plotar_tendencias_temporais(df: pd.DataFrame):
    """Q6: Plota o volume de acessos e a taxa de conversão hora a hora."""
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
    ax2.set_title('Tendência Temporal: Taxa de Conversão por Hora (%)')
    ax2.set_xlabel('Hora do Dia (0-23)')
    ax2.set_ylabel('Taxa de Conversão (%)')
    ax2.set_xticks(range(0, 24))
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. Carregamento e Limpeza
    df_bruto = carregar_dados('ab_test_data 1.xlsx')
    df_limpo = limpar_dados_sessao(df_bruto)
    
    # 2. Impressão dos Resultados Analíticos
    print("--- Q1: Acessos ---")
    print(calcular_acessos(df_bruto))
    
    print("\n--- Q2: Tempo de Sessão (Base Limpa) ---")
    print(calcular_estatisticas_tempo(df_limpo))
    
    print("\n--- Q3: Conversão (%) ---")
    print(calcular_taxa_conversao(df_bruto))
    
    print("\n--- Q4: Teste Estatístico (Conversão) ---")
    res_estat = testar_significancia_conversao(df_bruto)
    print("Hipótese Nula (H0): As taxas de conversão são iguais entre os grupos.")
    print(f"Estatística de Teste (Chi2): {res_estat['Estatistica_Chi2']}")
    print(f"P-valor: {res_estat['P_valor']:.4e}")
    print(f"Conclusão: {res_estat['Mensagem']}")
    
    print("\n--- Analise por Dispositivo ---")
    print(analisar_por_dispositivo(df_bruto))
    
    print("\n--- Analise por País ---")
    print(analisar_por_pais(df_bruto))
    
    print("\n--- Auditoria de Qualidade dos Dados (Casos Zero/Nulos) ---")
    print(analisar_casos_vazios_zeros(df_bruto))
    
    # 3. Impressão de Texto (Insights e Recomendações)
    imprimir_insights_e_recomendacoes()
    
    # 4. Geração de Gráficos
    plotar_distribuicao_sessao(df_limpo)
    plotar_tendencias_temporais(df_bruto)

if __name__ == "__main__":
    main()