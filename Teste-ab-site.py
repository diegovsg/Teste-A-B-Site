import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega a base de dados em excel para um DataFrame do Pandas."""
    return pd.read_excel(caminho_arquivo)

def limpar_dados_sessao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove registros onde o tempo de sessão é nulo ou igual a zero, 
    pois representam rejeições imediatas ou erros de carregamento 
    que distorceriam o tempo de permanência real.
    """
    df_limpo = df.dropna(subset=['session_duration']).copy()
    return df_limpo[df_limpo['session_duration'] > 0]

def calcular_acessos(df: pd.DataFrame) -> pd.Series:
    """Questão 1: Retorna a quantidade total de acessos por grupo."""
    return df['group'].value_counts()

def calcular_estatisticas_tempo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Questão 2: Calcula a Média, Mediana e Desvio Padrão 
    do tempo de sessão (em segundos) por grupo.
    """
    return df.groupby('group')['session_duration'].agg(
        Media='mean', 
        Mediana='median', 
        Desvio_Padrao='std'
    ).round(2)

def calcular_taxa_conversao(df: pd.DataFrame) -> pd.Series:
    """Questão 3: Calcula a taxa de conversão (%) por grupo."""
    return (df.groupby('group')['converted'].mean() * 100).round(2)

def plotar_distribuicao_sessao(df: pd.DataFrame):
    """Gera um histograma com linha de densidade para avaliar a distribuição do tempo."""
    plt.figure(figsize=(10, 6))
    
    # O kde=True adiciona a linha de tendência (curva de densidade)
    sns.histplot(
        data=df, 
        x='session_duration', 
        hue='group', 
        kde=True, 
        bins=30, 
        palette='Set2', 
        alpha=0.6
    )
    
    plt.title('Distribuição do Tempo de Sessão por Grupo (Nível 0 vs Nível 1)')
    plt.xlabel('Duração da Sessão (segundos)')
    plt.ylabel('Quantidade de Acessos')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUÇÃO DO PIPELINE DE DADOS
# ==========================================
if __name__ == "__main__":
    # 1. Carregamento e Limpeza
    df_bruto = carregar_dados('ab_test_data 1.xlsx')
    df_limpo = limpar_dados_sessao(df_bruto)
    
    # 2. Resolução das Questões
    acessos_totais = calcular_acessos(df_bruto)
    estatisticas_tempo = calcular_estatisticas_tempo(df_limpo)
    taxa_conversao = calcular_taxa_conversao(df_bruto)
    
    # 3. Exibição dos Resultados
    print("--- Q1: Acessos Totais por Configuração ---")
    print(acessos_totais)
    
    print("\n--- Q2: Estatísticas de Tempo de Sessão (Base Limpa) ---")
    print(estatisticas_tempo)
    
    print("\n--- Q3: Taxa de Conversão (%) ---")
    print(taxa_conversao)
    
    # 4. Geração do Gráfico para análise da Questão 2
    plotar_distribuicao_sessao(df_limpo)