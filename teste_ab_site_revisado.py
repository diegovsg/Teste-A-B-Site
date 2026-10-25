"""
Teste A/B do menu lateral: Nível 0 (acesso direto) x Nível 1 (sob aba geral).

  1. Remove duplicatas exatas (todas eram conversões) antes de qualquer cálculo.
  2. Audita a base: desbalanceamento 64/36 (SRM), user_id instável, balanceamento por hora.
  3. Q1 responde por acessos à página (converted == 1) e mostra o tráfego separado.
  4. Q2 mostra duas visões: tempo de quem chegou à página e tempo por acesso (zeros incluídos).
     Zeros (não chegou) e nulos (convertido sem duração) são tratados separadamente.
  5. Q3/Q4 trazem IC 95%, diferença em p.p., odds ratio e testes de robustez.
  6. Dispositivo, país e hora têm testes de significância (evita ler ruído como padrão).
  7. Textos de insights e recomendações são calculados a partir dos dados (sem números digitados).
  8. Exporta a base tratada para uso em ferramentas de BI (Power BI/Looker).

Dependências: pandas, numpy, matplotlib, seaborn, scipy, statsmodels
    pip install pandas numpy matplotlib seaborn scipy statsmodels openpyxl
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

CAMINHO_ARQUIVO = 'ab_test_data 1.xlsx'
GRUPO_NIVEL_0 = 'Nível 0'       # acesso direto à página (controle)
GRUPO_NIVEL_1 = 'Nível 1'       # acesso sob a aba geral (nova escolha)
NIVEL_SIGNIFICANCIA = 0.05      # p-valores abaixo disso são considerados significativos


# ----------------------------------------------------------------------------
# Utilitários de formatação (padrão brasileiro: vírgula decimal)
# ----------------------------------------------------------------------------
def formatar_numero(valor: float, casas_decimais: int = 1) -> str:
    return f'{valor:.{casas_decimais}f}'.replace('.', ',')


def formatar_p_valor(p_valor: float) -> str:
    return '< 0,001' if p_valor < 0.001 else formatar_numero(p_valor, 3)


def formatar_dias(quantidade_dias: int) -> str:
    return f"{quantidade_dias} dia{'s' if quantidade_dias != 1 else ''}"


def formatar_colunas_p_valor(tabela: pd.DataFrame, colunas_p_valor: list) -> pd.DataFrame:
    """Troca p-valores minúsculos (que o round() mostraria como 0.0) por '< 0,001'."""
    tabela_formatada = tabela.copy()
    for coluna in colunas_p_valor:
        tabela_formatada[coluna] = tabela_formatada[coluna].map(formatar_p_valor)
    return tabela_formatada


def imprimir_p_valores(resultados_testes: dict):
    for nome_teste, valor in resultados_testes.items():
        if nome_teste.startswith('p_'):
            print(f"  {nome_teste}: p {formatar_p_valor(valor)}")


# ----------------------------------------------------------------------------
# Carga, Preparação e Exportação
# ----------------------------------------------------------------------------
def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega a base e cria a coluna 'hour' a partir do timestamp."""
    base_bruta = pd.read_excel(caminho_arquivo)
    base_bruta['timestamp'] = pd.to_datetime(base_bruta['timestamp'])
    base_bruta['hour'] = base_bruta['timestamp'].dt.hour
    return base_bruta


def remover_duplicatas(base_bruta: pd.DataFrame):
    """
    Remove linhas idênticas em TODAS as colunas (mesmo id, minuto, grupo, dispositivo, país
    e duração até a 6ª casa decimal). Obs: Não pode ser a mesma pessoa voltando: é o mesmo
    registro gravado duas vezes. Retorna a base limpa e um resumo do que foi removido.
    """
    linha_duplicada = base_bruta.duplicated()
    resumo_duplicatas = {
        'removidas': int(linha_duplicada.sum()),
        'pct_removidas': linha_duplicada.mean() * 100,
        'pct_conversoes_entre_removidas': (base_bruta.loc[linha_duplicada, 'converted'].mean() * 100
                                           if linha_duplicada.any() else np.nan),
        'por_grupo': base_bruta.loc[linha_duplicada, 'group'].value_counts().to_dict(),
    }
    base_sem_duplicatas = base_bruta.loc[~linha_duplicada].copy() #Procura as linhas que não são duplicadas e cria uma cópia do DataFrame sem as duplicatas.
    return base_sem_duplicatas, resumo_duplicatas


def auditar_base(base_sem_duplicatas: pd.DataFrame) -> dict:
    """Checagens de qualidade da base e do desenho do teste (usar na base sem duplicatas)."""
    total_linhas = len(base_sem_duplicatas)
    total_nivel_0 = int((base_sem_duplicatas['group'] == GRUPO_NIVEL_0).sum())

    linhas_por_user_id = base_sem_duplicatas.groupby('user_id').size()
    ids_repetidos = linhas_por_user_id[linhas_por_user_id > 1].index
    base_ids_repetidos = base_sem_duplicatas[base_sem_duplicatas['user_id'].isin(ids_repetidos)]
    variacao_por_user_id = base_ids_repetidos.groupby('user_id').agg(
        paises_distintos=('country', 'nunique'), dispositivos_distintos=('device', 'nunique'))

    sem_duracao = base_sem_duplicatas['session_duration'].isnull()
    duracao_zero = base_sem_duplicatas['session_duration'] == 0
    converteu = base_sem_duplicatas['converted'] == 1

    return {
        'linhas': total_linhas,
        'ids_unicos': base_sem_duplicatas['user_id'].nunique(),
        'ids_repetidos': len(ids_repetidos),
        'pct_repetidos_pais_diferente': (variacao_por_user_id['paises_distintos'] > 1).mean() * 100,
        'pct_repetidos_dispositivo_diferente': (variacao_por_user_id['dispositivos_distintos'] > 1).mean() * 100,
        'ids_nos_dois_grupos': int((base_sem_duplicatas.groupby('user_id')['group'].nunique() > 1).sum()),
        'periodo': (base_sem_duplicatas['timestamp'].min(), base_sem_duplicatas['timestamp'].max()),
        'dias': base_sem_duplicatas['timestamp'].dt.date.nunique(),
        'nulos_duracao_sessao': int(sem_duracao.sum()),
        'nulos_entre_convertidos': int((sem_duracao & converteu).sum()),
        'zeros_entre_convertidos': int((duracao_zero & converteu).sum()),
        'zeros_entre_nao_convertidos': int((duracao_zero & ~converteu).sum()),
        'pct_trafego_nivel_0': total_nivel_0 / total_linhas * 100,
        # SRM - Sample Ratio Mismatch: se o desenho era 50/50, a proporção observada indica falha de alocação
        'p_valor_divisao_50_50': stats.binomtest(total_nivel_0, total_linhas, 0.5).pvalue,
        # Balanceamento: grupos comparáveis em dispositivo, país e hora?
        'p_valor_balanceamento': {
            coluna: chi2_contingency(pd.crosstab(base_sem_duplicatas[coluna], base_sem_duplicatas['group']))[1]
            for coluna in ['device', 'country', 'hour']},
    }


def exportar_base_limpa(base_sem_duplicatas: pd.DataFrame, nome_arquivo_saida: str = 'ab_test_data_sem_duplicatas.xlsx'):
    """
    Exporta a base tratada para Excel (uso no Power BI): sem duplicatas exatas.
    Mantém os convertidos sem duração (nulos);
    """
    base_para_exportacao = base_sem_duplicatas.copy()
    base_para_exportacao.to_excel(nome_arquivo_saida, index=False, sheet_name='ab_data')
    print(f"\n[!] Base exportada com sucesso para '{nome_arquivo_saida}' (Tamanho: {len(base_para_exportacao)} linhas).")


# ----------------------------------------------------------------------------
# Q1: qual configuração levou mais usuários a acessar a página?
# ----------------------------------------------------------------------------
def calcular_acessos(base_sem_duplicatas: pd.DataFrame) -> pd.DataFrame:
    """
    Q1: 'acessar a página' = converted == 1 (conversão para a página de interesse).
    O tráfego total mede apenas a alocação do teste, não o efeito do menu.
    """
    tabela_acessos = base_sem_duplicatas.groupby('group').agg(
        Trafego_Total=('user_id', 'size'), Acessos_Pagina=('converted', 'sum'))
    tabela_acessos['Pct_do_Trafego'] = (
        tabela_acessos['Trafego_Total'] / tabela_acessos['Trafego_Total'].sum() * 100).round(1)
    return tabela_acessos


# ----------------------------------------------------------------------------
# Q2: tempo médio de sessão
# ----------------------------------------------------------------------------
def calcular_estatisticas_tempo(base_sem_duplicatas: pd.DataFrame) -> pd.DataFrame:
    """
    Q2: duas visões do tempo na página de interesse.
      - Quem chegou: só session_duration > 0. Zero = não chegou à página.
      - Por acesso: zeros incluídos. Sem seleção, mistura 'chegar' e 'ficar'.
    Nulos (convertidos sem duração medida) são dado faltante e saem dos dois cálculos.
    """
    base_com_duracao = base_sem_duplicatas.dropna(subset=['session_duration'])
    base_quem_chegou = base_com_duracao[base_com_duracao['session_duration'] > 0]

    duracao_quem_chegou = base_quem_chegou.groupby('group')['session_duration']
    tabela_tempo = pd.DataFrame({
        'N_Acessos': base_com_duracao.groupby('group').size(),
        'N_Chegaram (session_duration > 0)': base_quem_chegou.groupby('group').size(),
        'Media_Quem_Chegou': duracao_quem_chegou.mean(),
        'Mediana_Quem_Chegou': duracao_quem_chegou.median(),
        'Desvio_Quem_Chegou': duracao_quem_chegou.std(),
        'Media_Por_Acesso': base_com_duracao.groupby('group')['session_duration'].mean(),
    })
    tabela_tempo.insert(2, 'Pct_Chegaram', tabela_tempo['N_Chegaram (session_duration > 0)'] / tabela_tempo['N_Acessos'] * 100)
    return tabela_tempo.round(2)


def calcular_ic_diferenca_medias_welch(amostra_nivel_0: pd.Series, amostra_nivel_1: pd.Series,
                                       nivel_confianca: float = 0.95):
    """IC da diferença de médias (Welch). Independe da versão do scipy."""
    variancia_media_nivel_0 = amostra_nivel_0.var(ddof=1) / len(amostra_nivel_0)
    variancia_media_nivel_1 = amostra_nivel_1.var(ddof=1) / len(amostra_nivel_1)
    variancia_total = variancia_media_nivel_0 + variancia_media_nivel_1
    graus_liberdade = variancia_total ** 2 / (
        variancia_media_nivel_0 ** 2 / (len(amostra_nivel_0) - 1)
        + variancia_media_nivel_1 ** 2 / (len(amostra_nivel_1) - 1))
    margem_erro = stats.t.ppf(1 - (1 - nivel_confianca) / 2, graus_liberdade) * np.sqrt(variancia_total)
    diferenca_medias = amostra_nivel_0.mean() - amostra_nivel_1.mean()
    return diferenca_medias - margem_erro, diferenca_medias + margem_erro


def testar_tempo_sessao(base_sem_duplicatas: pd.DataFrame) -> pd.DataFrame:
    """
    Teste da diferença de tempo entre grupos (Nível 0 - Nível 1).
    Welch (não assume variâncias iguais) + Mann-Whitney (robusto à assimetria; a duração
    é assimétrica e tem mais de um 'bloco' de valores, então a média sozinha engana).
    """
    base_com_duracao = base_sem_duplicatas.dropna(subset=['session_duration'])
    bases_por_visao = {
        'Quem chegou à página': base_com_duracao[base_com_duracao['session_duration'] > 0],
        'Por acesso (zeros incluídos)': base_com_duracao,
    }
    resultados_por_visao = {}
    for nome_visao, base_visao in bases_por_visao.items():
        duracao_nivel_0 = base_visao.loc[base_visao['group'] == GRUPO_NIVEL_0, 'session_duration']
        duracao_nivel_1 = base_visao.loc[base_visao['group'] == GRUPO_NIVEL_1, 'session_duration']
        ic_diferenca = calcular_ic_diferenca_medias_welch(duracao_nivel_0, duracao_nivel_1)
        resultados_por_visao[nome_visao] = {
            'Dif_Media_s': duracao_nivel_0.mean() - duracao_nivel_1.mean(),
            'IC95_inf': ic_diferenca[0],
            'IC95_sup': ic_diferenca[1],
            'p_Welch': stats.ttest_ind(duracao_nivel_0, duracao_nivel_1, equal_var=False).pvalue,
            'p_MannWhitney': stats.mannwhitneyu(duracao_nivel_0, duracao_nivel_1).pvalue,
        }
    return pd.DataFrame(resultados_por_visao).T


# ----------------------------------------------------------------------------
# Q3: taxa de conversão
# ----------------------------------------------------------------------------
def calcular_taxa_conversao(base_sem_duplicatas: pd.DataFrame) -> pd.DataFrame:
    """Q3: taxa de conversão (%) por grupo, com IC 95% de Wilson (bom para proporções)."""
    tabela_conversao = base_sem_duplicatas.groupby('group')['converted'].agg(N='size', Conversoes='sum')
    tabela_conversao['Taxa_Pct'] = tabela_conversao['Conversoes'] / tabela_conversao['N'] * 100
    ics_wilson = [proportion_confint(conversoes, total_acessos, method='wilson')
                  for conversoes, total_acessos in zip(tabela_conversao['Conversoes'], tabela_conversao['N'])]
    tabela_conversao['IC95_inf'] = [intervalo[0] * 100 for intervalo in ics_wilson]
    tabela_conversao['IC95_sup'] = [intervalo[1] * 100 for intervalo in ics_wilson]
    return tabela_conversao.round(2)


# ----------------------------------------------------------------------------
# Q4: significância estatística
# ----------------------------------------------------------------------------
def testar_significancia_conversao(base: pd.DataFrame) -> dict:
    """
    Q4: qui-quadrado de independência (grupo x converteu) + tamanho do efeito.
    Em tabela 2x2 o scipy aplica a correção de Yates por padrão.
    O p-valor diz se a diferença existe; a diferença em p.p. e o IC dizem o quanto importa.
    """
    tabela_contingencia = pd.crosstab(base['group'], base['converted'])
    qui_quadrado, p_valor, _, _ = chi2_contingency(tabela_contingencia)

    conversao_nivel_0 = base.loc[base['group'] == GRUPO_NIVEL_0, 'converted']
    conversao_nivel_1 = base.loc[base['group'] == GRUPO_NIVEL_1, 'converted']
    taxa_nivel_0, taxa_nivel_1 = conversao_nivel_0.mean(), conversao_nivel_1.mean()

    diferenca_taxas = taxa_nivel_0 - taxa_nivel_1
    erro_padrao_diferenca = np.sqrt(taxa_nivel_0 * (1 - taxa_nivel_0) / len(conversao_nivel_0)
                                    + taxa_nivel_1 * (1 - taxa_nivel_1) / len(conversao_nivel_1))
    return {
        'taxa_nivel_0': taxa_nivel_0 * 100,
        'taxa_nivel_1': taxa_nivel_1 * 100,
        'diferenca_pp': diferenca_taxas * 100,
        'ic95_inferior_pp': (diferenca_taxas - 1.96 * erro_padrao_diferenca) * 100,
        'ic95_superior_pp': (diferenca_taxas + 1.96 * erro_padrao_diferenca) * 100,
        'lift_relativo_pct': (taxa_nivel_0 / taxa_nivel_1 - 1) * 100,
        'odds_ratio': (taxa_nivel_0 / (1 - taxa_nivel_0)) / (taxa_nivel_1 / (1 - taxa_nivel_1)),
        'qui_quadrado': qui_quadrado,
        'p_valor': p_valor,
        'significativo': p_valor < NIVEL_SIGNIFICANCIA,
    }


def testar_robustez(base_bruta: pd.DataFrame, base_sem_duplicatas: pd.DataFrame):
    """
    A conclusão depende de duplicatas ou do user_id? Refaz o teste em quatro cenários
    e estima o odds ratio ajustado por hora, dispositivo e país (regressão logística).
    """
    base_primeira_visita = base_sem_duplicatas.sort_values('timestamp').drop_duplicates('user_id')
    grupos_por_user_id = base_sem_duplicatas.groupby('user_id')['group'].transform('nunique')
    base_ids_em_um_grupo = base_sem_duplicatas[grupos_por_user_id == 1]

    bases_por_cenario = {
        'Base bruta (com duplicatas)': base_bruta,
        'Base sem duplicatas (principal)': base_sem_duplicatas,
        '1ª visita de cada user_id': base_primeira_visita,
        'Só user_id vistos em um grupo': base_ids_em_um_grupo,
    }
    linhas_tabela = []
    for nome_cenario, base_cenario in bases_por_cenario.items():
        resultado_cenario = testar_significancia_conversao(base_cenario)
        linhas_tabela.append({
            'Cenario': nome_cenario,
            'N_N0': (base_cenario['group'] == GRUPO_NIVEL_0).sum(),
            'N_N1': (base_cenario['group'] == GRUPO_NIVEL_1).sum(),
            'Conv_N0_Pct': resultado_cenario['taxa_nivel_0'],
            'Conv_N1_Pct': resultado_cenario['taxa_nivel_1'],
            'Dif_pp': resultado_cenario['diferenca_pp'],
            'P_valor': resultado_cenario['p_valor'],
        })
    tabela_robustez = pd.DataFrame(linhas_tabela).set_index('Cenario').round(2)

    base_com_indicador_nivel_1 = base_sem_duplicatas.assign(
        nivel_1=(base_sem_duplicatas['group'] == GRUPO_NIVEL_1).astype(int))

    def estimar_odds_ratio_nivel_0(formula_modelo: str) -> dict:
        modelo_logistico = smf.logit(formula_modelo, base_com_indicador_nivel_1).fit(disp=0)
        ic_coeficiente = modelo_logistico.conf_int().loc['nivel_1']
        # O coeficiente compara Nível 1 com Nível 0. Invertemos o sinal para expressar Nível 0 vs Nível 1.
        return {
            'odds_ratio': np.exp(-modelo_logistico.params['nivel_1']),
            'ic95_inferior': np.exp(-ic_coeficiente[1]),
            'ic95_superior': np.exp(-ic_coeficiente[0]),
        }

    odds_ratios = {
        'bruto': estimar_odds_ratio_nivel_0('converted ~ nivel_1'),
        'ajustado': estimar_odds_ratio_nivel_0('converted ~ nivel_1 + C(device) + C(country) + C(hour)'),
    }
    return tabela_robustez, odds_ratios


# ----------------------------------------------------------------------------
# Q5/Q7: dispositivo e país (com testes, para não ler ruído como padrão)
# ----------------------------------------------------------------------------
def analisar_por_segmento(base_sem_duplicatas: pd.DataFrame, coluna_segmento: str):
    """
    Conversão por segmento x grupo, com:
      - Fisher por segmento (efeito do menu dentro do segmento);
      - qui-quadrado entre segmentos DENTRO de cada grupo (segmentos diferem entre si?);
      - teste de interação grupo x segmento (o efeito do menu muda conforme o segmento?).
    """
    linhas_tabela = []
    for nome_segmento, base_segmento in base_sem_duplicatas.groupby(coluna_segmento):
        conversao_nivel_0 = base_segmento.loc[base_segmento['group'] == GRUPO_NIVEL_0, 'converted']
        conversao_nivel_1 = base_segmento.loc[base_segmento['group'] == GRUPO_NIVEL_1, 'converted']
        _, p_valor_fisher = stats.fisher_exact([
            [conversao_nivel_0.sum(), len(conversao_nivel_0) - conversao_nivel_0.sum()],
            [conversao_nivel_1.sum(), len(conversao_nivel_1) - conversao_nivel_1.sum()]])
        linhas_tabela.append({
            coluna_segmento: nome_segmento,
            'Acessos_N0': len(conversao_nivel_0),
            'Conv_N0_Pct': conversao_nivel_0.mean() * 100,
            'Acessos_N1': len(conversao_nivel_1),
            'Conversoes_N1': int(conversao_nivel_1.sum()),
            'Conv_N1_Pct': conversao_nivel_1.mean() * 100,
            'Dif_pp': (conversao_nivel_0.mean() - conversao_nivel_1.mean()) * 100,
            'p_Fisher': p_valor_fisher,
        })
    tabela_segmentos = pd.DataFrame(linhas_tabela)

    resultados_testes = {}
    for grupo in (GRUPO_NIVEL_0, GRUPO_NIVEL_1):
        base_grupo = base_sem_duplicatas[base_sem_duplicatas['group'] == grupo]
        resultados_testes[f'p_entre_segmentos_{grupo}'] = chi2_contingency(
            pd.crosstab(base_grupo[coluna_segmento], base_grupo['converted']))[1]

    base_com_indicador_nivel_1 = base_sem_duplicatas.assign(
        nivel_1=(base_sem_duplicatas['group'] == GRUPO_NIVEL_1).astype(int))
    modelo_sem_interacao = smf.logit(
        f'converted ~ nivel_1 + C({coluna_segmento})', base_com_indicador_nivel_1).fit(disp=0)
    modelo_com_interacao = smf.logit(
        f'converted ~ nivel_1 * C({coluna_segmento})', base_com_indicador_nivel_1).fit(disp=0)
    estatistica_razao_verossimilhanca = 2 * (modelo_com_interacao.llf - modelo_sem_interacao.llf)
    graus_liberdade_interacao = modelo_com_interacao.df_model - modelo_sem_interacao.df_model
    resultados_testes['p_interacao'] = stats.chi2.sf(estatistica_razao_verossimilhanca, graus_liberdade_interacao)

    resultados_testes['qtd_segmentos_nivel_0_maior'] = int((tabela_segmentos['Dif_pp'] > 0).sum())
    resultados_testes['qtd_segmentos'] = len(tabela_segmentos)
    return tabela_segmentos, resultados_testes


def avaliar_segmento_critico(base_sem_duplicatas: pd.DataFrame, tabela_segmentos: pd.DataFrame,
                             coluna_segmento: str) -> dict:
    """
    Segmento com pior conversão no Nível 1 vs. os demais. Como ele foi escolhido DEPOIS de
    olhar os dados, o p-valor recebe correção de Bonferroni (multiplica pelo nº de segmentos).
    """
    segmento_critico = tabela_segmentos.loc[tabela_segmentos['Conv_N1_Pct'].idxmin(), coluna_segmento]
    base_nivel_1 = base_sem_duplicatas[base_sem_duplicatas['group'] == GRUPO_NIVEL_1]
    conversao_segmento_critico = base_nivel_1.loc[base_nivel_1[coluna_segmento] == segmento_critico, 'converted']
    conversao_demais_segmentos = base_nivel_1.loc[base_nivel_1[coluna_segmento] != segmento_critico, 'converted']

    _, p_valor_fisher = stats.fisher_exact([
        [conversao_segmento_critico.sum(), len(conversao_segmento_critico) - conversao_segmento_critico.sum()],
        [conversao_demais_segmentos.sum(), len(conversao_demais_segmentos) - conversao_demais_segmentos.sum()]])
    ic_wilson = proportion_confint(conversao_segmento_critico.sum(), len(conversao_segmento_critico),
                                   method='wilson')
    return {
        'segmento': segmento_critico,
        'conversoes': int(conversao_segmento_critico.sum()),
        'total_acessos': len(conversao_segmento_critico),
        'taxa_segmento': conversao_segmento_critico.mean() * 100,
        'taxa_demais_segmentos': conversao_demais_segmentos.mean() * 100,
        'ic95_inferior': ic_wilson[0] * 100,
        'ic95_superior': ic_wilson[1] * 100,
        'p_valor_bonferroni': min(1.0, p_valor_fisher * len(tabela_segmentos)),
    }


# ----------------------------------------------------------------------------
# Q6: tendência ao longo do dia
# ----------------------------------------------------------------------------
def analisar_tendencia_horaria(base_sem_duplicatas: pd.DataFrame):
    """
    Q6: acessos e conversão por hora. A base cobre um único dia, então o padrão pode ser
    específico dele. Testes: a conversão varia por hora? o volume por hora é uniforme?
    """
    base_horas_grupo = base_sem_duplicatas.groupby(['hour', 'group']).agg(
        Acessos=('user_id', 'size'), Conversoes=('converted', 'sum')).reset_index()
    base_horas_grupo['Taxa_Pct'] = base_horas_grupo['Conversoes'] / base_horas_grupo['Acessos'] * 100
    ic_wilson = proportion_confint(base_horas_grupo['Conversoes'], base_horas_grupo['Acessos'], method='wilson')
    base_horas_grupo['IC95_inf'] = ic_wilson[0] * 100
    base_horas_grupo['IC95_sup'] = ic_wilson[1] * 100

    resultados_testes = {}
    for grupo in (GRUPO_NIVEL_0, GRUPO_NIVEL_1):
        base_grupo = base_sem_duplicatas[base_sem_duplicatas['group'] == grupo]
        resultados_testes[f'p_conversao_varia_por_hora_{grupo}'] = chi2_contingency(
            pd.crosstab(base_grupo['hour'], base_grupo['converted']))[1]
        acessos_por_hora = base_grupo['hour'].value_counts().reindex(range(24), fill_value=0)
        resultados_testes[f'p_volume_uniforme_{grupo}'] = stats.chisquare(acessos_por_hora)[1]
    return base_horas_grupo, resultados_testes


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
PALETA_CORES_POR_GRUPO = dict(zip((GRUPO_NIVEL_0, GRUPO_NIVEL_1), sns.color_palette('Set2')[:2]))


def plotar_distribuicao_sessao(base_sem_duplicatas: pd.DataFrame):
    """Q2 (visual): distribuição do tempo na página, somente de quem chegou até ela."""
    base_quem_chegou = base_sem_duplicatas[base_sem_duplicatas['session_duration'] > 0]
    plt.figure(figsize=(10, 6))
    sns.histplot(data=base_quem_chegou, x='session_duration', hue='group', kde=True, bins=30,
                 stat='density', common_norm=False, palette=PALETA_CORES_POR_GRUPO, alpha=0.6)
    plt.gca().get_legend().set_title('Grupo')
    plt.title('Tempo na página (apenas quem chegou até ela): Nível 0 vs Nível 1')
    plt.xlabel('Duração da sessão (segundos)')
    plt.ylabel('Densidade (cada grupo soma 100%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plotar_tendencias_temporais(base_horas_grupo: pd.DataFrame):
    """Q6 (visual): acessos e conversão por hora. A faixa mostra o IC 95%: faixas largas = pouco dado."""
    figura, (eixo_acessos, eixo_conversao) = plt.subplots(2, 1, figsize=(12, 10))
    for grupo in (GRUPO_NIVEL_0, GRUPO_NIVEL_1):
        base_grupo = base_horas_grupo[base_horas_grupo['group'] == grupo]
        cor_grupo = PALETA_CORES_POR_GRUPO[grupo]
        eixo_acessos.plot(base_grupo['hour'], base_grupo['Acessos'], marker='o', label=grupo, color=cor_grupo)
        eixo_conversao.plot(base_grupo['hour'], base_grupo['Taxa_Pct'], marker='o', label=grupo, color=cor_grupo)
        eixo_conversao.fill_between(base_grupo['hour'], base_grupo['IC95_inf'], base_grupo['IC95_sup'],
                                    color=cor_grupo, alpha=0.15)
    eixo_acessos.set_title('Acessos por hora do dia (base sem duplicatas, 1 dia de dados)')
    eixo_acessos.set_ylabel('Quantidade de acessos')
    eixo_conversao.set_title('Taxa de conversão por hora (%) com IC 95%')
    eixo_conversao.set_ylabel('Taxa de conversão (%)')
    for eixo in (eixo_acessos, eixo_conversao):
        eixo.set_xlabel('Hora do dia (0-23)')
        eixo.set_xticks(range(24))
        eixo.grid(axis='y', linestyle='--', alpha=0.7)
        eixo.legend(title='Grupo')
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Q5 e Q7
# ----------------------------------------------------------------------------
def imprimir_insights_e_recomendacoes(resultados: dict):
    resultado_conversao = resultados['resultado_conversao']
    auditoria = resultados['auditoria']
    resumo_duplicatas = resultados['resumo_duplicatas']
    tabela_tempo = resultados['tabela_tempo']
    testes_tempo = resultados['testes_tempo']
    tabela_robustez = resultados['tabela_robustez']
    odds_ratios = resultados['odds_ratios']
    testes_dispositivo = resultados['testes_dispositivo']
    testes_pais = resultados['testes_pais']
    pais_critico = resultados['pais_critico']
    testes_hora = resultados['testes_hora']

    tempo_nivel_0 = tabela_tempo.loc[GRUPO_NIVEL_0]
    tempo_nivel_1 = tabela_tempo.loc[GRUPO_NIVEL_1]
    teste_tempo_por_acesso = testes_tempo.loc['Por acesso (zeros incluídos)']

    p_dispositivos_nivel_0 = testes_dispositivo[f'p_entre_segmentos_{GRUPO_NIVEL_0}']
    p_dispositivos_nivel_1 = testes_dispositivo[f'p_entre_segmentos_{GRUPO_NIVEL_1}']
    ha_diferenca_entre_dispositivos = (testes_dispositivo['p_interacao'] < NIVEL_SIGNIFICANCIA
                                       or min(p_dispositivos_nivel_0, p_dispositivos_nivel_1) < NIVEL_SIGNIFICANCIA)

    p_hora_nivel_0 = testes_hora[f'p_conversao_varia_por_hora_{GRUPO_NIVEL_0}']
    p_hora_nivel_1 = testes_hora[f'p_conversao_varia_por_hora_{GRUPO_NIVEL_1}']
    conversao_varia_por_hora = {
        GRUPO_NIVEL_0: 'varia' if p_hora_nivel_0 < NIVEL_SIGNIFICANCIA else 'sem evidência de variação',
        GRUPO_NIVEL_1: 'varia' if p_hora_nivel_1 < NIVEL_SIGNIFICANCIA else 'sem evidência de variação',
    }
    nivel_0_superior = resultado_conversao['significativo'] and resultado_conversao['diferenca_pp'] > 0

# ----------------------------------------------------------------------------
# Execução
# ----------------------------------------------------------------------------
def main():
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)

    # 1. Carga, limpeza e auditoria
    base_bruta = carregar_dados(CAMINHO_ARQUIVO)
    base_sem_duplicatas, resumo_duplicatas = remover_duplicatas(base_bruta)
    auditoria = auditar_base(base_sem_duplicatas)

    print('--- Auditoria da base ---')
    print(f"Linhas brutas: {len(base_bruta)} | duplicatas exatas removidas: {resumo_duplicatas['removidas']} "
          f"({formatar_numero(resumo_duplicatas['pct_conversoes_entre_removidas'], 0)}% eram conversões) "
          f"| linhas analisadas: {auditoria['linhas']}")
    print(f"Período: {auditoria['periodo'][0]} a {auditoria['periodo'][1]} ({formatar_dias(auditoria['dias'])})")
    print(f"Divisão do tráfego: {formatar_numero(auditoria['pct_trafego_nivel_0'])}% Nível 0 / "
          f"{formatar_numero(100 - auditoria['pct_trafego_nivel_0'])}% Nível 1 "
          f"(p {formatar_p_valor(auditoria['p_valor_divisao_50_50'])} contra 50/50)")
    print(f"user_id: {auditoria['ids_unicos']} únicos em {auditoria['linhas']} linhas; "
          f"{auditoria['ids_repetidos']} repetidos, dos quais "
          f"{formatar_numero(auditoria['pct_repetidos_pais_diferente'], 0)}% com país diferente e "
          f"{auditoria['ids_nos_dois_grupos']} nos dois grupos")
    print(f"Duração zero: {auditoria['zeros_entre_nao_convertidos']} entre não convertidos e "
          f"{auditoria['zeros_entre_convertidos']} entre convertidos | duração nula: "
          f"{auditoria['nulos_duracao_sessao']}, dos quais {auditoria['nulos_entre_convertidos']} entre convertidos")
    print('Balanceamento entre grupos (p-valor do qui-quadrado): '
          + ', '.join(f'{coluna} = {formatar_p_valor(p_valor)}'
                      for coluna, p_valor in auditoria['p_valor_balanceamento'].items()))

    # 2. Respostas
    print('\n--- Q1: Acessos à página de interesse ---')
    print(calcular_acessos(base_sem_duplicatas))

    print('\n--- Q2: Tempo de sessão (segundos) ---')
    tabela_tempo = calcular_estatisticas_tempo(base_sem_duplicatas)
    print(tabela_tempo.T)
    testes_tempo = testar_tempo_sessao(base_sem_duplicatas)
    print('\nDiferença Nível 0 - Nível 1 (segundos):')
    print(formatar_colunas_p_valor(testes_tempo.round(2), ['p_Welch', 'p_MannWhitney']))

    print('\n--- Q3: Conversão (%) ---')
    print(calcular_taxa_conversao(base_sem_duplicatas))

    print('\n--- Q4: Teste estatístico (conversão) ---')
    resultado_conversao = testar_significancia_conversao(base_sem_duplicatas)
    print('H0: as taxas de conversão são iguais entre os grupos.')
    print(f"Chi2 = {formatar_numero(resultado_conversao['qui_quadrado'], 2)} | "
          f"p = {resultado_conversao['p_valor']:.2e} | "
          f"{'diferença significativa (rejeita H0)' if resultado_conversao['significativo'] else 'sem evidência de diferença'}")
    print(f"Diferença = {formatar_numero(resultado_conversao['diferenca_pp'])} p.p. "
          f"(IC 95%: {formatar_numero(resultado_conversao['ic95_inferior_pp'])} a "
          f"{formatar_numero(resultado_conversao['ic95_superior_pp'])}) "
          f"| odds ratio = {formatar_numero(resultado_conversao['odds_ratio'], 2)}")
    print('\nRobustez:')
    tabela_robustez, odds_ratios = testar_robustez(base_bruta, base_sem_duplicatas)
    print(formatar_colunas_p_valor(tabela_robustez, ['P_valor']))
    print(f"Odds ratio Nível 0 vs Nível 1: bruto = {formatar_numero(odds_ratios['bruto']['odds_ratio'], 2)} "
          f"| ajustado por hora, dispositivo e país = {formatar_numero(odds_ratios['ajustado']['odds_ratio'], 2)} "
          f"(IC 95%: {formatar_numero(odds_ratios['ajustado']['ic95_inferior'], 2)} a "
          f"{formatar_numero(odds_ratios['ajustado']['ic95_superior'], 2)})")

    print('\n--- Análise por dispositivo ---')
    tabela_dispositivos, testes_dispositivo = analisar_por_segmento(base_sem_duplicatas, 'device')
    print(formatar_colunas_p_valor(tabela_dispositivos.round(2), ['p_Fisher']))
    imprimir_p_valores(testes_dispositivo)

    print('\n--- Análise por país ---')
    tabela_paises, testes_pais = analisar_por_segmento(base_sem_duplicatas, 'country')
    print(formatar_colunas_p_valor(tabela_paises.round(2), ['p_Fisher']))
    imprimir_p_valores(testes_pais)
    pais_critico = avaliar_segmento_critico(base_sem_duplicatas, tabela_paises, 'country')

    print('\n--- Q6: Tendência ao longo do dia ---')
    base_horas_grupo, testes_hora = analisar_tendencia_horaria(base_sem_duplicatas)
    print(base_horas_grupo.pivot(index='hour', columns='group', values=['Acessos', 'Taxa_Pct']).round(1))
    imprimir_p_valores(testes_hora)

    # 3. Insights e recomendações (calculados a partir dos resultados acima)
    imprimir_insights_e_recomendacoes({
        'resumo_duplicatas': resumo_duplicatas,
        'auditoria': auditoria,
        'resultado_conversao': resultado_conversao,
        'tabela_robustez': tabela_robustez,
        'odds_ratios': odds_ratios,
        'tabela_tempo': tabela_tempo,
        'testes_tempo': testes_tempo,
        'testes_dispositivo': testes_dispositivo,
        'testes_pais': testes_pais,
        'pais_critico': pais_critico,
        'testes_hora': testes_hora,
    })

    # 4. Gráficos
    plotar_distribuicao_sessao(base_sem_duplicatas)
    plotar_tendencias_temporais(base_horas_grupo)
    
    # 5. Exportação da base limpa para o Power BI
    exportar_base_limpa(base_sem_duplicatas)


if __name__ == '__main__':
    main()