import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = 'qrcode_data.db'


def carregar_dados():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM qr_extractions ORDER BY timestamp DESC', conn)
    conn.close()

    # Descompactar JSON de filtros para colunas separadas
    if 'filtros' in df.columns:
        filtros_df = pd.json_normalize(df['filtros'].apply(json.loads))
        df = pd.concat([df.drop(columns=['filtros']), filtros_df], axis=1)

    return df


def mostrar_estatisticas(df):
    print('\n===== Estatísticas gerais =====')
    print(df[['score']].describe())
    total = len(df)
    qtd_det = (df['score'] > 0).sum()
    qtd_decod = df['decoded_text'].notna().sum()
    print(f'Total de linhas: {total}')
    print(f'Linhas com algum score > 0: {qtd_det}')
    print(f'Linhas com texto decodificado: {qtd_decod}')

    # Quanto cada filtro está presente por score
    print('\n===== Principais filtros + escalar score =====')
    for col in ['kernel_size', 'contrast', 'bright', 'sat', 'sharp', 'clahe', 'thresh_block', 'thresh_c', 'gamma', 'denoise', 'expo']:
        if col in df.columns:
            media = df.loc[df['score'] > 0, col].mean()
            geral = df[col].mean()
            print(f'{col:12}: media_score_pos={media:.2f} / media_geral={geral:.2f}')


def plotar_graficos(df):
    plt.figure(figsize=(10, 5))
    df['score'].hist(bins=20)
    plt.title('Distribuição do Score')
    plt.xlabel('score')
    plt.ylabel('quantidade')
    plt.tight_layout()
    plt.savefig('score_hist.png')
    print('Salvo histogram score em score_hist.png')

    plt.figure(figsize=(10, 6))
    var = df.groupby('kernel_size')['score'].mean().sort_index()
    var.plot(kind='bar')
    plt.title('Média de Score por kernel_size')
    plt.xlabel('kernel_size')
    plt.ylabel('score médio')
    plt.tight_layout()
    plt.savefig('kernel_score.png')
    print('Salvo gráfico kernel_size x score em kernel_score.png')

    # Decodificados por filtro
    if 'decoded_text' in df.columns:
        print('\n===== Top 10 textos decodificados =====')
        print(df[df['decoded_text'].notna()]['decoded_text'].value_counts().head(10))


def main():
    df = carregar_dados()
    if df.empty:
        print('Banco de dados vazio ou tabela inexistente.')
        return

    mostrar_estatisticas(df)
    print('\n===== Exibindo 20 registros mais recentes =====')
    print(df[['id', 'image_name', 'score', 'decoded_text', 'timestamp']].head(20).to_string(index=False))

    plotar_graficos(df)


if __name__ == '__main__':
    main()
