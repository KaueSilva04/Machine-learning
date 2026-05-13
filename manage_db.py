import sqlite3
import json
import os

DB_PATH = 'qrcode_data.db'


def conectar():
    return sqlite3.connect(DB_PATH)


def listar_todos(limite=50):
    conn = conectar()
    c = conn.cursor()
    c.execute('SELECT id, image_name, score, decoded_text, timestamp FROM qr_extractions ORDER BY id DESC LIMIT ?', (limite,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        print('Nenhum registro encontrado.')
        return

    print(f'Exibindo últimos {len(rows)} registros:')
    for r in rows:
        print(f'id={r[0]} | img={r[1]} | score={r[2]:.2f} | decoded={(r[3][:80] + "...") if r[3] and len(r[3]) > 80 else r[3]} | ts={r[4]}')


def buscar_por_imagem(nome):
    conn = conectar()
    c = conn.cursor()
    c.execute('SELECT * FROM qr_extractions WHERE image_name LIKE ? ORDER BY id DESC', (f'%{nome}%',))
    rows = c.fetchall()
    conn.close()

    if not rows:
        print('Nenhum registro encontrado para essa imagem.')
        return

    for r in rows:
        print('---')
        print(f'id: {r[0]}')
        print(f'image_name: {r[1]}')
        print(f'score: {r[3]}')
        print(f'decoded_text: {r[4]}')
        print(f'bbox: {r[5]}')
        print(f'timestamp: {r[6]}')


def excluir_id(uid):
    conn = conectar()
    c = conn.cursor()
    c.execute('DELETE FROM qr_extractions WHERE id = ?', (uid,))
    conn.commit()
    afetados = c.rowcount
    conn.close()

    if afetados:
        print(f'{afetados} linha(s) excluída(s)')
    else:
        print('Nenhuma linha excluída (id não encontrado).')


def exportar_csv(arquivo='export_qr_extractions.csv'):
    import csv
    conn = conectar()
    c = conn.cursor()
    c.execute('SELECT * FROM qr_extractions ORDER BY id')
    rows = c.fetchall()
    colunas = [d[0] for d in c.description]

    with open(arquivo, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(colunas)
        writer.writerows(rows)

    conn.close()
    print(f'Exportado para {arquivo}')


def listar_caracteristicas(limite=50):
    conn = conectar()
    c = conn.cursor()
    c.execute('SELECT image_name, brightness, contrast, saturation, laplacian_variance, edge_density, qr_raw_detected, width, height, timestamp FROM image_features ORDER BY id DESC LIMIT ?', (limite,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        print('Nenhum registro de características encontrado.')
        return

    print(f'Exibindo últimos {len(rows)} registros de características:')
    for r in rows:
        print(f'img={r[0]} | br={r[1]:.2f} | cn={r[2]:.2f} | sat={r[3]:.2f} | lap={r[4]:.2f} | edge={r[5]:.4f} | qr={r[6]} | {r[7]}x{r[8]} | ts={r[9]}')


def buscar_caracteristicas_por_imagem(nome):
    conn = conectar()
    c = conn.cursor()
    c.execute('SELECT * FROM image_features WHERE image_name LIKE ? ORDER BY id DESC', (f'%{nome}%',))
    rows = c.fetchall()
    conn.close()

    if not rows:
        print('Nenhum registro de características encontrado para essa imagem.')
        return

    for r in rows:
        print('---')
        print(f'id: {r[0]}')
        print(f'image_name: {r[1]}')
        print(f'brightness: {r[2]:.2f}')
        print(f'contrast: {r[3]:.2f}')
        print(f'saturation: {r[4]:.2f}')
        print(f'laplacian_variance: {r[5]:.2f}')
        print(f'edge_density: {r[6]:.4f}')
        print(f'qr_raw_detected: {r[7]}')
        print(f'qr_raw_text: {r[8]}')
        print(f'width: {r[9]} height: {r[10]}')
        print(f'timestamp: {r[11]}')


def compactar_db():
    conn = conectar()
    conn.execute('VACUUM')
    conn.close()
    print('Banco compactado.')


def menu():
    while True:
        print('\n=== Gerenciador de qrcode_data.db ===')
        print('1. Listar registros recentes')
        print('2. Buscar registros por nome de imagem')
        print('3. Excluir registro por id')
        print('4. Exportar tabela para CSV')
        print('5. Listar características de imagem')
        print('6. Buscar características por imagem')
        print('7. Compactar banco (VACUUM)')
        print('8. Sair')

        opcao = input('Escolha opção: ').strip()
        if opcao == '1':
            limite = input('Limite de registros (default 50): ').strip()
            limite = int(limite) if limite.isdigit() else 50
            listar_todos(limite)
        elif opcao == '2':
            nome = input('Nome parcial da imagem: ').strip()
            buscar_por_imagem(nome)
        elif opcao == '3':
            uid = input('ID para excluir: ').strip()
            if uid.isdigit():
                excluir_id(int(uid))
            else:
                print('ID inválido.')
        elif opcao == '4':
            arquivo = input('Nome do arquivo CSV (default export_qr_extractions.csv): ').strip()
            if not arquivo:
                arquivo = 'export_qr_extractions.csv'
            exportar_csv(arquivo)
        elif opcao == '5':
            limite = input('Limite de registros de características (default 50): ').strip()
            limite = int(limite) if limite.isdigit() else 50
            listar_caracteristicas(limite)
        elif opcao == '6':
            nome = input('Nome parcial da imagem para procurar nas características: ').strip()
            buscar_caracteristicas_por_imagem(nome)
        elif opcao == '7':
            compactar_db()
        elif opcao == '8':
            print('Saindo.')
            break
        else:
            print('Opção inválida.')


if __name__ == '__main__':
    if not os.path.exists(DB_PATH):
        print('Banco não encontrado:', DB_PATH)
    else:
        menu()
