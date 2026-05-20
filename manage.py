import sys
import io
import argparse
import uvicorn

from src.database import (
    inicializar_db, 
    listar_todos, 
    buscar_por_imagem, 
    excluir_id, 
    compactar_db, 
    obter_estatisticas_gerais, 
    exportar_csv
)
from src.ga_engine import algoritmo_genetico
from src.ml_model import treinar_modelo

def cmd_run_web(args):
    """Inicia o servidor web FastAPI usando Uvicorn."""
    print("🚀 Iniciando Servidor Web FastAPI...")
    print("👉 Acesse a interface premium em: http://127.0.0.1:8000")
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=True)

def cmd_run_ga(args):
    """Inicia o ciclo evolutivo do Algoritmo Genético no terminal."""
    inicializar_db()
    use_nn = not args.no_nn
    
    print(f"🧬 Iniciando Algoritmo Genético no terminal...")
    print(f"🔹 População: {args.pop} | Gerações: {args.gen} | Usar IA: {use_nn}")
    
    def console_progress(g, total_g, melhor_score, media_score, melhor_cfg):
        print(f"   🧬 [Geração {g:02d}/{total_g}] 🏆 Melhor Score: {melhor_score:.2%} | 📈 Média: {media_score:.2%}")
        sys.stdout.flush()
    
    res = algoritmo_genetico(
        use_nn=use_nn,
        pop_size=args.pop,
        gen_count=args.gen,
        progress_callback=console_progress
    )
    
    if res:
        print("\n🏆 ALGORITMO CONCLUÍDO! CONFIGURAÇÃO VENCEDORA FINAL:")
        for gene, valor in res['vencedor'].items():
            print(f"   🔹 {gene}: {valor}")
        print(f"\n📊 Resultados: Melhor Score final: {res['melhor_score_final']:.2%}")
        print(f"📊 Tempo Total: {res['tempo_total']}s")

def cmd_ml_train(args):
    """Treina ou retreina o classificador MLP com os dados atuais."""
    print("🧠 Carregando base histórica e treinando MLPRegressor...")
    model = treinar_modelo()
    if model:
        print("✅ Treinamento concluído. O modelo em models/filtro_predictor.pkl foi atualizado.")
    else:
        print("❌ Falha ao treinar modelo.")

def cmd_db(args):
    """Gerencia operações do banco de dados SQLite."""
    inicializar_db()
    
    if args.action == "stats":
        stats = obter_estatisticas_gerais()
        print("\n===== Estatísticas Gerais do Banco =====")
        for k, v in stats.items():
            print(f"  🔹 {k.replace('_', ' ').capitalize()}: {v}")
            
    elif args.action == "listar":
        rows = listar_todos(args.limite)
        if not rows:
            print("Nenhum registro encontrado.")
            return
        print(f"\nExibindo os últimos {len(rows)} registros:")
        for r in rows:
            print(f"id={r[0]} | img={r[1]} | score={r[2]:.2f} | decoded={(r[3][:60] + '...') if r[3] and len(r[3]) > 60 else r[3]} | ts={r[4]}")
            
    elif args.action == "buscar":
        if not args.nome:
            print("❌ Especifique o nome parcial com --nome")
            return
        rows = buscar_por_imagem(args.nome)
        if not rows:
            print("Nenhum registro encontrado para essa imagem.")
            return
        for r in rows:
            print("---")
            print(f"id: {r[0]}")
            print(f"image_name: {r[1]}")
            print(f"score: {r[3]}")
            print(f"decoded_text: {r[4]}")
            print(f"bbox: {r[5]}")
            print(f"timestamp: {r[6]}")
            
    elif args.action == "excluir":
        if not args.id:
            print("❌ Especifique o ID com --id")
            return
        afetados = excluir_id(args.id)
        if afetados:
            print(f"✅ {afetados} linha(s) excluída(s) com sucesso.")
        else:
            print("❌ ID não encontrado.")
            
    elif args.action == "vacuum":
        print("⏳ Compactando banco de dados (VACUUM)...")
        compactar_db()
        print("✅ Banco compactado.")
        
    elif args.action == "export":
        arquivo = args.arquivo or "export_qr_extractions.csv"
        print(f"⏳ Exportando base de extrações para {arquivo}...")
        exportar_csv(arquivo)
        print("✅ Exportação concluída.")

def main():
    # Reconfigura o console para UTF-8 em ambientes Windows para suportar emojis sem erros de encoding
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="CLI Unificada: Otimização Inteligente de QR Codes (GA + IA)"
    )
    subparsers = parser.add_subparsers(title="Comandos disponíveis", dest="command", required=True)

    # 1. run-web
    subparsers.add_parser("run-web", help="Inicia o painel gráfico web FastAPI + CSS")

    # 2. run-ga
    parser_ga = subparsers.add_parser("run-ga", help="Inicia a otimização evolutiva no terminal")
    parser_ga.add_argument("--pop", type=int, default=10, help="Tamanho da população (default: 10)")
    parser_ga.add_argument("--gen", type=int, default=10, help="Quantidade de gerações (default: 10)")
    parser_ga.add_argument("--no-nn", action="store_true", help="Desativa o guimento preditivo por Rede Neural")

    # 3. ml-train
    subparsers.add_parser("ml-train", help="Retreina a Rede Neural MLPRegressor com os dados do DB")

    # 4. db
    parser_db = subparsers.add_parser("db", help="Gerencia e interage com o banco de dados SQLite")
    parser_db.add_argument("action", choices=["stats", "listar", "buscar", "excluir", "vacuum", "export"], 
                           help="Ação a ser executada no banco de dados")
    parser_db.add_argument("--limite", type=int, default=50, help="Limite de listagem (default: 50)")
    parser_db.add_argument("--nome", type=str, help="Nome parcial da imagem para busca")
    parser_db.add_argument("--id", type=int, help="ID do registro para exclusão")
    parser_db.add_argument("--arquivo", type=str, help="Caminho do CSV de destino para exportação")

    args = parser.parse_args()

    # Mapeamento de funções CLI
    comandos = {
        "run-web": cmd_run_web,
        "run-ga": cmd_run_ga,
        "ml-train": cmd_ml_train,
        "db": cmd_db
    }

    comandos[args.command](args)

if __name__ == "__main__":
    main()
