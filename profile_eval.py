import cProfile
import pstats
import os
import time

from src.ga_engine import carregar_dataset, avaliar_entidade, GLOBAL_DATASET_PATH, gerar_entidade
import src.ga_engine as ga

def run_profile():
    print("Carregando dataset...")
    ga.GLOBAL_DATASET_PATH = ga.DATASET_PATH
    dataset = carregar_dataset(ga.DATASET_PATH, silencioso=False)
    ga.GLOBAL_DATASET = dataset
    
    entidade = gerar_entidade()
    print("Entidade a ser avaliada:", entidade)
    
    # Executar uma vez para aquecer caches se tiver
    print("Rodando a avaliação com cProfile...")
    args = (entidade, 0, False) # usar_wechat = False
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    resultado = avaliar_entidade(args)
    
    profiler.disable()
    
    with open("profile_results.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumtime')
        stats.print_stats(50)
        
    print("Profile salvo em profile_results.txt")

if __name__ == '__main__':
    run_profile()
