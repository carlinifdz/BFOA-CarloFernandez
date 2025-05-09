from copy import copy
from multiprocessing import Manager, Pool
import time
from bacteria import bacteria
import numpy
import copy
import random
import csv
import os

from fastaReader import fastaReader

if __name__ == "__main__":
    numeroDeBacterias = 6
    numRandomBacteria = 1
    iteraciones = 5
    tumbo = 300
    nado = 3
    secuencias = list()
    secuencias = fastaReader().seqs
    names = fastaReader().names

    for i in range(len(secuencias)):
        secuencias[i] = list(secuencias[i])
    globalNFE = 0
    dAttr= 0.1
    wAttr= 0.002
    hRep=0.1
    wRep= 0.001

    manager = Manager()
    numSec = len(secuencias)
    print("numSec: ", numSec)

    all_results = []

    num_ejecuciones = 10
    for ejecucion in range(1, num_ejecuciones + 1):
        print(f"\n--- Ejecución: {ejecucion} ---")

        poblacion = manager.list(range(numeroDeBacterias))
        names = manager.list(names)
        NFE = manager.list(range(numeroDeBacterias))

        veryBestBacteria = None
        veryBestFitness = -float('inf')
        veryBestBlosum = None
        veryBestInteraction = None

        def poblacionInicial():
            for i in range(numeroDeBacterias):
                bacterium = []
                for j in range(numSec):
                    bacterium.append(secuencias[j])
                poblacion[i] = list(bacterium)

        def printPoblacion():
            for i in range(numeroDeBacterias):
                print(poblacion[i])

        operadorBacterial = bacteria(numeroDeBacterias)

        start_time = time.time()

        print("poblacion inicial ...")
        poblacionInicial()

        for it in range(iteraciones):
            print(f"\n--- Iteración: {it + 1} ---")
            print("Tumbo ...")
            operadorBacterial.tumbo(numSec, poblacion, tumbo)
            print("Cuadrando ...")
            operadorBacterial.cuadra(numSec, poblacion)
            print("Creando granLista de Pares...")
            operadorBacterial.creaGranListaPares(poblacion)
            print("Evaluando Blosum Parallel...")
            operadorBacterial.evaluaBlosum()
            print("Creando Tablas Atract/Repel Parallel...")
            operadorBacterial.creaTablasAtractRepel(poblacion, dAttr, wAttr,hRep, wRep)
            operadorBacterial.creaTablaInteraction()
            print("Creando tabla Fitness...")
            operadorBacterial.creaTablaFitness()
            print("Tabla Fitness creada.")
            globalNFE += operadorBacterial.getNFE()
            bestIdx, bestFitness, bestBlosum, bestInteraccion = operadorBacterial.obtieneBest(globalNFE)

            if bestFitness > veryBestFitness:
                veryBestFitness = bestFitness
                veryBestBacteria = copy.deepcopy(poblacion[bestIdx])
                veryBestBlosum = bestBlosum
                veryBestInteraction = bestInteraccion
                print(f"Nuevo mejor global encontrado. Fitness: {veryBestFitness}, Blosum: {veryBestBlosum}, Interaction: {veryBestInteraction}")
                veryBestBacteria = operadorBacterial.refineMejorBacteria(veryBestBacteria, numSec)
            worstIdx = 0
            for i in range(len(operadorBacterial.tablaFitness)):
                if operadorBacterial.tablaFitness[i] < operadorBacterial.tablaFitness[worstIdx]:
                    worstIdx = i

            if veryBestBacteria is not None:
                num_mutations = 5
                mutated_best = operadorBacterial.mutateBest(veryBestBacteria, num_mutations)
                poblacion[worstIdx] = mutated_best
                print(f"Bacteria {worstIdx} reemplazada con mutación del mejor global.")
            else:
                operadorBacterial.replaceWorst(poblacion, bestIdx)

            operadorBacterial.resetListas(numeroDeBacterias)

        execution_time = (time.time() - start_time)
        print(f"\n--- Resultados de la Ejecución {ejecucion} ---")
        print("Mejor Fitness Global Encontrado:", veryBestFitness)
        print("Mejor Blosum Score Global:", veryBestBlosum)
        print("Tiempo de Ejecución:", execution_time, "segundos")

        all_results.append({
            'ejecucion': ejecucion,
            'tiempo': execution_time,
            'mejor_fitness': veryBestFitness,
            'mejor_blosum': veryBestBlosum
        })

    csv_file = 'ejecuciones_bacterias_mejoradas.csv'
    fieldnames = ['ejecucion', 'tiempo', 'mejor_fitness', 'mejor_blosum']

    try:
        with open(csv_file, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nLos resultados de las {num_ejecuciones} ejecuciones se han guardado en '{csv_file}'")
    except Exception as e:
        print(f"Error al escribir el archivo CSV: {e}")