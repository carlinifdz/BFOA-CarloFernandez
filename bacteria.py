import copy
import math
from multiprocessing import Manager, Pool, managers
from pickle import FALSE, TRUE
from evaluadorBlosum import evaluadorBlosum
import numpy
from fastaReader import fastaReader
import random
from copy import copy
import copy
import concurrent.futures


class bacteria():

    def __init__(self, numBacterias):
        manager = Manager()
        self.blosumScore = manager.list(range(numBacterias))
        self.tablaAtract = manager.list(range(numBacterias))
        self.tablaRepel = manager.list(range(numBacterias))
        self.tablaInteraction = manager.list(range(numBacterias))
        self.tablaFitness = manager.list(range(numBacterias))
        self.granListaPares = manager.list(range(numBacterias))
        self.NFE = manager.list(range(numBacterias))

    def resetListas(self, numBacterias):
        manager = Manager()
        self.blosumScore = manager.list(range(numBacterias))
        self.tablaAtract = manager.list(range(numBacterias))
        self.tablaRepel = manager.list(range(numBacterias))
        self.tablaInteraction = manager.list(range(numBacterias))
        self.tablaFitness = manager.list(range(numBacterias))
        self.granListaPares = manager.list(range(numBacterias))
        self.NFE = manager.list(range(numBacterias))

    def cuadra(self, numSec, poblacion):
        # ciclo para recorrer poblacion
        for i in range(len(poblacion)):
            # obtiene las secuencias de la bacteria
            bacterTmp_tuple = poblacion[i]
            bacterTmp_list = [list(seq) for seq in bacterTmp_tuple] # Convertir la tupla de tuplas a una lista de listas
            # obtiene el tamaño de la secuencia más larga
            maxLen = 0
            for j in range(numSec):
                if len(bacterTmp_list[j]) > maxLen:
                    maxLen = len(bacterTmp_list[j])
                    # rellena con gaps las secuencias más cortas
                    for t in range(numSec):
                        gap_count = maxLen - len(bacterTmp_list[t])
                        if gap_count > 0:
                            bacterTmp_list[t].extend(["-"] * gap_count)
                            # actualiza la poblacion
                            # Convertir la lista de listas de nuevo a una tupla de tuplas
                            poblacion[i] = tuple(tuple(seq) for seq in bacterTmp_list)

    def limpiaColumnas(self):
        i = 0
        while i < len(self.matrix.seqs[0]):
            if self.gapColumn(i):
                self.deleteCulmn(i)
            else:
                i += 1

    def deleteCulmn(self, pos):
        for i in range(len(self.matrix.seqs)):
            self.matrix.seqs[i] = self.matrix.seqs[i][:pos] + self.matrix.seqs[i][pos+1:]

    def gapColumn(self, col):
        for i in range(len(self.matrix.seqs)):
            if self.matrix.seqs[i][col] != "-":
                return False
        return True

    def tumbo(self, numSec, poblacion, numGaps):
        # inserta un gap en una posicion aleatoria de una secuencia aleatoria
        # recorre la poblacion
        for i in range(len(poblacion)):
            # obtiene las secuencias de la bacteria
            bacterTmp_tuple = poblacion[i]
            bacterTmp_list = [list(seq) for seq in bacterTmp_tuple] # Convertir la tupla de tuplas a una lista de listas
            # ciclo para insertar gaps
            for j in range(numGaps):
                # selecciona secuencia
                seqnum = random.randint(0, len(bacterTmp_list) - 1)
                # selecciona posicion
                pos = random.randint(0, len(bacterTmp_list[seqnum]))
                part1 = bacterTmp_list[seqnum][:pos]
                part2 = bacterTmp_list[seqnum][pos:]
                temp = part1 + ["-"] + part2
                bacterTmp_list[seqnum] = temp
            poblacion[i] = tuple(tuple(seq) for seq in bacterTmp_list)

    def creaGranListaPares(self, poblacion):
        for i in range(len(poblacion)):
            pares = list()
            bacterTmp = list(poblacion[i])
            for j in range(len(bacterTmp)):
                column = self.getColumn(bacterTmp, j)
                pares = pares + self.obtener_pares_unicos(column)
            self.granListaPares[i] = pares

    def evaluaFila(self, fila, num):
        evaluador = evaluadorBlosum()
        score = 0
        for par in fila:
            score += evaluador.getScore(par[0], par[1])
        self.blosumScore[num] = score

    def evaluaBlosum(self):
        with Pool() as pool:
            args = [(copy.deepcopy(self.granListaPares[i]), i) for i in range(len(self.granListaPares))]
            pool.starmap(self.evaluaFila, args)

    def getColumn(self, bacterTmp, colNum):
        column = []
        for i in range(len(bacterTmp)):
            column.append(bacterTmp[i][colNum])
        return column

    def obtener_pares_unicos(self, columna):
        pares_unicos = set()
        for i in range(len(columna)):
            for j in range(i+1, len(columna)):
                par = tuple(sorted([columna[i], columna[j]]))
                pares_unicos.add(par)
        return list(pares_unicos)

    #------------------------------------------------------------Atract y Repel lineal

    def compute_diff(self, args):
        indexBacteria, otherBlosumScore, self.blosumScore, d, w = args
        diff = (self.blosumScore[indexBacteria] - otherBlosumScore) ** 2.0
        self.NFE[indexBacteria] += 1
        return d * numpy.exp(w * diff)

    def compute_cell_interaction(self, indexBacteria, d, w, atracTrue):
        with Pool() as pool:
            args = [(indexBacteria, otherBlosumScore, self.blosumScore, d, w) for otherBlosumScore in self.blosumScore]
            results = pool.map(self.compute_diff, args)
            pool.close()
            pool.join()

        total = sum(results)

        if atracTrue:
            self.tablaAtract[indexBacteria] = total
        else:
            self.tablaRepel[indexBacteria] = total

    def creaTablaAtract(self, poblacion, d, w):         #lineal
        for indexBacteria in range(len(poblacion)):
            self.compute_cell_interaction(indexBacteria, d, w, TRUE)

    def creaTablaRepel(self, poblacion, d, w):         #lineal
        for indexBacteria in range(len(poblacion)):
            self.compute_cell_interaction(indexBacteria, d, w, FALSE)

    def creaTablasAtractRepel(self, poblacion, dAttr, wAttr, dRepel, wRepel):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.creaTablaAtract, poblacion, dAttr, wAttr)
            executor.submit(self.creaTablaRepel, poblacion, dRepel, wRepel)
            #-----------------------------------------------------------

    def creaTablaInteraction(self):
        for i in range(len(self.tablaAtract)):
            self.tablaInteraction[i] = self.tablaAtract[i] + self.tablaRepel[i]

    def creaTablaFitness(self):
        for i in range(len(self.tablaInteraction)):
            valorBlsm = self.blosumScore[i]
            valorInteract = self.tablaInteraction[i]
            valorFitness =  valorBlsm + valorInteract
            self.tablaFitness[i] = valorFitness

    def getNFE(self):
        return sum(self.NFE)

    def obtieneBest(self, globalNFE):
        bestIdx = 0
        for i in range(len(self.tablaFitness)):
            if self.tablaFitness[i] > self.tablaFitness[bestIdx]:
                bestIdx = i
        print("-------------------  Best: ", bestIdx, " Fitness: ", self.tablaFitness[bestIdx], "BlosumScore ",  self.blosumScore[bestIdx], "Interaction: ", self.tablaInteraction[bestIdx], "NFE: ", globalNFE)
        return bestIdx, self.tablaFitness[bestIdx], self.blosumScore[bestIdx], self.tablaInteraction[bestIdx]

    def replaceWorst(self, poblacion, best):
        worst = 0
        for i in range(len(self.tablaFitness)):
            if self.tablaFitness[i] < self.tablaFitness[worst]:
                worst = i
        poblacion[worst] = copy.deepcopy(poblacion[best])

    def mutateBest(self, bestBacteria, numMutations):
        mutatedBacteria_list = list(bestBacteria)
        numSec = len(mutatedBacteria_list)
        for _ in range(numMutations):
            seq_index = random.randint(0, numSec - 1)
            seq = list(mutatedBacteria_list[seq_index])
            if not seq:
                continue
            pos = random.randint(0, len(seq))
            seq.insert(pos, "-")
            mutatedBacteria_list[seq_index] = tuple(seq)
        return tuple(mutatedBacteria_list)

    def refineMejorBacteria(self, bacteria, num_secuencias, probabilidad_mutacion=1, num_mutaciones_max=5):
        bacteria_lista = [list(seq) for seq in bacteria]
        for i in range(num_secuencias):
            if random.random() < probabilidad_mutacion:
                num_mutaciones = random.randint(1, num_mutaciones_max)
                for _ in range(num_mutaciones):
                    seq = list(bacteria_lista[i])
                    if not seq:
                        continue
                    pos = random.randint(0, len(seq))
                    seq.insert(pos, "-")
                    bacteria_lista[i] = tuple(seq)
        return tuple(bacteria_lista)