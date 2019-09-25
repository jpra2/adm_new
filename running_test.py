import os

path_ant = os.getcwd()
parent_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(parent_dir)

from rodar_simulacao import RodarSimulacao

RodarSimulacao(3)

os.chdir(path_ant)
