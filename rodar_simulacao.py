import os
import time


parent_dir = os.path.dirname(os.path.abspath(__file__))
dir_from = parent_dir
parent_parent_dir = os.path.dirname(parent_dir)
# bifasico_dir = os.path.join(parent_parent_dir, 'bifasico_v2')
# flying_dir_0 = os.path.join(parent_parent_dir, 'flying')

class RodarSimulacao:
    passou1 = False

    def __init__(self, n, rodar_ate_n2=3):

        """
        numbers:
            0: gerar malha inicial
            1: carregar malha inicial
        """

        ns = [0, 1, 2, 3]
        self.n = n
        self.n2 = rodar_ate_n2
        assert self.n2 >= n, 'n2 deve ser maior que n1'

        while(self.n <= rodar_ate_n2):
            ok = self.run()
            self.n += 1

    def initial_mesh(self):
        print('\ngenerate initial mesh\n')
        from preprocess.generate_mesh0 import GenerateInitialMeshStructured3D
        mesh = GenerateInitialMeshStructured3D()
        mesh.run()

    def take_initial_mesh(self):

        print('\ntaking initial mesh\n')
        from preprocess.load_mesh import LoadMesh
        self.mesh = LoadMesh()
        self.mesh.run(self.n)

    def create_dual_mesh(self):
        print('generate dual mesh')
        from preprocess.generate_dual_mesh import GenerateDualMeshMultinivel
        dual_mesh = GenerateDualMeshMultinivel(self.mesh)
        dual_mesh.run()

    def load_dual_mesh(self):
        print('\ntaking dual mesh\n')
        from preprocess.load_mesh import LoadMesh
        self.mesh = LoadMesh()
        self.mesh.run(self.n)

    def run(self):

        if self.n == 0:
            self.initial_mesh()
        if self.n in [1, 2] and not RodarSimulacao.passou1:
            self.take_initial_mesh()
            RodarSimulacao.passou1 = True
        if self.n == 2:
            self.create_dual_mesh()
        if self.n == 3:
            self.load_dual_mesh()
        if self.n == 4:
            # TODO: setar condicoes de contorno
            pass

        return 0
