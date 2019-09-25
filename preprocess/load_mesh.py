import numpy as np
from pymoab import core, types, rng, topo_util
import yaml
import time
import pickle
import os

# parent_dir = os.path.dirname(os.path.abspath(__file__))
# dir_from = parent_dir
# parent_parent_dir = os.path.dirname(parent_dir)
# bifasico_dir = os.path.join(parent_parent_dir, 'bifasico_v2')
# flying_dir_0 = os.path.join(parent_parent_dir, 'flying')
flying_dir = 'flying'
mesh_dir = 'mesh'

class LoadMesh:

    def __init__(self):
        self.mb = core.Core()
        self.root_set = self.mb.get_root_set()
        self.mtu = topo_util.MeshTopoUtil(self.mb)
        self.dimension = 3
        self.tags = dict()
        self.datas = dict()
        self.excluir = []
        self.n_levels = 2
        with open("inputs.yaml", 'r') as stream:
            data_loaded = yaml.load(stream)
        self.data_loaded = data_loaded

    def load_entities_0(self):

        volumes = self.mb.get_entities_by_dimension(0, self.dimension)
        nodes = self.mb.get_entities_by_dimension(0, 0)
        self.mtu.construct_aentities(nodes)
        faces = self.mb.get_entities_by_dimension(0, 2)

        self.entities = dict()

        self.entities['volumes'] = volumes
        self.entities['faces'] = faces
        self.entities['nodes'] = nodes

        boundary_faces = self.mb.tag_get_data(self.tags['BOUNDARY_FACES'], 0, flat=True)[0]
        boundary_faces = self.mb.get_entities_by_handle(boundary_faces)
        self.entities['boundary_faces'] = boundary_faces
        intern_faces = self.mb.tag_get_data(self.tags['INTERN_FACES'], 0, flat=True)[0]
        intern_faces = self.mb.get_entities_by_handle(intern_faces)
        self.entities['intern_faces'] = intern_faces
        self.entities['vols_viz_face'] = self.mb.tag_get_data(self.tags['VOLS_VIZ_FACE'], intern_faces)

        excluir_nomes = ['BOUNDARY_FACES', 'INTERN_FACES', 'VOLS_VIZ_FACE']

        return excluir_nomes

    def load_entities_1(self):

        names_tags = ['COARSE_VOLUMES_LV_', 'FACES_BOUNDARY_MESHSETS_LEVEL_', 'VIZINHOS_FACE_LV_',
                      'PRIMAL_ID_']
        excluir_nomes = []

        for i in range(self.n_levels):
            coarse_volume_name = 'coarse_volumes_lv'+str(i+1)
            face_boundary_name = 'faces_boundary_meshsets_level_'+str(i+1)
            excluir_nomes.append(names_tags[0]+str(i+1))
            excluir_nomes.append(names_tags[1]+str(i+1))
            coarse_volumes = self.mb.tag_get_data(self.tags[names_tags[0]+str(i+1)], 0, flat=True)
            self.entities[coarse_volume_name] = coarse_volumes
            self.datas[names_tags[0]+str(i+1)] = coarse_volumes
            faces_boundary = self.mb.tag_get_data(self.tags[names_tags[1]+str(i+1)], 0, flat=True)[0]
            faces_boundary = self.mb.get_entities_by_handle(faces_boundary)
            self.datas[names_tags[1]+str(i+1)] = faces_boundary
            vizinhos_face = self.mb.tag_get_data(self.tags[names_tags[2]+str(i+1)], coarse_volumes)
            self.datas[names_tags[2]+str(i+1)] = vizinhos_face
            primais_ids = self.mb.tag_get_data(self.tags[names_tags[3]+str(i+1)], coarse_volumes, flat=True)
            self.datas[names_tags[3]+str(i+1)] = primais_ids
            excluir_nomes.append(names_tags[2]+str(i+1))
            excluir_nomes.append(names_tags[3]+str(i+1))

        l2 = ['L2_MESHSET']
        l2_meshset = self.mb.tag_get_data(self.tags[l2[0]], 0, flat=True)[0]
        self.datas[l2[0]] = l2_meshset
        excluir_nomes.append(l2[0])

        return excluir_nomes

    def load_initial_mesh(self):

        global flying_dir
        name_out_initial_mesh_file = 'name_out_initial_mesh.txt'
        names_tags_out_initial_mesh_file = 'names_tags_out_initial_mesh.txt'
        entities_to_tags_file = 'entities_to_tags_initial_mesh.txt'
        tags_to_infos_file = 'tags_to_infos_initial_mesh.txt'

        with open(os.path.join(flying_dir, name_out_initial_mesh_file), 'rb') as handle:
            file_name = pickle.loads(handle.read())

        file_name = os.path.join(flying_dir, file_name)

        self.mb.load_file(file_name)

        with open(os.path.join(flying_dir, entities_to_tags_file), 'rb') as handle:
            self.entities_to_tags = pickle.loads(handle.read())
        with open(os.path.join(flying_dir, tags_to_infos_file), 'rb') as handle:
            self.tags_to_infos = pickle.loads(handle.read())

        for name in self.tags_to_infos.keys():
            try:
                self.tags[name] = self.mb.tag_get_handle(name)
            except:
                raise NameError(f'A tag {name} nao esta no arquivo')

        self.excluir += self.load_entities_0()

        self.datas = dict()
        self.datas.update(self.loading_datas(self.mb, self.tags, self.excluir, self.tags_to_infos, self.entities))

    @staticmethod
    def loading_datas(mb, tags, excluir, tags_to_infos, entities):
        datas = dict()
        lista = set(list(tags.keys())) - set(excluir)
        for name in lista:
            infos = tags_to_infos[name]
            n = infos['n']
            tipo = infos['type']
            entitie_name = infos['entitie']
            if entitie_name == 'root_set':
                import pdb; pdb.set_trace()
            entitie = entities[entitie_name]
            if n > 1:
                datas[name] = mb.tag_get_data(tags[name], entitie)
            else:
                datas[name] = mb.tag_get_data(tags[name], entitie, flat=True)
            excluir.append(name)

        return datas

    def load_determinated_mesh(self, file_name_file_h5m, file_name_tags_out, file_name_entities_to_tags, file_name_tags_to_infos):

        with open(file_name_file_h5m, 'rb') as handle:
            file_name = pickle.loads(handle.read())

        file_name = os.path.join('flying', file_name)

        self.mb.load_file(file_name)

        with open(file_name_entities_to_tags, 'rb') as handle:
            self.entities_to_tags = pickle.loads(handle.read())
        with open(file_name_tags_to_infos, 'rb') as handle:
            self.tags_to_infos = pickle.loads(handle.read())

        for name in self.tags_to_infos.keys():
            try:
                self.tags[name] = self.mb.tag_get_handle(name)
            except:
                raise NameError(f'A tag {name} nao esta no arquivo')

    def run(self, n):
        global flying_dir

        if n in [1, 2]:
            t0 = time.time()
            print('\ncarregando malha inicial\n')
            self.load_initial_mesh()
            print('carregou malha inicial')
            t1 = time.time()
            print(f'\ntempo para carregar malha inicial {t1 - t0}\n')
        elif n == 3:
            t0 = time.time()
            file_name_file_h5m = os.path.join(flying_dir, 'name_out_dual_mesh.txt')
            file_name_tags_out = os.path.join(flying_dir, 'names_tags_out_dual_mesh.txt')
            file_name_entities_to_tags = os.path.join(flying_dir, 'entities_to_tags_dual_mesh.txt')
            file_name_tags_to_infos = os.path.join(flying_dir, 'tags_to_infos_dual_mesh.txt')
            self.load_determinated_mesh(file_name_file_h5m, file_name_tags_out, file_name_entities_to_tags, file_name_tags_to_infos)
            t1 = time.time()
            print(f'\ntempo para carregar malha dual {t1 - t0}\n')
        else:
            raise ValueError('Digite um valor adequado')
        if n in [3]:
            self.excluir += self.load_entities_0()
            self.excluir += self.load_entities_1()
            self.datas.update(self.loading_datas(self.mb, self.tags, self.excluir, self.tags_to_infos, self.entities))
