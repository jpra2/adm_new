import numpy as np
from pymoab import core, types, rng, topo_util
import os
import yaml
import time
import pickle

__all__ = []
parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
bifasico_dir = os.path.join(parent_parent_dir, 'bifasico_v2')
flying_dir_0 = os.path.join(parent_parent_dir, 'flying')
flying_dir = 'flying'
mesh_dir = 'mesh'

class GenerateInitialMeshStructured3D:

    def __init__(self):
        global parent_parent_dir, mesh_dir
        self.dimension = 3
        self.mb = core.Core()
        self.root_set = self.mb.get_root_set()
        self.mtu = topo_util.MeshTopoUtil(self.mb)

        with open("inputs.yaml", 'r') as stream:
            data_loaded = yaml.load(stream)

        input_name = data_loaded['input_name']
        ext_msh = mesh_dir + '/' +  input_name + '.msh'
        self.ext_out = input_name + '_out_initial_mesh3D'
        self.ext_out_h5m = self.ext_out + '.h5m'
        self.mb.load_file(ext_msh)

        self.all_volumes = self.mb.get_entities_by_dimension(0, self.dimension)
        self.all_nodes = self.mb.get_entities_by_dimension(0, 0)
        self.mtu.construct_aentities(self.all_nodes)
        self.all_faces = self.mb.get_entities_by_dimension(0, 2)
        lx = data_loaded['lx']
        ly = data_loaded['ly']
        lz = data_loaded['lz']
        self.hs = [lx, ly, lz]
        self.Areas = np.array([ly*lz, lx*lz, lx*ly])
        self.mi = 1.0
        self.data = dict()
        self.entities = dict()
        self.tags = dict()
        self.tags_to_infos = dict()
        self.entities_to_tags = dict()

    @staticmethod
    def getting_tag(mb, name, n, t1, t2, create, entitie, tipo, tags, tags_to_infos, entities_to_tags):
        types_data = ['meshset', 'integer', 'array', 'double']
        entities = ['volumes', 'root_set', 'intern_faces', 'boundary_faces', 'faces', 'nodes', 'vols_viz_face',
                    'coarse_volumes_lv1', 'coarse_volumes_lv2']

        assert tipo in types_data, f'tipo nao listado: {tipo}'

        if entitie not in entities:
            raise NameError(f'\nA entidade {entitie} nao esta na lista\n')

        tag = mb.tag_get_handle(name, n, t1, t2, create)
        tag_to_infos = dict(zip(['entitie', 'type', 'n'], [entitie, tipo, n]))
        tags[name] = tag
        tags_to_infos[name] = tag_to_infos
        names_tags = entities_to_tags.setdefault(entitie, [])
        if set([name]) & set(names_tags):
            pass
        else:
            entities_to_tags[entitie].append(name)

    def create_tags(self):

        l = ['PERM']
        for name in l:
            n = 9
            tipo = 'array'
            entitie = 'volumes'
            t1 = types.MB_TYPE_DOUBLE
            t2 = types.MB_TAG_SPARSE
            self.getting_tag(self.mb, name, n, t1, t2, True, entitie, tipo, self.tags, self.tags_to_infos, self.entities_to_tags)

        l = ['BOUNDARY_FACES', 'INTERN_FACES']
        for name in l:
            n = 1
            tipo = 'meshset'
            entitie = 'root_set'
            t1 = types.MB_TYPE_HANDLE
            t2 = types.MB_TAG_MESH
            self.getting_tag(self.mb, name, n, t1, t2, True, entitie, tipo, self.tags, self.tags_to_infos, self.entities_to_tags)

        l = ['VOLS_VIZ_FACE']
        for name in l:
            n = 2
            tipo = 'array'
            entitie = 'intern_faces'
            t1 = types.MB_TYPE_HANDLE
            t2 = types.MB_TAG_SPARSE
            self.getting_tag(self.mb, name, n, t1, t2, True, entitie, tipo, self.tags, self.tags_to_infos, self.entities_to_tags)

        l = ['GIDS_VOLS']
        for name in l:
            n = 1
            tipo = 'integer'
            entitie = 'volumes'
            t1 = types.MB_TYPE_INTEGER
            t2 = types.MB_TAG_SPARSE
            self.getting_tag(self.mb, name, n, t1, t2, True, entitie, tipo, self.tags, self.tags_to_infos, self.entities_to_tags)

        l = ['GIDS_FACES']
        for name in l:
            n = 1
            tipo = 'integer'
            entitie = 'faces'
            t1 = types.MB_TYPE_INTEGER
            t2 = types.MB_TAG_SPARSE
            self.getting_tag(self.mb, name, n, t1, t2, True, entitie, tipo, self.tags, self.tags_to_infos, self.entities_to_tags)

        l = ['GIDS_INTERN_FACES']
        for name in l:
            n = 1
            tipo = 'integer'
            entitie = 'intern_faces'
            t1 = types.MB_TYPE_INTEGER
            t2 = types.MB_TAG_SPARSE
            self.getting_tag(self.mb, name, n, t1, t2, True, entitie, tipo, self.tags, self.tags_to_infos, self.entities_to_tags)

        l = ['CENT_VOLS']
        for name in l:
            n = 3
            tipo = 'array'
            entitie = 'volumes'
            t1 = types.MB_TYPE_DOUBLE
            t2 = types.MB_TAG_SPARSE
            self.getting_tag(self.mb, name, n, t1, t2, True, entitie, tipo, self.tags, self.tags_to_infos, self.entities_to_tags)

        l = ['UNITARY_FACE']
        for name in l:
            n = 3
            tipo = 'array'
            entitie = 'intern_faces'
            t1 = types.MB_TYPE_DOUBLE
            t2 = types.MB_TAG_SPARSE
            self.getting_tag(self.mb, name, n, t1, t2, True, entitie, tipo, self.tags, self.tags_to_infos, self.entities_to_tags)

        l = ['PHI', 'VOLUME']
        for name in l:
            n = 1
            tipo = 'double'
            entitie = 'volumes'
            t1 = types.MB_TYPE_DOUBLE
            t2 = types.MB_TAG_SPARSE
            self.getting_tag(self.mb, name, n, t1, t2, True, entitie, tipo, self.tags, self.tags_to_infos, self.entities_to_tags)

        l = ['AREA', 'KHARM', 'DIST_CENTROIDS', 'TRANSMISSIBILITY']
        for name in l:
            n = 1
            tipo = 'double'
            entitie = 'intern_faces'
            t1 = types.MB_TYPE_DOUBLE
            t2 = types.MB_TAG_SPARSE
            self.getting_tag(self.mb, name, n, t1, t2, True, entitie, tipo, self.tags, self.tags_to_infos, self.entities_to_tags)

        l = ['NODES']
        for name in l:
            n = 3
            tipo = 'double'
            entitie = 'nodes'
            t1 = types.MB_TYPE_DOUBLE
            t2 = types.MB_TAG_SPARSE
            self.getting_tag(self.mb, name, n, t1, t2, True, entitie, tipo, self.tags, self.tags_to_infos, self.entities_to_tags)

    def get_boundary_faces(self):
        all_boundary_faces = self.mb.create_meshset()
        intern_faces_set = self.mb.create_meshset()
        for face in self.all_faces:
            elems = self.mtu.get_bridge_adjacencies(face, 2, 3)
            if len(elems) < 2:
                self.mb.add_entities(all_boundary_faces, [face])

        self.mb.tag_set_data(self.tags['BOUNDARY_FACES'], 0, all_boundary_faces)
        self.all_boundary_faces = self.mb.get_entities_by_handle(all_boundary_faces)
        self.intern_faces = rng.subtract(self.all_faces, self.all_boundary_faces)
        self.mb.add_entities(intern_faces_set, self.intern_faces)
        self.mb.tag_set_data(self.tags['INTERN_FACES'], 0, intern_faces_set)
        self.mb.tag_set_data(self.tags['GIDS_INTERN_FACES'], self.intern_faces, np.arange(len(self.intern_faces)))

    def get_viz_faces(self):
        ADJs=np.array([self.mb.get_adjacencies(face, 3) for face in self.all_faces])

        self.viz_faces = np.array([np.array(self.mb.get_adjacencies(face, 3)) for face in self.intern_faces])
        self.mb.tag_set_data(self.tags['VOLS_VIZ_FACE'], self.intern_faces, self.viz_faces)

    def get_centroid_volumes(self):
        self.centroid_vols = np.array([self.mtu.get_average_position([v]) for v in self.all_volumes])
        self.mb.tag_set_data(self.tags['CENT_VOLS'], self.all_volumes, self.centroid_vols)

    def get_ids_vols_faces(self):
        self.mb.tag_set_data(self.tags['GIDS_VOLS'], self.all_volumes, np.arange(len(self.all_volumes)))
        self.mb.tag_set_data(self.tags['GIDS_FACES'], self.all_faces, np.arange(len(self.all_faces)))

    def set_k_and_phi_structured_spe10(self):
        global parent_parent_dir
        ks = np.load('spe10_perms_and_phi.npz')['perms']
        phi = np.load('spe10_perms_and_phi.npz')['phi']

        nx = 60
        ny = 220
        nz = 85
        perms = []
        phis = []

        k = 1.0  #para converter a unidade de permeabilidade
        centroids = self.centroid_vols
        cont=0
        for v in self.all_volumes:
            centroid = centroids[cont]
            cont+=1
            ijk = np.array([centroid[0]//20.0, centroid[1]//10.0, centroid[2]//2.0])
            e = int(ijk[0] + ijk[1]*nx + ijk[2]*nx*ny)
            # perm = ks[e]*k
            # fi = phi[e]
            perms.append(ks[e]*k)
            phis.append(phi[e])

        self.mb.tag_set_data(self.tags['PERM'], self.all_volumes, perms)
        self.mb.tag_set_data(self.tags['PHI'], self.all_volumes, phis)

    def set_kharm_faces(self):

        vols_0 = self.viz_faces[:,0]
        vols_1 = self.viz_faces[:,1]
        centroids = self.centroid_vols

        ids_vols_0 = self.mb.tag_get_data(self.tags['GIDS_VOLS'], np.array(vols_0), flat=True)
        ids_vols_1 = self.mb.tag_get_data(self.tags['GIDS_VOLS'], np.array(vols_1), flat=True)
        self.perms = self.mb.tag_get_data(self.tags['PERM'], self.all_volumes, flat=True)
        k0s = self.perms[ids_vols_0]
        k1s = self.perms[ids_vols_1]

        centroids_0 = centroids[ids_vols_0]
        centroids_1 = centroids[ids_vols_1]
        dX = np.absolute(centroids_1 - centroids_0)
        norms = np.linalg.norm(dX, axis=1)
        self.mb.tag_set_data(self.tags['DIST_CENTROIDS'], self.intern_faces, norms)
        n = len(norms)
        unitarys = np.zeros([n, 3])
        unitarys[:,0] = np.divide(dX[:,0], norms)
        unitarys[:,1] = np.divide(dX[:,1], norms)
        unitarys[:,2] = np.divide(dX[:,2], norms)
        # unitarys = np.absolute(unitarys)
        self.mb.tag_set_data(self.tags['UNITARY_FACE'], self.intern_faces, unitarys)

        area_s = np.zeros(n)
        kharm_s = area_s.copy()
        for i in range(n):
            area = np.dot(unitarys[i], self.Areas)
            area_s[i] = area
            k0 = np.dot(k0s[i], unitarys[i])
            k0 = np.dot(k0, unitarys[i])
            k1 = np.dot(k1s[i], unitarys[i])
            k1 = np.dot(k1, unitarys[i])
            kharm = 2*k1*k0/(k0 + k1)
            kharm_s[i] = kharm

        self.mb.tag_set_data(self.tags['AREA'], self.intern_faces, area_s)
        self.mb.tag_set_data(self.tags['KHARM'], self.intern_faces, kharm_s)
        transm_s = np.divide(area_s * kharm_s, norms)
        self.mb.tag_set_data(self.tags['TRANSMISSIBILITY'], self.intern_faces, transm_s)

    def get_node_coords(self):

        coords = self.mb.get_coords(self.all_nodes)
        n = len(self.all_nodes)
        coords = coords.reshape([n, 3])

        self.mb.tag_set_data(self.tags['NODES'], self.all_nodes, coords)

    def save_initial_mesh(self):
        global flying_dir

        import pdb; pdb.set_trace()

        names_tags = list(self.tags.keys())
        file_names_tags = os.path.join(flying_dir, 'names_tags_out_initial_mesh.txt')

        with open(file_names_tags, 'wb') as handle:
            pickle.dump(names_tags, handle)

        file_name_initial_mesh = os.path.join(flying_dir, 'name_out_initial_mesh.txt')

        with open(file_name_initial_mesh, 'wb') as handle:
            pickle.dump(self.ext_out_h5m, handle)


        file_name_tags_to_infos = os.path.join(flying_dir, 'tags_to_infos_initial_mesh.txt')
        with open(file_name_tags_to_infos, 'wb') as handle:
            pickle.dump(self.tags_to_infos, handle)

        file_name_entities_to_tags = os.path.join(flying_dir, 'entities_to_tags_initial_mesh.txt')
        with open(file_name_entities_to_tags, 'wb') as handle:
            pickle.dump(self.entities_to_tags, handle)

        # with open('file.txt', 'rb') as handle:
        #     b = pickle.loads(handle.read())
        file_name_h5m_out = os.path.join(flying_dir, self.ext_out_h5m)
        self.mb.write_file(file_name_h5m_out)

    def set_volumes(self):
        vol = float(self.hs[0]*self.hs[1]*self.hs[2])
        n = len(self.all_volumes)

        self.mb.tag_set_data(self.tags['VOLUME'], self.all_volumes, np.repeat(vol, n))

    def run(self):
        t1 = time.time()
        self.create_tags()
        self.get_boundary_faces()
        self.get_viz_faces()
        self.get_centroid_volumes()
        self.get_ids_vols_faces()
        self.set_k_and_phi_structured_spe10()
        self.set_kharm_faces()
        self.get_node_coords()
        self.set_volumes()
        t2 = time.time()
        print(f'\ntempo para gerar malha inicial: {t2 - t1}\n')

        self.save_initial_mesh()
