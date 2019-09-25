from .generate_mesh0 import getting_tag
# getting_tag = GenerateInitialMeshStructured3D.getting_tag
import time
from pymoab import core, types, rng, topo_util
import yaml
import numpy as np
import os
import pickle

flying_dir = 'flying'
mesh_dir = 'mesh'

def get_box(conjunto, all_centroids, limites, return_inds):
    # conjunto-> lista
    # all_centroids->coordenadas dos centroides do conjunto
    # limites-> diagonal que define os volumes objetivo (numpy array com duas coordenadas)
    # Retorna os volumes pertencentes a conjunto cujo centroide está dentro de limites
    inds0 = np.where(all_centroids[:,0] > limites[0,0])[0]
    inds1 = np.where(all_centroids[:,1] > limites[0,1])[0]
    inds2 = np.where(all_centroids[:,2] > limites[0,2])[0]
    c1 = set(inds0) & set(inds1) & set(inds2)
    inds0 = np.where(all_centroids[:,0] < limites[1,0])[0]
    inds1 = np.where(all_centroids[:,1] < limites[1,1])[0]
    inds2 = np.where(all_centroids[:,2] < limites[1,2])[0]
    c2 = set(inds0) & set(inds1) & set(inds2)
    inds_vols = list(c1 & c2)
    if return_inds:
        return (rng.Range(np.array(conjunto)[inds_vols]),inds_vols)
    else:
        return rng.Range(np.array(conjunto)[inds_vols])

def Min_Max(e, M1):
    verts = M1.mb.get_connectivity(e)
    coords = np.array([M1.mb.get_coords([vert]) for vert in verts])
    xmin, xmax = coords[0][0], coords[0][0]
    ymin, ymax = coords[0][1], coords[0][1]
    zmin, zmax = coords[0][2], coords[0][2]
    for c in coords:
        if c[0]>xmax: xmax=c[0]
        if c[0]<xmin: xmin=c[0]
        if c[1]>ymax: ymax=c[1]
        if c[1]<ymin: ymin=c[1]
        if c[2]>zmax: zmax=c[2]
        if c[2]<zmin: zmin=c[2]
    return([xmin,xmax,ymin,ymax,zmin,zmax])

class GenerateDualMeshMultinivel:

    def __init__(self, mesh):
        self.mesh = mesh
        with open("inputs.yaml", 'r') as stream:
            self.data_loaded = yaml.load(stream)

        input_name = self.data_loaded['input_name']
        ext_out_dual_mesh = '_out_dual_mesh3D'
        self.ext_out = input_name + ext_out_dual_mesh
        self.ext_out_h5m = self.ext_out + '.h5m'

        self.n_levels = 2

    def create_tags_dual(self):

        l = ['D1', 'D2', 'FINE_TO_PRIMAL_CLASSIC_1', 'FINE_TO_PRIMAL_CLASSIC_2']
        for name in l:
            n = 1
            tipo = 'integer'
            entitie = 'volumes'
            t1 = types.MB_TYPE_INTEGER
            t2 = types.MB_TAG_SPARSE
            getting_tag(self.mesh.mb, name, n, t1, t2, True, entitie, tipo, self.mesh.tags, self.mesh.tags_to_infos, self.mesh.entities_to_tags)

        l = ['PRIMAL_ID_1']
        for name in l:
            n = 1
            tipo = 'integer'
            entitie = 'coarse_volumes_lv1'
            t1 = types.MB_TYPE_INTEGER
            t2 = types.MB_TAG_SPARSE
            getting_tag(self.mesh.mb, name, n, t1, t2, True, entitie, tipo, self.mesh.tags, self.mesh.tags_to_infos, self.mesh.entities_to_tags)

        l = ['PRIMAL_ID_2']
        for name in l:
            n = 1
            tipo = 'integer'
            entitie = 'coarse_volumes_lv2'
            t1 = types.MB_TYPE_INTEGER
            t2 = types.MB_TAG_SPARSE
            getting_tag(self.mesh.mb, name, n, t1, t2, True, entitie, tipo, self.mesh.tags, self.mesh.tags_to_infos, self.mesh.entities_to_tags)

        l = ['L2_MESHSET']
        for name in l:
            n = 1
            tipo = 'meshset'
            entitie = 'root_set'
            t1 = types.MB_TYPE_HANDLE
            t2 = types.MB_TAG_MESH
            getting_tag(self.mesh.mb, name, n, t1, t2, True, entitie, tipo, self.mesh.tags, self.mesh.tags_to_infos, self.mesh.entities_to_tags)

    def generate_dual_and_primal(self):

        cr1 = self.data_loaded['Crs']['Cr1']
        cr2 = self.data_loaded['Crs']['Cr2']

        nx = self.data_loaded['nx']
        ny = self.data_loaded['ny']
        nz = self.data_loaded['nz']

        lx = self.data_loaded['lx']
        ly = self.data_loaded['ly']
        lz = self.data_loaded['lz']
        dx0 = lx
        dy0 = ly
        dz0 = lz

        Lx, Ly, Lz = self.mesh.datas['NODES'].max(axis = 0)
        xmin, ymin, zmin = self.mesh.datas['NODES'].min(axis = 0)
        xmax, ymax, zmax = Lx, Ly, Lz

        l1 = [cr1[0]*lx,cr1[1]*ly,cr1[2]*lz]
        l2 = [cr2[0]*lx,cr2[1]*ly,cr2[2]*lz]

        x1=nx*lx
        y1=ny*ly
        z1=nz*lz

        L2_meshset = self.mesh.mb.create_meshset()
        self.mesh.mb.tag_set_data(self.mesh.tags['L2_MESHSET'], 0, L2_meshset)

        lx2, ly2, lz2 = [], [], []
        # O valor 0.01 é adicionado para corrigir erros de ponto flutuante
        for i in range(int(Lx/l2[0])):    lx2.append(xmin+i*l2[0])
        for i in range(int(Ly/l2[1])):    ly2.append(ymin+i*l2[1])
        for i in range(int(Lz/l2[2])):    lz2.append(zmin+i*l2[2])
        lx2.append(Lx)
        ly2.append(Ly)
        lz2.append(Lz)

        lx1, ly1, lz1 = [], [], []
        for i in range(int(l2[0]/l1[0])):   lx1.append(i*l1[0])
        for i in range(int(l2[1]/l1[1])):   ly1.append(i*l1[1])
        for i in range(int(l2[2]/l1[2])):   lz1.append(i*l1[2])


        D_x=max(Lx-int(Lx/l1[0])*l1[0],Lx-int(Lx/l2[0])*l2[0])
        D_y=max(Ly-int(Ly/l1[1])*l1[1],Ly-int(Ly/l2[1])*l2[1])
        D_z=max(Lz-int(Lz/l1[2])*l1[2],Lz-int(Lz/l2[2])*l2[2])
        nD_x=int((D_x+0.001)/l1[0])
        nD_y=int((D_y+0.001)/l1[1])
        nD_z=int((D_z+0.001)/l1[2])

        lxd1=[xmin+dx0/100]
        for i in range(int(Lx/l1[0])-2-nD_x):
            lxd1.append(l1[0]/2+(i+1)*l1[0])
        lxd1.append(xmin+Lx-dx0/100)

        lyd1=[ymin+dy0/100]
        for i in range(int(Ly/l1[1])-2-nD_y):
            lyd1.append(l1[1]/2+(i+1)*l1[1])
        lyd1.append(ymin+Ly-dy0/100)

        lzd1=[zmin+dz0/100]

        for i in range(int(Lz/l1[2])-2-nD_z):
            lzd1.append(l1[2]/2+(i+1)*l1[2])
        lzd1.append(xmin+Lz-dz0/100)

        print("definiu planos do nível 1")
        lxd2=[lxd1[0]]
        for i in range(1,int(len(lxd1)*l1[0]/l2[0])-1):
            lxd2.append(lxd1[int(i*l2[0]/l1[0]+0.0001)+1])
        lxd2.append(lxd1[-1])

        lyd2=[lyd1[0]]
        for i in range(1,int(len(lyd1)*l1[1]/l2[1])-1):
            lyd2.append(lyd1[int(i*l2[1]/l1[1]+0.00001)+1])
        lyd2.append(lyd1[-1])

        lzd2=[lzd1[0]]
        for i in range(1,int(len(lzd1)*l1[2]/l2[2])-1):
            lzd2.append(lzd1[int(i*l2[2]/l1[2]+0.00001)+1])
        lzd2.append(lzd1[-1])

        print("definiu planos do nível 2")

        centroids = self.mesh.datas['CENT_VOLS']
        volumes = self.mesh.entities['volumes']

        D1_tag = self.mesh.tags['D1']
        D2_tag = self.mesh.tags['D2']
        primal_id_tag1 = self.mesh.tags['PRIMAL_ID_1']
        primal_id_tag2 = self.mesh.tags['PRIMAL_ID_2']
        fine_to_primal1_classic_tag = self.mesh.tags['FINE_TO_PRIMAL_CLASSIC_1']
        fine_to_primal2_classic_tag = self.mesh.tags['FINE_TO_PRIMAL_CLASSIC_2']

        nc1=0
        nc2=0

        # add_parent_child(self, parent_meshset, child_meshset, exceptions = ()):
        ##-----------------------------------------------------------------
        for i in range(len(lx2)-1):
            t1=time.time()
            if i==len(lx2)-2:
                sx=D_x
            sy=0

            #################################################
            x0=lx2[i]
            x1=lx2[i+1]
            box_x=np.array([[x0-0.01,ymin,zmin],[x1+0.01,ymax,zmax]])
            vols_x=get_box(volumes, centroids, box_x, False)
            x_centroids=np.array([self.mesh.mtu.get_average_position([v]) for v in vols_x])
            ######################################

            for j in range(len(ly2)-1):
                if j==len(ly2)-2:
                    sy=D_y
                sz=0
                #########################
                y0=ly2[j]
                y1=ly2[j+1]
                box_y=np.array([[x0-0.01,y0-0.01,zmin],[x1+0.01,y1+0.01,zmax]])
                vols_y=get_box(vols_x, x_centroids, box_y, False)
                y_centroids=np.array([self.mesh.mtu.get_average_position([v]) for v in vols_y])
                ###############
                for k in range(len(lz2)-1):
                    if k==len(lz2)-2:
                        sz=D_z
                    ########################################
                    z0=lz2[k]
                    z1=lz2[k+1]
                    tb=time.time()
                    box_dual_1=np.array([[x0-0.01,y0-0.01,z0-0.01],[x1+0.01,y1+0.01,z1+0.01]])
                    vols=get_box(vols_y, y_centroids, box_dual_1, False)
                    ####################
                    l2_meshset=self.mesh.mb.create_meshset()
                    cont=0
                    elem_por_L2=vols
                    self.mesh.mb.add_entities(l2_meshset,elem_por_L2)
                    centroid_p2=np.array([self.mesh.mtu.get_average_position([np.uint64(v)]) for v in elem_por_L2])
                    cx,cy,cz=centroid_p2[:,0],centroid_p2[:,1],centroid_p2[:,2]
                    posx=np.where(abs(cx-lxd2[i])<=l1[0]/1.9)[0]
                    posy=np.where(abs(cy-lyd2[j])<=l1[1]/1.9)[0]
                    posz=np.where(abs(cz-lzd2[k])<=l1[2]/1.9)[0]
                    f1a2v3=np.zeros(len(elem_por_L2),dtype=int)
                    f1a2v3[posx]+=1
                    f1a2v3[posy]+=1
                    f1a2v3[posz]+=1
                    self.mesh.mb.tag_set_data(D2_tag, elem_por_L2, f1a2v3)
                    self.mesh.mb.tag_set_data(fine_to_primal2_classic_tag, elem_por_L2, np.repeat(nc2,len(elem_por_L2)))
                    self.mesh.mb.add_parent_child(L2_meshset,l2_meshset)
                    sg=self.mesh.mb.get_entities_by_handle(l2_meshset)
                    print(k, len(sg), time.time()-t1)
                    t1=time.time()
                    self.mesh.mb.tag_set_data(primal_id_tag2, l2_meshset, nc2)
                    centroids_primal2=np.array([self.mesh.mtu.get_average_position([np.uint64(v)]) for v in elem_por_L2])
                    nc2+=1
                    s1x=0
                    for m in range(len(lx1)):
                        a=int(l2[0]/l1[0])*i+m
                        if Lx-D_x==lx2[i]+lx1[m]+l1[0]:# and D_x==Lx-int(Lx/l1[0])*l1[0]:
                            s1x=D_x
                        s1y=0
                        for n in range(len(ly1)):
                            b=int(l2[1]/l1[1])*j+n
                            if Ly-D_y==ly2[j]+ly1[n]+l1[1]:# and D_y==Ly-int(Ly/l1[1])*l1[1]:
                                s1y=D_y
                            s1z=0

                            for o in range(len(lz1)):
                                c=int(l2[2]/l1[2])*k+o
                                if Lz-D_z==lz2[k]+lz1[o]+l1[2]:
                                    s1z=D_z
                                l1_meshset=self.mesh.mb.create_meshset()
                                box_primal1 = np.array([np.array([lx2[i]+lx1[m], ly2[j]+ly1[n], lz2[k]+lz1[o]]), np.array([lx2[i]+lx1[m]+l1[0]+s1x, ly2[j]+ly1[n]+l1[1]+s1y, lz2[k]+lz1[o]+l1[2]+s1z])])
                                elem_por_L1 = get_box(elem_por_L2, centroids_primal2, box_primal1, False)
                                self.mesh.mb.add_entities(l1_meshset,elem_por_L1)
                                cont1=0
                                values_1=[]
                                for e in elem_por_L1:
                                    cont1+=1
                                    f1a2v3=0
                                    M_M=Min_Max(e, self.mesh)
                                    if (M_M[0]<lxd1[a] and M_M[1]>=lxd1[a]):
                                        f1a2v3+=1
                                    if (M_M[2]<lyd1[b] and M_M[3]>=lyd1[b]):
                                        f1a2v3+=1
                                    if (M_M[4]<lzd1[c] and M_M[5]>=lzd1[c]):
                                        f1a2v3+=1
                                    values_1.append(f1a2v3)
                                self.mesh.mb.tag_set_data(D1_tag, elem_por_L1,values_1)
                                self.mesh.mb.tag_set_data(fine_to_primal1_classic_tag, elem_por_L1, np.repeat(nc1,len(elem_por_L1)))
                                self.mesh.mb.tag_set_data(primal_id_tag1, l1_meshset, nc1)
                                nc1+=1
                                self.mesh.mb.add_parent_child(l2_meshset,l1_meshset)
        #-------------------------------------------------------------------------------

    def get_boundary_coarse_faces(self):
        meshsets_nv1 = self.mesh.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([self.mesh.tags['PRIMAL_ID_1']]), np.array([None]))
        meshsets_nv2 = self.mesh.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([self.mesh.tags['PRIMAL_ID_2']]), np.array([None]))

        n_levels = self.n_levels

        name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'
        all_meshsets = [meshsets_nv1, meshsets_nv2]

        from utils import pymoab_utils as utpy

        for i in range(n_levels):
            name = name_tag_faces_boundary_meshsets + str(i+1)
            meshsets = all_meshsets[i]
            n = 1
            tipo = 'meshset'
            entitie = 'root_set'
            t1 = types.MB_TYPE_HANDLE
            t2 = types.MB_TAG_MESH
            getting_tag(self.mesh.mb, name, n, t1, t2, True, entitie, tipo, self.mesh.tags, self.mesh.tags_to_infos, self.mesh.entities_to_tags)
            tag_boundary = self.mesh.tags[name]
            utpy.set_faces_in_boundary_by_meshsets(self.mesh.mb, self.mesh.mtu, meshsets, tag_boundary)

            name = 'COARSE_VOLUMES_LV_' + str(i+1)
            n = len(meshsets)
            tipo = 'array'
            entitie = 'root_set'
            t1 = types.MB_TYPE_HANDLE
            t2 = types.MB_TAG_MESH
            getting_tag(self.mesh.mb, name, n, t1, t2, True, entitie, tipo, self.mesh.tags, self.mesh.tags_to_infos, self.mesh.entities_to_tags)
            self.mesh.mb.tag_set_data(self.mesh.tags[name], 0, meshsets)
            self.mesh.entities['coarse_volumes_lv' + str(i+1)] = meshsets

    def get_vizinhos_de_face(self):

        n_levels = self.n_levels
        # n_levels = 1 # colocar vizinhos de face apenas no nivel grosso 1
        name_vizinhos_tag = 'VIZINHOS_FACE_LV_'

        for i in range(n_levels):
            meshsets_vistos = set()
            meshsets = self.mesh.entities['coarse_volumes_lv' + str(i+1)]
            fine_to_primal_tag = self.mesh.tags['FINE_TO_PRIMAL_CLASSIC_' + str(i+1)]
            primal_id_tag = self.mesh.tags['PRIMAL_ID_'+str(i+1)]
            todos_elementos_vizinhos = dict()

            for m in meshsets:
                elems_in_meshset = self.mesh.mb.get_entities_by_handle(m)
                elems_fora = self.mesh.mtu.get_bridge_adjacencies(elems_in_meshset, 2, 3)
                elems_fora = rng.subtract(elems_fora, elems_in_meshset)
                ids_meshsets_vizinhos = np.unique(self.mesh.mb.tag_get_data(fine_to_primal_tag, elems_fora, flat=True))
                meshsets_vizinhos1 = todos_elementos_vizinhos.setdefault(m, list())
                for j in ids_meshsets_vizinhos:
                    m2 = self.mesh.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag]), np.array([j]))[0]
                    elems_in_m2 = self.mesh.mb.get_entities_by_handle(m2)
                    if set([m2]) & meshsets_vistos:
                        continue
                    meshsets_vizinhos1.append(m2)
                    meshsets_vizinhos2 = todos_elementos_vizinhos.setdefault(m2, list())
                    meshsets_vizinhos2.append(m)

                meshsets_vistos.add(m)

            name = name_vizinhos_tag + str(i+1)
            n = 6
            tipo = 'array'
            entitie = 'coarse_volumes_lv' + str(i+1)
            t1 = types.MB_TYPE_HANDLE
            t2 = types.MB_TAG_SPARSE
            getting_tag(self.mesh.mb, name, n, t1, t2, True, entitie, tipo, self.mesh.tags, self.mesh.tags_to_infos, self.mesh.entities_to_tags)
            tag_vizinhos = self.mesh.tags[name]

            for m, vizinhos in todos_elementos_vizinhos.items():
                n2 = len(vizinhos)
                if n2 < 6:
                    for i in range(6-n2):
                        m3 = self.mesh.mb.create_meshset()
                        vizinhos.append(m3)
                self.mesh.mb.tag_set_data(tag_vizinhos, m, vizinhos)

    def save_dual_mesh(self):

        global flying_dir

        names_tags = list(self.mesh.tags.keys())
        file_names_tags = os.path.join(flying_dir, 'names_tags_out_dual_mesh.txt')
        with open(file_names_tags, 'wb') as handle:
            pickle.dump(names_tags, handle)

        file_name_dual_mesh = os.path.join(flying_dir, 'name_out_dual_mesh.txt')
        with open(file_name_dual_mesh, 'wb') as handle:
            pickle.dump(self.ext_out_h5m, handle)

        file_name_tags_to_infos = os.path.join(flying_dir, 'tags_to_infos_dual_mesh.txt')
        with open(file_name_tags_to_infos, 'wb') as handle:
            pickle.dump(self.mesh.tags_to_infos, handle)

        file_name_entities_to_tags = os.path.join(flying_dir, 'entities_to_tags_dual_mesh.txt')
        with open(file_name_entities_to_tags, 'wb') as handle:
            pickle.dump(self.mesh.entities_to_tags, handle)

        # with open('file.txt', 'rb') as handle:
        #     b = pickle.loads(handle.read())
        file_name_h5m_out = os.path.join(flying_dir, self.ext_out_h5m)
        self.mesh.mb.write_file(file_name_h5m_out)

    def run(self):

        t1 = time.time()
        self.create_tags_dual()
        self.generate_dual_and_primal()
        self.get_boundary_coarse_faces()
        self.get_vizinhos_de_face()
        self.save_dual_mesh()
        t2 = time.time()
        print(f'\ntempo para criar malha dual: {t2 - t1}\n')
