from .generate_dual_mesh import get_box
from .generate_mesh0 import getting_tag
import numpy as np

class SetB:

    def __init__(self, mesh):
        self.mesh = mesh
        self.wells = dict()
        self.bvn = list()
        self.bvd = list()

    def create_tags(self):

        l = ['P', 'Q']
        for name in l:
            n = 1
            tipo = 'double'
            entitie = 'volumes'
            t1 = types.MB_TYPE_DOUBLE
            t2 = types.MB_TAG_SPARSE
            self.getting_tag(self.mesh.mb, name, n, t1, t2, True, entitie, tipo, self.mesh.tags, self.mesh.tags_to_infos, self.mesh.entities_to_tags)

    def set_wells(self):

        for name in self.mesh.data_loaded['wells']:
            # presc = well['presc']
            # tipo = well['type']
            # val = well['val']
            well = self.mesh.data_loaded['wells'][name]
            tipo_region = well['region']['type']
            # lim = well['region']['lim']
            if tipo_region == 'box':
                self.set_wells_box(well, name)
            else:
                raise NameError(f'\ntipo errado de Regiao: {tipo_region}\n')


            #     self.set_wells_box(well)
            # else:
            #     raise NameError(f'\ntipo errado de Regiao: {tipo_region}\n')

    def set_wells_box(self, well, name):
        '''
        setar volumes com vazao e pressao prescrita se o tipo de região for 'box'
        '''
        # volumes com vazao prescrita: bvn
        bv = list()
        # volumes com vazao prescrita: bvd

        presc = well['presc']
        tipo = well['type']
        val = well['val']
        tipo_region = well['region']['type']
        lim = well['region']['lim']

        limites = np.array([np.array(lim[0]), np.array(lim[1])])
        volumes = get_box(self.mesh.entities['volumes'], self.mesh.datas['CENT_VOLS'], limites, False)







        import pdb; pdb.set_trace()





    def run(self):

        self.create_tags()
        self.set_wells()

        self.mesh.contour = self
