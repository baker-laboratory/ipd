import numpy as np
import pickle
import gzip
import ipd

def extract_helen_example(symdata: 'str|dict'):
    if isinstance(symdata, str):
        symdata = ipd.cast(dict, load_symdata_from_pickle(symdata))
    data = ipd.bunchify(symdata.data)

    # grabbing AtomArrays and also the information regarding the query pn_unit_iid
    atom_array = data['atom_array'][
        data['atom_array'].occupancy ==
        1]  # only doing this for now    since NaN values are a problem for SymBody class

    # getting information for q_pn_unit and its atom array
    q_unit_info = {'q_pn_unit_iid': data['query_pn_unit_iids'][0]}
    q_pn_unit_atom_array = atom_array[atom_array.pn_unit_iid == q_unit_info['q_pn_unit_iid']]
    q_unit_info['q_entity'] = np.unique(q_pn_unit_atom_array.pn_unit_entity)[0]
    q_unit_info['q_pn_unit_id'] = np.unique(q_pn_unit_atom_array.pn_unit_id)[0]

    # centering complex at q_pn_unit COM (so that xforms are computed wrt q_pn_unit at origin)
    q_pn_unit_COM = q_pn_unit_atom_array.coord.mean(axis=0)
    atom_array.coord -= q_pn_unit_COM

    # grabbing atom array for all "like" subunits, in this case via entity type
    atom_array_same_entity = atom_array[atom_array.pn_unit_entity == q_unit_info['q_entity']]
    atom_array_same_entity = ipd.dev.set_metadata(atom_array_same_entity,
                                                  fname=data['path'],
                                                  pdbcode=data['pdb_id'])

    # construct subunit atom_array list to find frames
    atom_array_list = [
        atom_array_same_entity[atom_array_same_entity.pn_unit_iid == q]
        for q in np.unique(atom_array_same_entity.pn_unit_iid)
    ]

    ipd.atom.info(q_pn_unit_atom_array)
    ipd.atom.info(atom_array_list)
    # find frames
    # pn_unit_body = ipd.atom.Body(q_pn_unit_atom_array)
    comp = ipd.atom.find_components_by_seqaln_rmsfit(atom_array_list, seqmatch=.3)
    comp.print_intermediates()
    ipd.atom.merge_small_components(comp)

    # make SymBody
    symbody = ipd.atom.SymBody(atom_array_list[0], comp.frames[0])
    print(symbody)
    for i in range(len(symbody)):
        print(symbody.subunit_contacts(i))
    contactmat = symbody.subunit_contact_matrix(subunit=0)
    topk = contactmat.topk_fragment_contact_by_subset_summary(fragsize=21, k=13, stride=7)
    assert topk.index.keys() == topk.vals.keys()
    for subs, idx in topk.index.items():
        print(subs, idx[:,:4], topk.vals[subs][:4])
    # ipd.showme(symbody)
    # assert 0

def load_symdata_from_pickle(fname):
    with gzip.open(fname, 'rb') as f:
        return pickle.load(f, encoding='bytes')
