import json
import numpy as np
import pytest
from garnet.kinetics_data import read_regional_kinetics


def fixture():
    return dict(schema_version=1, quantity_kind='raw_delayed_production',
        units={'energy':'MeV','inverse_velocity':'s/cm','production':'1/cm','decay':'1/s'},
        region_ids=['fuel','reflector'], family_ids=['fast','slow'], energy_bounds=[20.,.001,0.],
        inverse_velocities=[[1e-7,2e-5],[2e-7,3e-5]], decay=[.3,.08],
        total_production=[[[.1,.2],[0.,0.]],[[0.,0.],[0.,0.]]],
        delayed_production=[[[[.0002,.0003],[0.,0.]],[[.0005,.001],[0.,0.]]],
                            [[[0.,0.],[0.,0.]],[[0.,0.],[0.,0.]]]],
        provenance={'transport_source':'a'*64}, approximations=[])


def test_explicit_symmetry_mapping_and_compatibility(tmp_path):
    p = tmp_path/'data.json'; p.write_text(json.dumps(fixture()))
    data = read_regional_kinetics(p)
    indices = [1,0,0]
    mapped = data.for_nodes(['reflector','fuel','fuel'], [20.,.001,0.], data.total_production[indices])
    np.testing.assert_array_equal(mapped['inverse_velocities'], data.inverse_velocities[indices])
    np.testing.assert_array_equal(mapped['delayed_production'][0], 0)
    assert not mapped['delayed_production'].flags.writeable
    with pytest.raises(ValueError, match='boundaries'):
        data.for_nodes(['fuel'], [20.,.002,0.], data.total_production[:1])
    with pytest.raises(ValueError, match='region ID'):
        data.for_nodes(['missing'], data.energy_bounds, data.total_production[:1])
    with pytest.raises(ValueError, match='frozen XS'):
        data.for_nodes(['fuel'], data.energy_bounds, data.total_production[:1]*2)


@pytest.mark.parametrize('key,value', [
    ('quantity_kind','global_beta_eff'), ('energy_bounds',[0.,.001,20.]),
    ('family_ids',['same','same']), ('decay',[.1]), ('inverse_velocities',[[0.,1.]]),
    ('provenance',{'source':'not-a-hash'}), ('approximations',None),
    ('total_production',np.zeros((2,2,2)).tolist()), ('units',{}),
])
def test_incompatible_or_missing_input_rejected(tmp_path,key,value):
    data=fixture();data[key]=value
    p=tmp_path/'bad.json';p.write_text(json.dumps(data))
    with pytest.raises(ValueError):read_regional_kinetics(p)


def test_duplicate_json_field_rejected(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match='duplicate'):read_regional_kinetics(p)


def test_general_prompt_delayed_kernel_is_bound_to_imported_node_data(tmp_path):
    from garnet.kinetics_production import assemble_kinetics_production
    native=fixture()
    delayed=np.asarray(native['delayed_production'])
    # Nonzero delayed tail conflicts with the old zero-emission bin.
    delayed[0,:,1,:]=delayed[0,:,0,:]*1e-8
    nuf=np.asarray(native['total_production']).sum(axis=1)
    production=assemble_kinetics_production(nuf,[[1.,0.],[0.,0.]],delayed)
    native['total_production']=production.total.tolist()
    native['delayed_production']=delayed.tolist()
    native['approximations']=['Explicit prompt-spectrum factorization; sampled total kernel replaced.']
    p=tmp_path/'general.json';p.write_text(json.dumps(native))
    data=read_regional_kinetics(p)
    got=data.for_nodes(native['region_ids'],native['energy_bounds'],production.total)
    np.testing.assert_array_equal(got['delayed_production'],production.delayed)
    old=np.zeros_like(production.total);old[:,0,:]=nuf
    with pytest.raises(ValueError,match='frozen XS'):
        data.for_nodes(native['region_ids'],native['energy_bounds'],old)
