"""Independent algebra, state isolation and NEM assembly checks for branches."""
from dataclasses import replace, FrozenInstanceError
import numpy as np
import pytest
from ndgpu.branch_xs import (BranchState, BranchLibrary, NativeBranchState,
                            import_serpent_state, build_nem_solver)
from ndgpu.nem_geometry import CuboidNodalGrid


def state(name, t=600., rho=.7, strength=1., **kw):
    data=dict(state_id=name, conditions={'fuel_temperature':t,'moderator_density':rho,
               'configuration':'ARO','family':'thermal'},region_ids=('fuel','reflector'),
              energy_bounds_mev=[10.,.1,0.],D=np.ones((2,2)),
              sigma_a=np.array([[.02,.04],[.01,.03]])*strength,
              sigma_s=np.array([[[.1,.03],[0.,.2]],[[.2,.02],[0.,.3]]]),
              fission_matrix=np.array([[[.02,.05],[0.,0.]],np.zeros((2,2))])*strength,
              chi=np.array([[1.,0.],[1.,0.]]),total=np.ones((2,2)),
              kappa_fission=np.ones((2,2))*strength,
              factors=np.ones((2,6,2)),metadata={'adf_prescription':'synthetic'},
              qualification='synthetic-only')
    data.update(kw)
    return BranchState(**data)


def line(a,b):
    return BranchLibrary([a,b],axes=('fuel_temperature',),transforms={'fuel_temperature':'sqrt'},simplices=[(a.state_id,b.state_id)])


def test_deep_immutability_and_source_independence():
    source=np.ones((2,2)); metadata={'nested':{'data':np.array([1.,2.])}}
    s=state('a',D=source,metadata=metadata,native=NativeBranchState({'x':{'error':np.array([.01])}}))
    source[:]=5; metadata['nested']['data'][:]=9
    assert np.all(s.D==1)
    assert s.metadata['nested']['data'][0]==1
    for arr in (s.D,s.metadata['nested']['data'],s.native.records['x']['error'],s.nu_sigma_f):
        with pytest.raises(ValueError): arr.setflags(write=True)
    with pytest.raises(TypeError): s.conditions['fuel_temperature']=800
    with pytest.raises(FrozenInstanceError): s.state_id='b'


def test_full_fission_not_product_of_interpolated_marginals_and_zero_channels():
    a=state('a',t=400.)
    F=np.array([[[0.,0.],[.06,.12]],np.zeros((2,2))])
    b=state('b',t=900.,strength=2.,fission_matrix=F,chi=[[0.,1.],[1.,0.]])
    library=line(a,b)
    out=library.evaluate({**dict(a.conditions),'fuel_temperature':625.})
    expected=(a.fission_matrix+b.fission_matrix)/2
    np.testing.assert_allclose(out.fission_matrix,expected)
    np.testing.assert_allclose(out.nu_sigma_f,expected.sum(axis=1))
    assert not np.allclose(expected,out.chi[:,:,None]*out.nu_sigma_f[:,None,:])
    assert out.fission_matrix[0,1,1]>0
    np.testing.assert_allclose(out.removal,out.sigma_a+np.array([[.03,0],[.02,0]]))
    assert out.weights==(.5,.5)
    assert out.qualification=='interpolated-unvalidated'
    for s in (a,b):
        selected=library.evaluate(s.conditions)
        assert not selected.interpolated
        assert selected.contributing_ids==(s.state_id,)
        np.testing.assert_array_equal(selected.fission_matrix,s.fission_matrix)


def test_simplex_affine_exactness_and_convex_domain():
    a=state('a',t=400.,rho=.7,strength=1)
    b=state('b',t=900.,rho=.7,strength=2)
    c=state('c',t=400.,rho=.9,strength=3)
    lib=BranchLibrary([a,b,c],transforms={'fuel_temperature':'sqrt'},simplices=[('a','b','c')])
    conditions={**dict(a.conditions),'fuel_temperature':22.5**2,'moderator_density':.75}
    out=lib.evaluate(conditions)
    np.testing.assert_allclose(out.weights,[.5,.25,.25],atol=1e-15)
    np.testing.assert_allclose(out.sigma_a,a.sigma_a*1.75)
    with pytest.raises(ValueError,match='outside'):
        lib.evaluate({**conditions,'fuel_temperature':900.,'moderator_density':.9})
    with pytest.raises(ValueError,match='outside'):
        lib.evaluate({**conditions,'configuration':'ARI'})
    with pytest.raises(ValueError,match='outside'):
        lib.evaluate({**conditions,'family':'rod'})


def test_log_adf_and_exterior_unity():
    mask=np.ones((2,6),dtype=bool);mask[0,2]=False;mask[1,0]=False
    fa=np.ones((2,6,2));fb=fa.copy();fa[~mask]=.5;fb[~mask]=2.
    a=state('a',t=400.,factors=fa,exterior_mask=mask)
    b=state('b',t=900.,factors=fb,exterior_mask=mask)
    out=line(a,b).evaluate({**dict(a.conditions),'fuel_temperature':625.})
    np.testing.assert_allclose(out.factors,1.)
    fb[0,0]=2
    with pytest.raises(ValueError,match='exterior'):state('bad',factors=fb,exterior_mask=mask)


@pytest.mark.parametrize('change',[
    {'region_ids':('r','f')},{'energy_bounds_mev':[20.,.1,0.]},
    {'metadata':{'adf_prescription':'different'}},{'kappa_fission':None},
    {'conditions':{'fuel_temperature':900.,'moderator_density':.7,'configuration':'ARI','family':'thermal'}}])
def test_simplex_rejects_incompatible_states(change):
    a=state('a',t=400.);b=state('b',t=900.,**change) if 'conditions' not in change else state('b',**change)
    with pytest.raises(ValueError):line(a,b)


@pytest.mark.parametrize('change',[
    {'D':np.zeros((2,2))},{'sigma_s':np.ones((2,2))},
    {'sigma_a':np.full((2,2),np.nan)},{'fission_matrix':-np.ones((2,2,2))},
    {'metadata':{'energy_unit':'eV'}},{'metadata':{'scattering_order':'emitted-source'}},
    {'chi':np.zeros((2,2))},{'energy_bounds_mev':[1.,2.,1.]},
])
def test_bad_state_rejected(change):
    with pytest.raises(ValueError):state('bad',**change)


def test_missing_and_negative_axis_and_degenerate_simplex():
    a=state('a',t=400.);b=state('b',t=400.)
    with pytest.raises(ValueError,match='degenerate'):line(a,b)
    lib=BranchLibrary([a],axes=('fuel_temperature',))
    with pytest.raises(ValueError):lib.evaluate({'fuel_temperature':-1})
    with pytest.raises(ValueError):lib.evaluate({})


def test_safe_json_round_trip(tmp_path):
    native=NativeBranchState({'r':{'fields':{'INF_NSF':[.1,.01]},'sha256':'abc'}})
    a=state('a',t=400.,native=native);b=state('b',t=900.)
    lib=line(a,b);path=tmp_path/'branch.json';lib.save(path)
    restored=BranchLibrary.load(path)
    np.testing.assert_array_equal(restored.select('a').D,a.D)
    assert restored.select('a').native.records['r']['fields']['INF_NSF']==(.1,.01)
    cond={**dict(a.conditions),'fuel_temperature':625.}
    np.testing.assert_array_equal(restored.evaluate(cond).fission_matrix,lib.evaluate(cond).fission_matrix)


def test_factory_fresh_operator_full_matrix_and_dense_reference():
    # Reflective uniform state has an independent dense multigroup eigenpair.
    grid=CuboidNodalGrid([[0,10,0,10,0,10],[10,20,0,10,0,10]],'reflective')
    F=np.array([[.03,.05],[.01,.02]])
    ss=np.array([[.1,.03],[.005,.2]])
    a=state('a',sigma_a=np.tile([.02,.04],(2,1)),sigma_s=np.tile(ss,(2,1,1)),
            fission_matrix=np.tile(F,(2,1,1)),chi=np.tile([.75,.25],(2,1)))
    solver=build_nem_solver(a,grid,degree=4)
    other=build_nem_solver(a,grid,degree=4)
    assert solver is not other and solver._lu is not other._lu
    np.testing.assert_array_equal(solver.fission,a.fission_matrix)
    R=np.array([[.02+.03,-.005],[-.03,.04+.005]])
    eigenvalues=np.linalg.eigvals(np.linalg.solve(R,F))
    result=solver.solve_eigenvalue()
    assert result.k_eff==pytest.approx(max(eigenvalues),rel=1e-8)
    changed=build_nem_solver(replace(a,D=a.D*2),grid,degree=4)
    assert changed is not solver
    with pytest.raises(ValueError):build_nem_solver(a,grid,region_indices=[0.,1.])


def test_serpent_import_preserves_errors_hash_and_explicit_diffusion(tmp_path):
    from ndgpu.serpent_xs import SerpentXSRecord
    fields={'VERSION':'Serpent 2.2.0','GC_UNIVERSE_NAME':'fuel',
      'MACRO_NG':np.array([2]),'MACRO_E':np.array([20.,6.25e-7,0.]),
      'INF_RABSXS':np.array([.01,.1,.1,.02]),'INF_ABS':np.array([.01,.1,.1,.02]),
      'INF_SP0':np.array([.44,.01,.05,.02,.001,.03,.899,.04]),
      'INF_S0':np.array([.44,.01,.05,.02,.001,.03,.899,.04]),
      'INF_NSF':np.array([.005,.02,.15,.03]),'INF_CHIT':np.array([.95,.01,.05,.02]),
      'INF_TOT':np.array([.5,.01,1.,.02]),'CMM_DIFFCOEF':np.array([1.25,.01,.25,.02]),
      'INF_DIFFCOEF':np.array([1.5,.01,.3,.02]),'INF_FLX':np.array([3.,.01,1.,.02]),
      'IMP_KEFF':np.array([1.,.001])}
    record=SerpentXSRecord(tmp_path/'raw_res.m','abc123',7,fields)
    s=import_serpent_state('nominal',{'fuel_temperature':600.},{'fuel':record})
    np.testing.assert_array_equal(s.D,[[1.5,.3]])
    assert s.native.records['fuel']['record_index']==7
    assert s.native.records['fuel']['source_sha256']=='abc123'
    np.testing.assert_array_equal(s.native.records['fuel']['fields']['INF_NSF'],fields['INF_NSF'])
    fields['INF_NSF'][:]=0
    assert s.native.records['fuel']['fields']['INF_NSF'][1]==.02
    assert s.metadata['adf_origin']=='unity' and s.qualification=='unvalidated'
    with pytest.raises(ValueError):import_serpent_state('bad',{}, {'fuel':record},diffusion='cmm')
    with pytest.raises(ValueError):import_serpent_state('bad',{}, {'fuel':record},metadata={'length_unit':'m'})


def test_metadata_compatibility_nuclear_geometry_solver_identities():
    a=state('a',t=400.)
    for key in ('nuclear_data_family','geometry_family','scoring_identity',
                'thermal_scattering_identity','reference_orders','solver_family','library_identity'):
        b=state('b',t=900.,metadata={**dict(a.metadata),key:'different'})
        with pytest.raises(ValueError,match='incompatible'):line(a,b)


def test_temperature_helper_benchmark_path_and_off_path_rejection():
    from ndgpu.branch_xs import BenchmarkModeratorPath
    path=BenchmarkModeratorPath()
    assert path.evaluate(573.15)==.72683
    assert path.evaluate(558.15)==.75572
    assert path.evaluate(588.15)==.69325
    intermediate=path.evaluate(578.15)
    assert .69325<intermediate<.72683
    a=state('a',t=400.,rho=path.evaluate(573.15))
    b=state('b',t=900.,rho=path.evaluate(573.15))
    c=state('c',t=400.,rho=path.evaluate(588.15))
    lib=BranchLibrary([a,b,c],simplices=[('a','b','c')])
    out=lib.evaluate_temperatures(450.,578.15,path,configuration='ARO',family='thermal')
    assert out.conditions['moderator_density']==intermediate
    assert out.metadata['evaluated_moderator_temperature']==578.15
    assert 'moderator_temperature' not in out.conditions
    for temp in (550.,600.,float('nan')):
        with pytest.raises(ValueError):path.evaluate(temp)
    with pytest.raises(ValueError):lib.evaluate_temperatures(450.,578.15,path,conditions={'moderator_density':.8})
    with pytest.raises(ValueError):BenchmarkModeratorPath(pressure_bar=150)
    with pytest.raises(ValueError):BenchmarkModeratorPath(densities=(.7,.8,.9))


def test_factory_does_not_hide_external_factor_error():
    grid=CuboidNodalGrid([[0,10,0,10,0,10],[10,20,0,10,0,10]],'reflective')
    s=state('bad',factors=np.full((2,6,2),1.1))
    with pytest.raises(ValueError,match='external'):build_nem_solver(s,grid,degree=4)


def test_ambiguous_exact_conditions_require_explicit_state_id():
    a=state('nominal');b=state('fuel-base')
    lib=BranchLibrary([a,b],axes=('fuel_temperature',))
    with pytest.raises(ValueError,match='ambiguous exact conditions'):
        lib.evaluate(a.conditions)
    assert lib.select('nominal').contributing_ids==('nominal',)
    assert lib.select('fuel-base').contributing_ids==('fuel-base',)


def test_evaluated_states_cannot_be_reused_as_original_table_states():
    a=state('a',t=400.);b=state('b',t=900.)
    lib=line(a,b)
    exact=lib.select('a')
    interpolated=lib.evaluate({**dict(a.conditions),'fuel_temperature':625.})
    for derived in (exact,interpolated):
        with pytest.raises(ValueError,match='evaluated states cannot be table inputs'):
            BranchLibrary([derived],axes=('fuel_temperature',))


def test_geometry_bounds_identity_mismatch_rejected():
    a=state('a',t=400.,metadata={'geometry_bounds_sha256':'abc'})
    b=state('b',t=900.,metadata={'geometry_bounds_sha256':'def'})
    with pytest.raises(ValueError,match='incompatible'):
        line(a,b)


def test_changed_density_knots_cannot_reuse_reserved_benchmark_identity():
    from ndgpu.branch_xs import BenchmarkModeratorPath
    with pytest.raises(ValueError,match='new path_id'):
        BenchmarkModeratorPath(densities=(.76,.73,.69))
    custom=BenchmarkModeratorPath(densities=(.76,.73,.69),path_id='explicit-other-path')
    assert custom.evaluate(573.15)==pytest.approx(.73)
