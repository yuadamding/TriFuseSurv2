"""Meaningful synthetic checks for phenotype arithmetic, support and agreement."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk

spec = importlib.util.spec_from_file_location('phenotype_measure', Path(__file__).with_name('measure.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def image(array, spacing=(.5,.5,1.), origin=(3.,7.,11.), direction=None):
    result = sitk.GetImageFromArray(array)
    result.SetSpacing(spacing); result.SetOrigin(origin)
    if direction is not None: result.SetDirection(direction)
    return result


class Measurements(unittest.TestCase):
    def test_quantile_independent_linear_zero_IQR_and_volume_units(self):
        mask = np.ones((5,5,4), dtype=bool)
        values = np.full(mask.shape, .25, dtype=np.float32)
        result = m.region_values(values, mask, [.5,.5,1.], 100)
        self.assertEqual(result['n'],100)
        self.assertAlmostEqual(result['V_cm3'],.025)
        self.assertTrue(result['support_valid'])
        self.assertEqual(result['H'],0)
        self.assertEqual(result['iqr_independent_delta'],0)
        rng=np.random.default_rng(72)
        for n in [1,2,3,17,100,1001]:
            values=rng.normal(size=n).astype(np.float32)
            for probability in [.25,.5,.75]:
                self.assertAlmostEqual(m.independent_linear_quantile(values, probability),
                    np.quantile(values.astype(np.float64),probability,method='linear'),places=12)

    def test_morphology_uses_spacing_and_detects_expansion_outside_source(self):
        mask=np.zeros((21,21,21),dtype=bool);mask[10,10,10]=True
        dilated, info=m.physical_morphology(mask,[.5,.5,1.],1.,'dilation')
        self.assertEqual(int(dilated.sum()),15)
        self.assertTrue(dilated[10,10,12])
        self.assertTrue(dilated[11,10,10])
        self.assertFalse(dilated[11,10,11])
        self.assertEqual(info['outside_delivered_voxels'],0)
        mask[:]=False;mask[10,10,0]=True
        _,info=m.physical_morphology(mask,[.5,.5,1.],1.,'dilation')
        self.assertGreater(info['outside_delivered_voxels'],0)
        empty=np.zeros(mask.shape,dtype=bool)
        grown,info=m.physical_morphology(empty,[.5,.5,1.],1.,'dilation')
        self.assertFalse(grown.any())
        self.assertEqual(info['outside_delivered_voxels'],0)

    def test_coarsening_preserves_outer_edges_with_oblique_direction(self):
        theta=.2;c=np.cos(theta);s=np.sin(theta)
        direction=(-1.,0.,0.,0.,-c,-s,0.,-s,c)
        ct=image(np.zeros((16,12,8),dtype=np.float32),direction=direction)
        mask=image(np.zeros((16,12,8),dtype=np.uint8),direction=direction)
        coarse, _,_=m.coarsen_images(ct,mask,mask,1.)
        self.assertEqual(coarse.GetSize(),(4,6,16))
        matrix=np.asarray(direction).reshape(3,3)
        def corners(value):
            origin=np.asarray(value.GetOrigin());sp=np.asarray(value.GetSpacing());size=np.asarray(value.GetSize())
            return origin-matrix@(sp/2),origin+matrix@((size-.5)*sp)
        for before,after in zip(corners(ct),corners(coarse)):
            np.testing.assert_allclose(before,after,rtol=0,atol=1e-12)

    def test_absolute_agreement_ICC_penalizes_systematic_offset(self):
        values=np.arange(1.,5.)
        self.assertAlmostEqual(m.icc_absolute_single(values,values),1.)
        self.assertAlmostEqual(m.icc_absolute_single(values,values+1),10/13)
        self.assertLess(m.icc_absolute_single(values,values+100),.001)
        self.assertTrue(np.isnan(m.icc_absolute_single(np.ones(4),np.ones(4))))

    def test_source_binding_errors_are_fatal_not_missing_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(m.SourceBindingError):
                m.extraction_case(('case',temporary,{}, {'minimum_voxels':100}))

    def test_small_perturbed_support_and_overlap_remain_unavailable(self):
        ct_array=np.linspace(0,1,40**3,dtype=np.float32).reshape(40,40,40)
        p=np.zeros(ct_array.shape,dtype=bool);n=p.copy()
        p[10:16,10:16,10:16]=True;n[10:16,10:16,17:23]=True
        ct=image(ct_array);pi=image(p.astype(np.uint8));ni=image(n.astype(np.uint8))
        loaded=(ct,pi,ni,ct_array,p,n,{}, {'ct.nii.gz':'a'*64})
        settings={'minimum_voxels':100,'morphology_mm':1.,'primary_conditions':['erosion_1mm','dilation_1mm'],'stress_conditions':[]}
        with patch.object(m,'load_case',return_value=loaded):
            rows=m.perturbation_case(('case','unused',{},settings,False))
        self.assertFalse(rows[0]['available'])
        self.assertIn('fewer_than_100_voxels',rows[0]['reason'])
        self.assertFalse(rows[1]['available'])
        self.assertIn('PT_LN_overlap',rows[1]['reason'])

    def test_fallback_cannot_activate_after_real_association_started(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'qualification').mkdir();(root/'statistics').mkdir()
            m.exclusive_json(root/'qualification/paired_decision.json',{'protocol_sha256':'b'*64,'passed':False})
            self.assertEqual(m.fallback_allowed(root,'b'*64)['kind'],'paired_measurement_failed')
            (root/'statistics/PRIMARY_ASSOCIATION_STARTED.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'after an association'):
                m.fallback_allowed(root,'b'*64)

    def test_exact_PyRadiomics_sphericity_in_isolated_overlay(self):
        # This tests the actual mesh implementation, not a voxel-face surrogate.
        try: import radiomics
        except ImportError: self.skipTest('Pinned isolated PyRadiomics overlay not present')
        ct=image(np.zeros((20,20,20),dtype=np.float32))
        mask=np.zeros((20,20,20),dtype=np.uint8);mask[5:15,5:15,5:15]=1
        value=m.sphericity(ct,image(mask))
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value,0)
        self.assertLessEqual(value,1)

    def test_actual_synthetic_producer_parquet_validation_and_failed_gate(self):
        try: import pyarrow.parquet as pq
        except ImportError: self.skipTest('Isolated Parquet overlay absent')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'images';source.mkdir()
            (root/'preparation').mkdir();records=[];inventory=[]
            for index in range(6):
                pid='fixture-'+str(index);directory=source/pid;directory.mkdir()
                rng=np.random.default_rng(index+21)
                data=rng.uniform(size=(32,32,32)).astype(np.float32)
                p=np.zeros(data.shape,dtype=np.uint8);n=p.copy()
                # 512 voxels, but 1-mm physical erosion leaves only 96.
                p[3:11,3:11,3:11]=1;n[17:25,17:25,17:25]=1
                for name,array in [('ct.nii.gz',data),('mask_primary.nii.gz',p),('mask_nodal.nii.gz',n)]:
                    sitk.WriteImage(image(array),str(directory/name))
                (directory/'meta.json').write_text('{"ct_spacing_xyz":".5|.5|1"}')
                for path in directory.iterdir():
                    inventory.append({'patient_id':pid,'file':path.name,'bytes':path.stat().st_size,'sha256':m.sha256(path)})
                records.append({'patient_id':pid,'split':'train','paired_measurement_candidate':True,
                    'primary_only_measurement_candidate':True,'paired_eligible':True,'primary_only_eligible':True})
            pd.DataFrame(inventory).to_csv(root/'inventory.csv',index=False)
            pd.DataFrame(records).to_csv(root/'preparation/cohort_manifest_pre_measurement.csv',index=False)
            pd.DataFrame(records).to_csv(root/'preparation/cohort_manifest.csv',index=False)
            settings={'minimum_voxels':100,'iqr_validation_tolerance':1e-6,'morphology_mm':1.,
                'primary_conditions':['erosion_1mm','dilation_1mm','isotropic_1mm'],
                'stress_conditions':['isotropic_2mm'],'bootstrap_resamples':1000,'bootstrap_seed':42,
                'fallback_bootstrap_seed':43,'zero_sd_tolerance':1e-12,'icc_min':.9,'icc_lower95_min':.75,
                'coverage_min':.9,'minimum_bootstrap_valid_fraction':.95,'median_change_sd_max':.2,
                'fallback_definition':'PyRadiomics_3_0_1_original_shape_Sphericity_mesh_based'}
            protocol={'paths':{'imaging_root':str(source),'image_source_inventory_csv':str(root/'inventory.csv')},
                      'measurement':settings}
            m.run_extraction(protocol,'d'*64,root,1)
            features=pd.read_csv(root/'measurements/phenotype_features.csv')
            validation=pd.read_csv(root/'measurements/measurement_validation.csv')
            self.assertEqual(len(features),6)
            self.assertTrue(validation.paired_measurement_valid.all())
            self.assertTrue((validation.PT_independent_IQR_delta<=1e-6).all())
            self.assertEqual(pq.read_table(root/'measurements/phenotype_features.parquet').num_rows,6)
            decision=m.run_qualification(protocol,'d'*64,root,1)
            self.assertFalse(decision['passed'])
            import json
            details=json.loads((root/'qualification/paired_decision.json').read_text())
            self.assertEqual(details['baseline_train_n'],6)
            self.assertEqual(details['conditions'][0]['coverage'],0)
            self.assertFalse(details['conditions'][0]['passed'])
            self.assertEqual(details['endpoint_columns_read'],[])
            # Actual fallback producer binds its phiP file AND its independent
            # voxel-volume/supporting measurement file to match the R consumer.
            m.run_fallback_extraction(protocol,'d'*64,root)
            m.run_qualification(protocol,'d'*64,root,1,fallback=True)
            shape_decision=json.loads((root/'qualification/primary_only_decision.json').read_text())
            self.assertEqual(shape_decision['baseline_feature_file_sha256'],
                             m.sha256(root/'measurements/primary_only_features.csv'))
            self.assertEqual(shape_decision['supporting_measurement_file_sha256'],
                             m.sha256(root/'measurements/phenotype_features.csv'))
            with self.assertRaises(FileExistsError): m.run_extraction(protocol,'d'*64,root,1)


if __name__=='__main__':unittest.main()
