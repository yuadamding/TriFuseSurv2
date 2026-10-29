"""Synthetic routing provenance through the trainer, checkpoints and exports."""
import argparse
import ast
import contextlib
import copy
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from trifusesurv2.models.routing_intervention import (
    ROUTING_INTERVENTION_ID, ROUTING_DESIGN_SHA256, ROUTING_EXPORT_LABEL,
    routing_intervention_metadata,
)
from trifusesurv2.multimodal_survival import train
from trifusesurv2.utils.experiment import canonical_experiment_spec


class RoutingInterventionContractTests(unittest.TestCase):
    def parse(self, *options):
        with patch('sys.argv', ['train', '--meta_csv', 'unused.csv', *options]):
            return train.parse_args()

    def active(self):
        return self.parse('--routing_intervention', ROUTING_INTERVENTION_ID)

    def model(self, active=False):
        model = torch.nn.Linear(1, 1)
        if active:
            model.routing_intervention = ROUTING_INTERVENTION_ID
        return model

    def contract(self, args):
        return {'fusion_mode': args.fusion_mode, 'experiment_spec': canonical_experiment_spec(args),
                **train._routing_intervention_contract(args)}

    def actual_aux(self, q=None, r=None):
        from trifusesurv2.models.habitat_survival import HabitatAlignedSurvivalModel
        q = torch.ones(1, 4, dtype=torch.bool) if q is None else q
        r = torch.ones_like(q) if r is None else r
        model = HabitatAlignedSurvivalModel(image_token_dim=8, radiomics_token_dim=5,
            clinical_token_dim=4, num_time_bins=2, model_dim=16, num_heads=4, transformer_layers=1,
            dropout=0., routing_intervention=ROUTING_INTERVENTION_ID).eval()
        image_presence = torch.ones(len(q), 6, dtype=torch.bool)
        image_presence[:, 1:5] = q
        with torch.no_grad():
            _, aux = model(image_tokens=torch.zeros(len(q), 6, 8), image_presence=image_presence,
                radiomics_tokens=torch.zeros(len(q), 4, 5), radiomics_presence=r,
                clinical_tokens=torch.zeros(len(q), 3, 4), return_aux=True)
        return aux

    def checkpoint(self, path, model, args, contract):
        train.save_checkpoint(path, 1, 2, model, args, None, None, None, 1., None, None,
                              include_training_state=False, encoders={'model_contract': contract})

    def test_default_parse_and_canonical_contract_preserve_old_shape(self):
        args = self.parse()
        self.assertIsNone(args.routing_intervention)
        old = copy.deepcopy(args)
        del old.routing_intervention
        self.assertEqual(canonical_experiment_spec(args), canonical_experiment_spec(old))
        self.assertEqual(train._routing_intervention_contract(args), {})
        self.assertEqual(train._routing_model_kwargs(args), {})
        for key in train.ROUTING_METADATA_KEYS:
            self.assertNotIn(key, canonical_experiment_spec(args)['settings'])

    def test_explicit_cli_binds_exact_versioned_design(self):
        args = self.active()
        expected = routing_intervention_metadata(ROUTING_INTERVENTION_ID)
        self.assertEqual({key: vars(args)[key] for key in expected}, expected)
        self.assertEqual(train._routing_intervention_contract(args), expected)
        self.assertEqual(train._routing_model_kwargs(args), {'routing_intervention': ROUTING_INTERVENTION_ID})
        self.assertEqual(args.fusion_mode, 'F3')
        self.assertEqual(args.routing_intervention_design_sha256, ROUTING_DESIGN_SHA256)
        self.assertEqual(args.routing_intervention_export_label, ROUTING_EXPORT_LABEL)

    def test_invalid_intervention_config_is_rejected_before_seed_or_device(self):
        for options in (('--fusion_mode', 'F2'), ('--fusion_mode', 'F4'),
                        ('--model_version', 'v1'), ('--region_mode', 'legacy'),
                        ('--comparison_recipe', 'robustness')):
            with self.subTest(options=options), patch('sys.argv',
                    ['train', '--meta_csv', 'unused.csv', '--routing_intervention', ROUTING_INTERVENTION_ID, *options]), \
                    patch.object(train, 'set_seed') as seed, patch.object(train, 'parse_device') as device:
                with self.assertRaises(ValueError):
                    train.main()
                seed.assert_not_called()
                device.assert_not_called()
        for value in ('unknown', 'F3', ''):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.parse('--routing_intervention', value)

    def test_explicit_derived_fields_cannot_be_changed_or_carried_when_inactive(self):
        for key in train.ROUTING_METADATA_KEYS[1:]:
            args = self.active()
            setattr(args, key, 'changed')
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'fixed design'):
                train._routing_intervention_contract(args)
            args.routing_intervention = None
            with self.assertRaisesRegex(ValueError, 'Inactive'):
                train._routing_intervention_contract(args)

    def test_real_trainer_model_construction_passes_only_explicit_intervention(self):
        # Execute the actual constructor call expression rather than a helper-only mock.
        tree = ast.parse(inspect.getsource(train.run_one_fold))
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == 'ContourAwareHabitatSurvivalModel')
        compiled = compile(ast.fix_missing_locations(ast.Expression(call)), '<actual trainer model construction>', 'eval')
        for args in (self.parse(), self.active()):
            seen = {}
            def factory(**kwargs):
                seen.update(kwargs)
                return object()
            scope = dict(vars(train), args=args, ContourAwareHabitatSurvivalModel=factory,
                         v2_backbone=object(), fold_num_time_bins=2, rad_dim=4, clin_dim=5)
            eval(compiled, scope)
            self.assertEqual(seen['fusion_mode'], 'F3')
            if args.routing_intervention is None:
                self.assertNotIn('routing_intervention', seen)
            else:
                self.assertEqual(seen['routing_intervention'], ROUTING_INTERVENTION_ID)

    def test_resume_rejects_changed_intervention_design_and_export_identity(self):
        old = self.contract(self.parse())
        active = self.contract(self.active())
        with self.assertRaisesRegex(RuntimeError, 'routing_intervention'):
            train._validate_resume_model_contract(old, active)
        for key in train.ROUTING_METADATA_KEYS:
            changed = copy.deepcopy(active)
            changed[key] = 'changed'
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                train._validate_resume_model_contract(active, changed)

    def test_checkpoint_serialization_omits_inactive_and_retains_explicit_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            for active in (False, True):
                args = self.active() if active else self.parse()
                path = Path(temporary) / (str(active) + '.pt')
                self.checkpoint(path, self.model(active), args, self.contract(args))
                saved = torch.load(path, weights_only=False, map_location='cpu')
                expected = routing_intervention_metadata(args.routing_intervention)
                self.assertEqual(train._checkpoint_routing_metadata(saved), expected)
                for location in (saved['args'], saved['encoders']['model_contract'],
                                 saved['encoders']['model_contract']['experiment_spec']['settings']):
                    self.assertEqual({key: location[key] for key in train.ROUTING_METADATA_KEYS if key in location}, expected)

    def test_save_fails_closed_before_writing_mislabeled_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'bad.pt'
            for model_active, args_active in ((True, False), (False, True)):
                args = self.active() if args_active else self.parse()
                with self.subTest(model=model_active, args=args_active), self.assertRaisesRegex(ValueError, 'model and arguments'):
                    self.checkpoint(path, self.model(model_active), args, self.contract(args))
                self.assertFalse(path.exists())
            args = self.active()
            with self.assertRaisesRegex(ValueError, 'embedded contract'):
                self.checkpoint(path, self.model(True), args, {})
            self.assertFalse(path.exists())

    def test_selected_checkpoint_restore_rejects_same_shape_other_routing(self):
        with tempfile.TemporaryDirectory() as temporary:
            for saved_active in (False, True):
                args = self.active() if saved_active else self.parse()
                path = Path(temporary) / (str(saved_active) + '.pt')
                self.checkpoint(path, self.model(saved_active), args, self.contract(args))
                train.restore_selected_curve_checkpoint(self.model(saved_active), path)
                with self.assertRaisesRegex(ValueError, 'checkpoint and model'):
                    train.restore_selected_curve_checkpoint(self.model(not saved_active), path)
                if saved_active:
                    saved = torch.load(path, weights_only=False)
                    saved['args']['routing_intervention_design_sha256'] = '0' * 64
                    bad = Path(temporary) / 'changed.pt'
                    torch.save(saved, bad)
                    with self.assertRaisesRegex(ValueError, 'checkpoint arguments'):
                        train.restore_selected_curve_checkpoint(self.model(True), bad)

    def test_curve_exports_use_distinct_label_and_reject_mislabeled_contract(self):
        payload = {'x': torch.zeros(1, 1), 't': torch.tensor([5.]), 'e': torch.tensor([1.]), 'pid': ['001']}
        outcomes = pd.DataFrame({'Patient_ID': ['001'], 'OS.TIME': [5.], 'OS.EVENT': [1]})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for active in (False, True):
                args = self.active() if active else self.parse()
                args.risk_horizon_days = 15.
                args.time_bin_width_days = 10.
                model = self.model(active)
                checkpoint, contract, output = root / (str(active) + '.pt'), root / (str(active) + '.json'), root / (str(active) + '.csv')
                self.checkpoint(checkpoint, model, args, self.contract(args))
                contract.write_text(json.dumps(self.contract(args)))
                prediction = {'OS': torch.zeros(1, 2)}
                if active:
                    prediction = (prediction, self.actual_aux())
                with patch.object(train, '_model_forward_eval', return_value=prediction):
                    receipt = train.export_survival_curves(model, [payload], torch.device('cpu'), outcome_frame=outcomes,
                        args=args, fold=0, split='val', checkpoint_path=checkpoint, contract_path=contract, weights='best',
                        autocast_ctx=contextlib.nullcontext, teacher_force_alpha=0., output_path=output)
                self.assertEqual(pd.read_csv(output)['mode'].unique().tolist(), [ROUTING_EXPORT_LABEL if active else 'F3'])
                expected = routing_intervention_metadata(args.routing_intervention)
                self.assertEqual({key: receipt['metadata'][key] for key in expected}, expected)
                if not active:
                    self.assertTrue(set(train.ROUTING_METADATA_KEYS).isdisjoint(receipt['metadata']))
                else:
                    broken = self.contract(args)
                    broken['routing_intervention_design_sha256'] = '0' * 64
                    contract.write_text(json.dumps(broken))
                    with self.assertRaisesRegex(ValueError, 'export model contract'):
                        train.export_survival_curves(model, [], torch.device('cpu'), outcome_frame=outcomes, args=args,
                            fold=0, split='val', checkpoint_path=checkpoint, contract_path=contract, weights='best',
                            autocast_ctx=contextlib.nullcontext, teacher_force_alpha=0., output_path=root/'must_not_exist.csv')
                    self.assertFalse((root/'must_not_exist.csv').exists())

    def test_active_exports_reject_same_shape_control_checkpoint_before_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = self.active(); model = self.model(True)
            source, contract = root/'control.pt', root/'intervention.json'
            control_args = self.parse()
            self.checkpoint(source, self.model(False), control_args, self.contract(control_args))
            contract.write_text(json.dumps(self.contract(args)))
            with patch.object(train, '_model_forward_eval') as forward:
                with self.assertRaisesRegex(ValueError, 'checkpoint and export model'):
                    train.export_survival_curves(model, [], torch.device('cpu'), outcome_frame=None, args=args,
                        fold=0, split='val', checkpoint_path=source, contract_path=contract, weights='best',
                        autocast_ctx=contextlib.nullcontext, teacher_force_alpha=0., output_path=root/'bad.csv')
                forward.assert_not_called()
            self.assertFalse((root/'bad.csv').exists())
            with self.assertRaisesRegex(ValueError, 'source checkpoint and inference model'):
                train.persist_selected_inference_weights(model, root/'bad.pt', source_checkpoint=source,
                                                          contract_path=contract, weights='ema')
            self.assertFalse((root/'bad.pt').exists())

    def test_embedded_base_mode_or_settings_cannot_disagree_with_intervention(self):
        args = self.active()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'source.pt'
            self.checkpoint(source, self.model(True), args, self.contract(args))
            checkpoint = torch.load(source, weights_only=False)
            for branch in ('arguments', 'contract', 'settings'):
                altered = copy.deepcopy(checkpoint)
                if branch == 'arguments':
                    altered['args']['fusion_mode'] = 'F2'
                elif branch == 'contract':
                    altered['encoders']['model_contract']['fusion_mode'] = 'F2'
                else:
                    altered['encoders']['model_contract']['experiment_spec']['settings']['routing_intervention_design_sha256'] = '0'*64
                with self.subTest(branch=branch), self.assertRaises(ValueError):
                    train._checkpoint_routing_metadata(altered)

    def test_actual_mixed_mask_export_coverage_is_complete_and_hash_bound(self):
        from trifusesurv2.models.contour_habitat_survival import ContourAwareHabitatSurvivalModel
        class FixedBackbone(torch.nn.Module):
            out_dim = 8
            num_tokens = 6
            def forward(self, image, **kwargs):
                presence = torch.ones(image.shape[0], 6, dtype=torch.bool)
                presence[:, 1:5] = image[:, 0, 0, 0, :4].bool()
                tokens = image.new_zeros(image.shape[0], 6, 8)
                return (tokens, presence, {}) if kwargs.get('return_aux') else (tokens, presence)
        args = self.active(); args.risk_horizon_days = 15.; args.time_bin_width_days = 10.
        model = ContourAwareHabitatSurvivalModel(backbone=FixedBackbone(), num_time_bins=2,
            time_bin_width_days=10., radiomics_token_dim=5, clinical_token_dim=4, model_dim=16, num_heads=4,
            transformer_layers=1, dropout=0., routing_intervention=ROUTING_INTERVENTION_ID).eval()
        q = torch.tensor([[True, True, True, True], [True, False, False, False]])
        image = torch.zeros(2, 1, 4, 4, 4); image[:, 0, 0, 0, :4] = q
        payload = {'x': image, 't': torch.tensor([5., 8.]), 'e': torch.tensor([1., 0.]),
                   'pid': ['001', '002'], 'clinical_tokens': torch.zeros(2, 3, 4),
                   'radiomics_tokens': torch.zeros(2, 4, 5), 'radiomics_presence': q}
        frame = pd.DataFrame({'Patient_ID': ['002', '001'], 'OS.TIME': [8., 5.], 'OS.EVENT': [0, 1]})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source, contract, output = root/'best.pt', root/'contract.json', root/'curves.csv'
            self.checkpoint(source, model, args, self.contract(args));contract.write_text(json.dumps(self.contract(args)))
            receipt = train.export_survival_curves(model, [payload], torch.device('cpu'), outcome_frame=frame,
                args=args, fold=0, split='val', checkpoint_path=source, contract_path=contract, weights='best',
                autocast_ctx=contextlib.nullcontext, teacher_force_alpha=0., output_path=output)
            coverage = receipt['metadata']['routing_intervention_coverage']
            self.assertEqual(coverage['patients'], 2)
            self.assertEqual(coverage['patients_with_effective_mismatch'], 1)
            self.assertEqual(coverage['patients_without_effective_mismatch'], 1)
            self.assertEqual(coverage['active_regional_edges'], 5)
            self.assertEqual(coverage['effective_mismatched_edges'], 4)
            self.assertEqual(coverage['active_identity_fallback_edges'], 1)
            from trifusesurv2.evaluation.curves import file_sha256
            self.assertEqual(coverage['csv_sha256'], file_sha256(root/coverage['artifact']))
            self.assertEqual(coverage['model_contract_sha256'], file_sha256(contract))
            self.assertEqual(coverage['checkpoint_sha256'], file_sha256(source))
            rows = pd.read_csv(root/coverage['artifact'], dtype=str)
            self.assertEqual(rows['Patient_ID'].tolist(), ['002', '001'])
            self.assertEqual(rows['image_presence'].tolist(), ['1000', '1111'])
            self.assertEqual(rows['pt_intra_key'].tolist(), ['pt_intra', 'pt_peri'])
            self.assertEqual(set(coverage['joint_availability_strata']), {'1000|1000', '1111|1111'})

    def test_actual_mask_coverage_rejects_undeclared_routing_before_publication(self):
        aux = self.actual_aux()
        for field in ('routing_assignment', 'local_context_routing', 'local_context_mask'):
            altered = copy.deepcopy(aux)
            if field == 'routing_assignment':
                altered[field][0, 0] = 0
            else:
                altered[field][0, 1, 1] = ~altered[field][0, 1, 1]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'routing'):
                train._routing_coverage_rows(altered, ['001'])

    def test_averaged_weights_keep_explicit_routing_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = self.active(); model = self.model(True)
            source, contract, selected = root/'last.pt', root/'contract.json', root/'selected.pt'
            self.checkpoint(source, model, args, self.contract(args));contract.write_text(json.dumps(self.contract(args)))
            train.persist_selected_inference_weights(model, selected, source_checkpoint=source, contract_path=contract, weights='ema')
            state = torch.load(selected, weights_only=False)
            self.assertEqual(train._checkpoint_routing_metadata(state), routing_intervention_metadata(ROUTING_INTERVENTION_ID))
            train.restore_selected_curve_checkpoint(model, selected)
            with self.assertRaisesRegex(ValueError, 'checkpoint and model'):
                train.restore_selected_curve_checkpoint(self.model(False), selected)


class RoutingInterventionFoldIntegrationTests(unittest.TestCase):
    def test_actual_synthetic_fold_binds_model_checkpoint_and_validation_export(self):
        import test_region_data as fixtures
        fixture = fixtures.RegionDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        rows, features = [], []
        for index in range(5):
            row, manifest, _ = fixture.case(f'{index + 1:03d}')
            row.update(time=120. + index * 120., event=float(index % 2 == 0),
                       AGE=50 + index, HPV=index % 2, T=2, N=1)
            rows.append(row)
            features.append(fixtures.extract.extract_case(manifest, fixtures.FeatureExtractor(),
                                                         fixture.config, software_version='test-adapter'))
        frame = pd.DataFrame(rows)
        metadata, radiomics = fixture.output/'meta.csv', fixture.output/'radiomics.csv'
        frame.to_csv(metadata, index=False)
        pd.DataFrame(features).to_csv(radiomics, index=False)
        argv = ['train', '--meta_csv', str(metadata), '--radiomics_root', str(radiomics),
                '--ct_col', 'ct', '--mask_pt_col', 'pt', '--mask_ln_col', 'ln',
                '--time_col', 'time', '--event_col', 'event', '--device', 'cpu',
                '--development-only', '--cv_folds', '1', '--report_metric', 'nll',
                '--clinical_type_policy', 'approved_categorical_v1',
                '--clinical_cols', 'AGE', 'HPV', 'T', 'N',
                '--routing_intervention', ROUTING_INTERVENTION_ID,
                '--img_size', '32', '32', '32', '--feature_size', '12', '--depths', '1', '1', '1', '1',
                '--img_token_dim', '16', '--v2_model_dim', '16', '--v2_num_heads', '4',
                '--v2_transformer_layers', '1', '--v2_radiomics_pcs_per_group', '2',
                '--epochs', '1', '--batch_size', '1', '--eval_batch_size', '2', '--grad_accumulation_steps', '2',
                '--workers', '0', '--eval_workers', '0', '--no_ema', '--no_swa', '--no_resume',
                '--risk_horizon_days', '240', '--auc_times_days', '120', '240', '--time_bin_width_days', '120',
                '--export_survival_curves', '--curve_export_times_days', '120', '240']
        with patch('sys.argv', argv):
            args = train.parse_args()
        split = {'train': ['001', '002', '003'], 'val': ['004', '005'], 'test': ['must-not-load-test']}
        root = fixture.root/'intervention-run'
        with patch.object(train, 'predict_risk_scores', side_effect=AssertionError('test risk path entered')), \
                contextlib.redirect_stdout(io.StringIO()):
            result = train.run_one_fold(fold=0, args=args, meta=frame, split=split, device=torch.device('cpu'),
                                       scaler=None, autocast_ctx=contextlib.nullcontext, out_root=root)
        fold = root/'fold_00'
        contract = json.loads((fold/'model_contract.json').read_text())
        expected = routing_intervention_metadata(ROUTING_INTERVENTION_ID)
        for source in (contract, contract['experiment_spec']['settings'], contract['experiment_spec']['runtime']):
            self.assertEqual({key: source[key] for key in expected}, expected)
        self.assertEqual(contract['fusion_mode'], 'F3')
        self.assertEqual(contract['routing_intervention_coverage_schema'], train.ROUTING_COVERAGE_SCHEMA)
        self.assertEqual(contract['test_ids'], ['must-not-load-test'])
        for name in ('best.pt', 'last.pt'):
            checkpoint = torch.load(fold/name, weights_only=False, map_location='cpu')
            self.assertEqual(train._checkpoint_routing_metadata(checkpoint), expected)
        selection = json.loads((fold/'development_selection.json').read_text())
        self.assertEqual({key: selection[key] for key in expected}, expected)
        exported = pd.read_csv(fold/'val_survival_curves_export.csv', dtype={'Patient_ID': str})
        self.assertEqual(exported['Patient_ID'].drop_duplicates().tolist(), ['004', '005'])
        self.assertEqual(exported['mode'].unique().tolist(), [ROUTING_EXPORT_LABEL])
        receipt = json.loads((fold/'val_survival_curves_export.csv.json').read_text())
        self.assertEqual({key: receipt['metadata'][key] for key in expected}, expected)
        self.assertEqual(receipt['metadata']['routing_intervention_coverage']['patients'], 2)
        training_export = pd.read_csv(fold/'train_survival_curves_export.csv', dtype={'Patient_ID': str})
        self.assertEqual(training_export['Patient_ID'].drop_duplicates().tolist(), ['001', '002', '003'])
        training_receipt = json.loads((fold/'train_survival_curves_export.csv.json').read_text())
        self.assertEqual(training_receipt['metadata']['routing_intervention_coverage']['patients'], 3)
        self.assertEqual(training_receipt['metadata']['routing_intervention_coverage']['phase'], 'train_eval')
        metrics = pd.read_csv(fold/'metrics.csv').iloc[0]
        self.assertEqual(int(metrics['optimizer_steps_succeeded']), 2)
        self.assertEqual(int(metrics['optimizer_steps_skipped']), 0)
        self.assertEqual(result['export_risks'], {})
        self.assertFalse(list(fold.glob('*test*csv')))


if __name__ == '__main__':
    unittest.main()
