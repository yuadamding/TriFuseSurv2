#!/usr/bin/env python3
"""Report the prespecified phenotype stages without invoking a model or test set."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from zoneinfo import ZoneInfo
import pandas as pd
import yaml

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path): return json.loads(Path(path).read_text())
def table(frame):
    def clean(value): return str(value).replace('|', '/').replace('\n', ' ')
    rows = [[clean(v) for v in frame.columns]] + [[clean(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    if not rows[0]: return '(No columns.)'
    return '\n'.join(['| ' + ' | '.join(rows[0]) + ' |', '| ' + ' | '.join(['---']*len(rows[0])) + ' |'] + ['| ' + ' | '.join(row) + ' |' for row in rows[1:]])
def bools(values): return values.astype(str).str.lower().isin(['true','1','1.0'])

def gate_text(value):
    return 'PASSED' if value and value.get('passed') is True else 'FAILED' if value and value.get('passed') is False else 'Not executed / unavailable'

def figures(root, destination, primary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    produced=[]
    cohort=root/'preparation/cohort_manifest.csv';features=root/'measurements/phenotype_features.csv'
    if cohort.is_file() and features.is_file():
        c=pd.read_csv(cohort,dtype={'patient_id':str});f=pd.read_csv(features,dtype={'patient_id':str})
        ids=c.loc[c.split.eq('train') & bools(c.paired_eligible),'patient_id']
        d=f[f.patient_id.isin(ids)].dropna(subset=['HP','HN','VP_cm3','VN_cm3'])
        if len(d):
            fig,ax=plt.subplots(1,2,figsize=(9,3.8));ax[0].scatter(d.HP,d.HN,s=14,alpha=.65,color='#176b91')
            limit=max(d.HP.max(),d.HN.max());ax[0].plot([0,limit],[0,limit],':',color='#70818a')
            ax[0].set(xlabel='Primary IQR',ylabel='Nodal IQR',title=f'Original qualified training records (n={len(d)})')
            ax[1].scatter(d.VP_cm3,d.VN_cm3,s=14,alpha=.65,color='#176b91');ax[1].set(xscale='log',yscale='log',xlabel='Primary voxel volume (cm³)',ylabel='Nodal voxel volume (cm³)',title='Matching delivered supports; no OS linkage')
            fig.tight_layout();name='measurement_training.png';fig.savefig(destination/name,dpi=180,bbox_inches='tight');plt.close(fig)
            produced.append(('Prespecified measurements in original training records',name))
    q=root/'qualification/paired_decision.json'
    if q.is_file():
        decision=read_json(q);rows=[]
        for condition in decision['conditions']:
            for variable,m in condition.get('measurements',{}).items():
                if all(m.get(k) is not None for k in ['ICC_A_1','ICC_lower95','ICC_upper95']):
                    rows.append((condition['condition']+' '+variable,m['ICC_A_1'],m['ICC_lower95'],m['ICC_upper95']))
        if rows:
            fig,ax=plt.subplots(figsize=(8,4.5));y=np.arange(len(rows))
            for i,(label,point,lower,upper) in enumerate(rows):ax.plot([lower,upper],[i,i],color='#176b91');ax.plot(point,i,'o',color='#176b91')
            ax.set_yticks(y,[r[0] for r in rows]);ax.axvline(.9,ls='--',color='#c67b20');ax.axvline(.75,ls=':',color='#71828a')
            ax.set(xlabel='ICC(A,1), paired-patient bootstrap 95% interval',title='Training-only perturbation agreement');ax.invert_yaxis();fig.tight_layout()
            name='perturbation_agreement.png';fig.savefig(destination/name,dpi=180,bbox_inches='tight');plt.close(fig);produced.append(('Perturbation robustness; the 2-mm condition is stress-only',name))
    if primary and primary.get('usable') and primary.get('compartment_effects'):
        effects=primary['compartment_effects'];names=[k for k in effects if not k.startswith('contrast_HR_')];fig,ax=plt.subplots(figsize=(8,3.6))
        for i,name in enumerate(names):
            e=effects[name];ax.plot([e['lower95'],e['upper95']],[i,i],color='#176b91');ax.plot(e['coefficient'],i,'o',color='#176b91')
        labels=['Primary','Nodal','Direct nodal − primary'][:len(names)]
        ax.set_yticks(range(len(names)),labels);ax.axvline(0,ls=':',color='#71828a');ax.set_xlabel('Log-hazard coefficient per normalized-intensity unit');fig.tight_layout()
        name='primary_effects.png';fig.savefig(destination/name,dpi=180,bbox_inches='tight');plt.close(fig);produced.append(('Prespecified compartment effects and their direct difference',name))
    return produced

def render_html(text, root):
    def inline(s):
        s = html.escape(s)
        s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
        s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', s)
    lines=text.splitlines(); blocks=[]; i=0
    while i<len(lines):
        s=lines[i]
        if not s.strip():i+=1;continue
        if s.startswith('#'):
            m=re.match(r'^(#{1,6}) (.+)$',s)
            if m:blocks.append(f'<h{len(m[1])}>{inline(m[2])}</h{len(m[1])}>');i+=1;continue
        m=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',s)
        if m:
            p=(root/m[2]).resolve();assert p.is_relative_to(root.resolve())
            blocks.append('<figure><img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"><figcaption>'+inline(m[1])+'</figcaption></figure>');i+=1;continue
        if s.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                r=[v.strip() for v in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(':?-+:?',v) for v in r):rows.append(r)
                i+=1
            blocks.append('<table><thead><tr>'+''.join('<th>'+inline(v)+'</th>' for v in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(v)+'</td>' for v in r)+'</tr>' for r in rows[1:])+'</tbody></table>');continue
        paragraph=[s];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
            paragraph.append(lines[i]);i+=1
        blocks.append('<p>'+inline(' '.join(paragraph))+'</p>')
    style='''@page{size:A4;margin:17mm}body{font:10.5pt/1.5 Arial;color:#213744;max-width:1000px;margin:30px auto;padding:0 16px}h1{font-size:25pt;color:#15546e}h2{font-size:16pt;color:#15546e;border-bottom:1px solid #bdd0da;break-after:avoid}h3{break-after:avoid}table{border-collapse:collapse;width:100%;font-size:8.5pt;margin:14px 0}th,td{padding:6px;border-bottom:1px solid #dae3e8;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#e9f1f5}tr{break-inside:avoid}thead{display:table-header-group}img{max-width:100%;max-height:180mm}figure{margin:20px 0;break-inside:avoid}figcaption{font-size:9pt;color:#526b78}code{overflow-wrap:anywhere;font-size:.9em}a{color:#176584}p{orphans:3;widows:3}@media print{body{margin:0;padding:0}}'''
    return '<!doctype html><html><head><meta charset="utf-8"><title>TriFuseSurv phenotype substudy</title><style>'+style+'</style></head><body>'+''.join(blocks)+'</body></html>'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--pdf',action='store_true');a=parser.parse_args()
    protocol=yaml.safe_load(a.protocol.read_text());lock=read_json(Path(protocol['paths']['protocol_lock']))
    assert lock['protocol_sha256']==sha(a.protocol) and lock['code_sha256']['report.py']==sha(__file__)
    root=a.run_root.resolve();destination=root/'reports';destination.mkdir(exist_ok=True)
    def optional(name):
        path=root/name
        return read_json(path) if path.is_file() else None
    paths={'paired qualification':'qualification/paired_decision.json',
           'primary-only qualification':'qualification/primary_only_decision.json',
           'paired design':'statistics/design_assessment.json',
           'primary-only design':'statistics/fallback_design_assessment.json'}
    gates={label:optional(path) for label,path in paths.items()}
    now=datetime.now(timezone.utc);display=now.astimezone(ZoneInfo('America/Chicago')).strftime('%B %d, %Y, %I:%M %p %Z')
    sections=['# TriFuseSurv prespecified compartment phenotype substudy',f'Report snapshot: **{display}**.',
      'The question is whether primary and nodal CT intensity IQR have different OS coefficients after adjustment for recorded age, sex and their separate voxel volumes. This is a separately specified substudy; the main neural experiment and original source records/split labels are preserved.',
      '## Protocol and interpretation',f'Protocol SHA-256: `{lock["protocol_sha256"]}`. Bound source: `{lock.get("source_commit","unavailable")}`.',
      'IQR is Q75 minus Q25 with linear quantiles, in delivered normalized-intensity units. M=(HP+HN)/2 and D=(HN−HP)/2; D is signed nodal-relative-to-primary dispersion. The two compartments are not independently standardized. Voxel volume, not mesh volume, supplies the volume covariates.',
      'Authentic scan/diagnosis/treatment timing is required for a verified baseline or delayed-entry interpretation. Unknown timing remains unknown in the manifests; any association on diagnosis-origin OS is explicitly exploratory retrospective, not verified pretreatment prediction.',
      '## Exact eligibility and outcome counts']
    count_file=root/'preparation/eligibility_counts.csv'
    if count_file.is_file():
        counts=pd.read_csv(count_file);sections.append(table(counts))
    else:sections.append('Final eligibility has not been completed. Preliminary candidate counts are not final analysis eligibility.')
    timing=optional('preparation/timing_audit.json')
    if timing:
        sections.append('Timing status counts: `'+json.dumps(timing.get('status_counts',{}),sort_keys=True)+'`. Processing timestamps, file modification times and UID/path tokens are not used as clinical dates.')
    sections+=['Both records in every unresolved image-source conflict are analytically excluded through manifest flags. No source file is moved or deleted. Every applicable exclusion and a fixed-order flow are retained to avoid double counting.',
      'The original training and validation portions constitute development for this substudy; the original test portion remains reserved. Training alone supplies measurement qualification. Outcome counts are cohort descriptions, not phenotype–outcome associations.']
    sections+=['## Measurement qualification and design decisions',table(pd.DataFrame([{'gate':k,'decision':gate_text(v)} for k,v in gates.items()]))]
    for label,value in gates.items():
        if not value:continue
        sections.append('### '+label.capitalize())
        if 'conditions' in value:
            rows=[]
            for condition in value['conditions']:
                measures=condition.get('measurements',{})
                for name,m in measures.items():
                    rows.append({'condition':condition['condition'],'variable':name,
                      'gating':condition['gating'],'available/baseline':f'{condition["evaluable_n"]}/{condition["baseline_training_n"]}',
                      'coverage':round(condition['coverage'],4),'ICC(A,1)':m.get('ICC_A_1'),
                      'CI lower':m.get('ICC_lower95'),'CI upper':m.get('ICC_upper95'),
                      'median change / train SD':m.get('median_absolute_change_in_baseline_training_SD'),
                      'passed':m.get('passed')})
            sections.append(table(pd.DataFrame(rows)))
            sections.append('Unavailable perturbations: `'+json.dumps({c['condition']:c.get('unavailable_reasons',{}) for c in value['conditions']},sort_keys=True)+'`.')
        if 'scenarios' in value:
            scenarios=value['scenarios'];scenarios=scenarios if isinstance(scenarios,list) else [dict(scenario=k,**v) for k,v in scenarios.items()]
            sections.append(table(pd.DataFrame(scenarios)))
    sections+=['For each gating perturbation—physical 1-mm erosion, padded physical 1-mm dilation and edge-preserving 1-mm isotropic coarsening—the fixed criteria require ≥90% evaluability, ICC(A,1) ≥0.90, lower 95% paired-bootstrap bound ≥0.75, and median absolute change ≤0.20 original-training SD. The 2-mm grid is a stress test, not a replacement gate. Unavailable supports remain unavailable rather than being clipped or silently retained.',
      'These are project perturbation-robustness criteria, not inter-reader, test–retest or biological-validity claims. No criterion is relaxed in response to these results. A zero IQR remains a valid observation; negligible between-patient variation cannot be repaired by adding a constant.']
    primary=optional('statistics/primary_result.json');fallback=optional('statistics/fallback_result.json')
    sections.append('## Association execution status')
    if primary:
        sections.append('The prespecified primary association was executed after its locked gates. Numerical result:')
        selected={k:primary.get(k) for k in ['usable','n_patients','deaths','full_slopes','LRT','df','p_value','time_interpretation','adjustment_scope','bootstrap']}
        sections.append('`'+json.dumps(selected,sort_keys=True)+'`')
        if primary.get('compartment_effects'):
            sections.append(table(pd.DataFrame([dict(effect=k,**v) for k,v in primary['compartment_effects'].items()])))
    elif (root/'statistics/PRIMARY_ASSOCIATION_STARTED.json').exists():
        sections.append('A real primary association was attempted, but no completed accepted result is available. Inspect the retained attempt and failure evidence; absence of a result is not proof that no model was attempted.')
    else:sections.append('**No real primary compartment–OS association has been fitted.** A failed measurement or design gate stops that test; it is not a null association result.')
    if fallback:
        sections.append('The separately qualified primary-only sphericity fallback was executed before any paired real association. Results remain a different morphology question, not a rescue of a null paired finding.')
        sections.append('`'+json.dumps({k:fallback.get(k) for k in ['usable','n_patients','deaths','full_slopes','LRT','p_value','coefficients','time_interpretation','bootstrap']},sort_keys=True)+'`')
    elif (root/'statistics/FALLBACK_ASSOCIATION_STARTED.json').exists():
        sections.append('A primary-only association was attempted, but no completed accepted result is available; retain and inspect its failure evidence.')
    else:sections.append('No real primary-only sphericity–OS association has been fitted. The fallback requires its own measurement/design qualification and cannot be promoted after observing a paired null result.')
    test=optional('statistics/heldout_assessment.json');fallback_test=optional('statistics/fallback_heldout_assessment.json')
    sections.append('Reserved-test application: '+('Executed; see frozen assessment artifact.' if test or fallback_test else '**Not executed.** No test-set effect fitting, tuning or prediction comparison has been performed by this pipeline.'))
    flow=root/'preparation/eligibility_flow.csv'
    if flow.is_file():
        f=pd.read_csv(flow);sections+=['## Fixed-order cohort flow',table(f[f.split.eq('all')])]
    for caption,name in figures(root,destination,primary):sections.append(f'![{caption}]({name})')
    sections+=['## Evidence and limitations',
      'Input/source manifest, restricted source-conflict records, timing evidence, cohort CSV/Parquet, measurement CSV/Parquet, numerical validation, perturbation records and per-gate decisions are retained in this local run directory. The code is source-bound; original split labels are retained in every manifest.',
      'Delivered images are previously normalized and cropped. The substudy does not validate original HU calibration, restore cropped tissue, establish patient identity, recover missing clinical code meanings or independently validate contour truth. The existing source-integrity screen is overlap-aware and registration-free; absence of a screen match is not proof of independence under arbitrary transforms.',
      'The primary six-slope Cox specification is age + sex + log primary volume + log nodal volume + M + D, with Efron ties and a same-patient one-degree-of-freedom nested likelihood-ratio test of D=0. The coefficient difference is tested directly. Stage, annotated-volume allocation, the fixed 1,826-day temporal extension and literal HPV-coded sensitivity have prespecified supporting roles; none replaces a failed or null primary result.',
      'Measurement definition: [PyRadiomics first-order documentation](https://pyradiomics.readthedocs.io/en/v3.0.1/_modules/radiomics/firstorder.html). Temporal risk-set requirements: [survival time-dependent-covariate vignette](https://stat.ethz.ch/R-manual/R-devel/library/survival/doc/timedep.pdf). Nested-comparison requirement: [anova.coxph](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/anova.coxph.html).']
    body='\n\n'.join(sections)+'\n';md=destination/'PHENOTYPE_SUBSTUDY_REPORT.md';md.write_text(body)
    html_path=destination/'PHENOTYPE_SUBSTUDY_REPORT.html';html_path.write_text(render_html(body,destination))
    known=['preparation/source_manifest.json','preparation/cohort_manifest.parquet','preparation/eligibility_counts.csv',
      'preparation/eligibility_flow.csv','preparation/timing_audit.json','measurements/phenotype_features.parquet',
      'measurements/measurement_validation.csv',*paths.values(),'statistics/primary_result.json',
      'statistics/fallback_result.json','statistics/heldout_assessment.json']
    bindings=[{'path':name,'bytes':(root/name).stat().st_size,'sha256':sha(root/name)} for name in known if (root/name).is_file()]
    result={'at_utc':now.isoformat(),'at_houston':display,'protocol_sha256':lock['protocol_sha256'],
      'code_sha256':sha(__file__),'gates':{k:gate_text(v) for k,v in gates.items()},
      'primary_association_attempted':primary is not None or (root/'statistics/PRIMARY_ASSOCIATION_STARTED.json').exists(),
      'primary_completed_result_available':primary is not None,
      'fallback_association_attempted':fallback is not None or (root/'statistics/FALLBACK_ASSOCIATION_STARTED.json').exists(),
      'fallback_completed_result_available':fallback is not None,
      'heldout_application_executed':test is not None or fallback_test is not None,'artifacts':bindings,
      'report_sha256':sha(md),'model_or_association_fits_by_report':0,'remote_calls':0}
    if a.pdf:
        profile=destination/'fresh-pdf-profile';assert not profile.exists()
        cmd=['/usr/bin/google-chrome','--headless=new','--disable-gpu','--disable-background-networking',
          '--disable-component-update','--disable-sync','--disable-default-apps','--disable-extensions',
          '--no-first-run','--no-default-browser-check','--disable-dev-shm-usage','--no-pdf-header-footer',
          '--user-data-dir='+str(profile),'--print-to-pdf='+str(destination/'PHENOTYPE_SUBSTUDY_REPORT.pdf'),html_path.as_uri()]
        with (destination/'pdf.log').open('w') as stream:rendered=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        assert rendered.returncode==0
        if profile.exists():shutil.rmtree(profile)
        result['PDF_sha256']=sha(destination/'PHENOTYPE_SUBSTUDY_REPORT.pdf')
    (destination/'REPORT_INDEX.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'report':str(md),'gates':result['gates'],'real_primary_association':primary is not None},sort_keys=True))

if __name__=='__main__': main()
