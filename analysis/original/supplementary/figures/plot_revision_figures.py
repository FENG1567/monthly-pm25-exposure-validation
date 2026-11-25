#!/usr/bin/env python3
"""Build the STOTEN revision figure set from catalog-verified saved results.

This script uses only the persisted Python/matplotlib backend. All data
selection is explicit below; panel CSVs, full source tables, hashes, and
captions are written beside the rendered figures.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

HERE = Path(__file__).resolve().parent
OUT = HERE
SOURCE_OUT = OUT / "source_data"
INPUTS = HERE / "inputs"
V = Path(os.environ.get('FIGURE_VALIDATION_ROOT', str(HERE.parents[1])))
DATA = V / 'submission_publication_normalized_v2_20261115'
RESULTS = V / 'stoten_revision_20261116' / 'results'
MONTHLY = V / 'monthly_satpm_v1_20261116'
PROTOCOL = V / 'stoten_revision_20261116' / 'protocol'
GRID = V / 'final_reference16_grids_v1_20261115'
THIN_RUN = V/'station_thinning_final_execution_v1_20261115/final_run/run_20261115T020544Z_32c4673b59'
THIN_ACCEPT = V/'station_thinning_postfit_audit_v1_20261115/acceptance.json'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'svg.fonttype':'none','pdf.fonttype':42,'font.size':6.5,'axes.spines.right':False,
    'axes.spines.top':False,'axes.linewidth':0.65,'legend.frameon':False,
    'savefig.facecolor':'white','figure.facecolor':'white','axes.labelsize':6.5,
    'xtick.labelsize':5.8,'ytick.labelsize':5.8})
BLUE='#3B6C8E'; ORANGE='#D98847'; GREEN='#568F7B'; RED='#A85858'; GREY='#777777'; LIGHT='#D9E2E8'; PURPLE='#81739A'
COUNTRIES=['AUS','BD','BRA','CAN','CL','ES','GR','IN','IT','MEX','NP','PK','PT','TH','USA','VN','ZA']
TASKS=['T1_station','T2_spatial','T2_spatial_buffer50km','T3_year_2022']
TASK_LABELS={'T1_station':'T1 station','T2_spatial':'T2 spatial','T2_spatial_buffer50km':'T2 buffer','T3_year_2022':'T3 2022'}
FIGURES=[]; VERIFIED={}

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def verify_catalog():
    cat=json.loads((DATA/'catalog.json').read_text(encoding='utf-8'))
    checked={}
    for ent in cat.get('catalog',[]):
        name=ent.get('name')
        if not name: continue
        p=DATA/f'{name}.csv'
        if p.exists():
            d=sha(p)
            if d!=ent.get('sha256'): raise RuntimeError(f'Catalog hash mismatch: {p.name}')
            checked[name]=d
    if len(checked)<100: raise RuntimeError(f'Expected complete old catalog verification; checked {len(checked)}')
    return checked

def load(name:str)->pd.DataFrame:
    return pd.read_csv(DATA/f'{name}.csv',low_memory=False)

def bool_series(s):
    return s.astype(str).str.lower().isin(['true','1','yes'])

def panel_label(ax, label):
    ax.text(-0.10,1.03,label,transform=ax.transAxes,fontsize=8,fontweight='bold',va='bottom')

def sanitize_frame(df):
    out=df.copy()
    for col in out.columns:
        if 'path' in str(col).lower() or 'root' in str(col).lower():
            mask=out[col].notna() & out[col].astype(str).ne('')
            out.loc[mask,col]='[redacted; provenance bound by source alias and hash]'
    return out

def source_info(stem, alias, path, role='saved analysis source', extra=None):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    dest=SOURCE_OUT/f'{stem}__{alias}{path.suffix}'
    dest.parent.mkdir(parents=True,exist_ok=True)
    original_hash=sha(path);n_rows=None;redacted_columns=[]
    if path.suffix.lower()=='.csv' and path.resolve()!=dest.resolve():
        import csv
        with path.open('r',encoding='utf-8-sig',errors='replace',newline='') as fin, dest.open('w',encoding='utf-8',newline='') as fout:
            reader=csv.reader(fin);writer=csv.writer(fout)
            header=next(reader,[]);writer.writerow(header)
            ix=[i for i,c in enumerate(header) if 'path' in str(c).lower() or 'root' in str(c).lower()]
            redacted_columns=[header[i] for i in ix]
            n_rows=0
            for row in reader:
                for i in ix:
                    if i<len(row) and row[i].strip(): row[i]='[redacted; provenance bound by source alias and hash]'
                writer.writerow(row);n_rows+=1
    elif path.suffix.lower()=='.csv':
        with path.open('r',encoding='utf-8-sig',errors='replace') as f: n_rows=max(0,sum(1 for _ in f)-1)
    elif path.resolve()!=dest.resolve(): shutil.copyfile(path,dest)
    out={'alias':alias,'role':role,'filename':str(dest.relative_to(OUT)).replace('\\','/'),
         'rows':n_rows,'sha256':sha(dest),'original_source_sha256':original_hash if original_hash!=sha(dest) else None}
    if redacted_columns: out['redacted_path_columns']=redacted_columns
    if extra: out.update(extra)
    return out

def save_pub(stem, fig, caption, panels, *, core_claim, sources, filters, replicate_meaning,
             stats='Descriptive saved values; no new inference was computed.', excluded=None):
    if 'Source data are provided' not in caption:
        caption=caption.rstrip()+ ' Source data are provided in the accompanying panel and complete-source files.'
    fig.savefig(OUT/f'{stem}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{stem}.svg',bbox_inches='tight')
    fig.savefig(OUT/f'{stem}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    files=[f'{stem}.pdf',f'{stem}.png',f'{stem}.svg']
    panel_counts={}
    for label,df in panels.items():
        fn=f'{stem}_Panel_{label}_Source_Data.csv'
        sanitize_frame(df).to_csv(OUT/fn,index=False,na_rep='')
        files.append(fn); panel_counts[label]=int(len(df))
    full_sources=[source_info(stem,s['alias'],s['path'],s.get('role','saved analysis source'),
         {k:s[k] for k in s if k not in {'alias','path','role'}}) for s in sources]
    for fn in files:
        (OUT/fn).resolve()
    rec={'figure':stem,'core_claim':core_claim,'caption':caption,'panels':list(panels),
         'panel_source_rows':panel_counts,'filters':filters,'excluded_or_unplotted':excluded or [],
         'replicate_unit_and_statistical_meaning':replicate_meaning,'statistics':stats,
         'source_tables':full_sources,'files':files,
         'file_sha256':{fn:sha(OUT/fn) for fn in files},
         'dimensions_mm':'183 mm width; height below 240 mm','backend':'Python/matplotlib',
         'pdf_text':'TrueType editable text; SVG text remains editable'}
    FIGURES.append(rec)
    (OUT/f'{stem}_caption_en.md').write_text(f'# {stem.replace("_"," ")}\n\n{caption}\n',encoding='utf-8')
    return rec

def register_catalog(stem,*aliases):
    return [{'alias':a,'path':DATA/f'{a}.csv','role':'full catalogued source table; complete rows retained'} for a in aliases]

def register_input(stem,*names):
    return [{'alias':Path(n).stem,'path':INPUTS/n,'role':'formal Figure_Source_Data input; complete rows retained'} for n in names]

def verify_thinning_sources():
    acceptance=json.loads(THIN_ACCEPT.read_text(encoding='utf-8'))
    if acceptance.get('status')!='accepted': raise RuntimeError('Station-thinning acceptance status is not accepted')
    def find_pin(obj):
        if isinstance(obj,dict):
            if 'output_artifact_sha256' in obj and isinstance(obj['output_artifact_sha256'],dict): return obj['output_artifact_sha256']
            for value in obj.values():
                found=find_pin(value)
                if found is not None: return found
        elif isinstance(obj,list):
            for value in obj:
                found=find_pin(value)
                if found is not None: return found
        return None
    pins=find_pin(acceptance)
    if not pins: raise RuntimeError('No output artifact hashes in thinning acceptance record')
    checked={}
    for name in ['metrics.csv','paired_deltas.csv','repeat_ranges.csv','level_status.csv']:
        p=THIN_RUN/name
        if not p.exists(): raise FileNotFoundError(p)
        actual=sha(p)
        if actual!=pins.get(name): raise RuntimeError(f'Thinning artifact hash mismatch for {name}')
        checked[name]=actual
    return checked

def hist_primary(raw):
    x=raw[(raw.qa_variant=='strict')&(raw.estimator_kind=='hgb')&(raw.arm=='full')&
          (raw.target_strategy=='log_smear')&(raw.backtransform=='smearing')&
          (raw.smear_strategy=='crossfit_station')&
          (((raw.country=='USA')&(raw.dataset_key=='usa_hist'))|
           ((raw.country=='CAN')&(raw.dataset_key=='can_hist_primary'))|
           (~raw.country.isin(['USA','CAN'])&raw.dataset_key.isna()))].copy()
    return x

def future_primary(raw):
    policy=raw.country.map({'USA':'usa_primary','CAN':'can_primary_all_published'})
    return raw[(raw.qa_variant=='strict')&(raw.estimator_kind=='hgb')&(raw.arm=='full')&
        (raw.target_strategy=='log_smear')&(raw.backtransform=='smearing')&
        (raw.smear_strategy=='crossfit_station')&raw.country.isin(['USA','CAN'])&
        (raw.dataset_policy==policy)].copy()

def plot_f1():
    raw=load('historical_metrics'); elig=load('historical_eligibility'); h=hist_primary(raw)
    if len(h)!=108 or h.duplicated(['country','profile','task']).any(): raise RuntimeError('F1 primary source cardinality/key check failed')
    e=elig[(elig.qa_variant=='strict')&(elig.estimator_kind=='hgb')&
      (((elig.country=='USA')&(elig.dataset_key=='usa_hist'))|((elig.country=='CAN')&(elig.dataset_key=='can_hist_primary'))|
       (~elig.country.isin(['USA','CAN'])&elig.dataset_key.isna()))].copy()
    rows={}; fig,axs=plt.subplots(2,2,figsize=(7.2,7.5));
    for ri,(metric,title,cmap) in enumerate([('r2','R²','RdBu_r'),('rmse','RMSE (µg m⁻³)','Blues')]):
        for ci,profile in enumerate(['with_met','no_met']):
            ax=axs[ri,ci]; q=h[h.profile==profile][['country','profile','task','dataset_key','source_job_id','n','n_stations','r2','rmse']].copy()
            expected=pd.MultiIndex.from_product([COUNTRIES,TASKS],names=['country','task']).to_frame(index=False)
            panel=expected.merge(q,on=['country','task'],how='left',validate='one_to_one'); es=e[e.profile==profile]
            if len(es):
                ec=es.groupby(['country','task']).agg(eligible_folds=('status',lambda z:int((z=='eligible').sum())),
                    ineligible_folds=('status',lambda z:int((z!='eligible').sum())),
                    guard_reasons=('reason',lambda z:'; '.join(sorted(set(str(v) for v in z.dropna() if str(v).strip()))))).reset_index()
                panel=panel.merge(ec,on=['country','task'],how='left',validate='one_to_one')
            else: panel['eligible_folds']=np.nan;panel['ineligible_folds']=np.nan;panel['guard_reasons']=''
            panel['cell_status']=np.where(panel[metric].notna(),'reported',np.where(panel.ineligible_folds.fillna(0)>0,'guard/ineligible','missing'))
            values=panel[metric].dropna().to_numpy(float)
            if not len(values): raise RuntimeError(f'F1 empty panel {metric}/{profile}')
            if metric=='r2': norm=SymLogNorm(linthresh=.1,vmin=float(values.min()),vmax=float(values.max()))
            else: norm=mpl.colors.Normalize(vmin=0,vmax=float(values.max()))
            im=ax.imshow(np.ma.masked_invalid(panel[metric].to_numpy(float).reshape(17,4)),aspect='auto',cmap=cmap,norm=norm)
            ax.set_facecolor('#ECEFF1')
            for y in range(17):
                for x in range(4):
                    v=panel.iloc[y*4+x][metric]
                    if pd.isna(v):
                        val='guard' if panel.iloc[y*4+x].cell_status=='guard/ineligible' else 'NA'; col=GREY
                    else:
                        val=f'{v:.2f}' if metric=='r2' else f'{v:.1f}'
                        rgb=im.cmap(im.norm(float(v)))[:3]; col='white' if np.dot(rgb,[.2126,.7152,.0722])<.48 else '#203038'
                    ax.text(x,y,val,ha='center',va='center',fontsize=5.3,color=col)
            ax.set_yticks(range(17),COUNTRIES);ax.set_xticks(range(4),['T1','T2','T2 buffer','T3 2022'])
            ax.set_title(f'{chr(97+ri*2+ci)}  {title} · {profile.replace("_"," ")}',loc='left',fontsize=7.3)
            ax.set_xlabel('Deployment task');ax.set_ylabel('Country');ax.tick_params(length=0)
            fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.09,label=title)
            rows[chr(97+ri*2+ci)]=panel
    fig.suptitle('Historical validation varies by country and deployment task',fontsize=9,y=1.01)
    fig.text(.5,.01,'Negative R² is retained on a signed-log color scale; gray cells are unavailable or guarded.',ha='center',fontsize=6.5)
    fig.subplots_adjust(left=.09,right=.98,bottom=.07,top=.96,wspace=.28,hspace=.28)
    cap=('Figure1 | Historical validation differs across country and deployment task. Strict-QA HGB full models with log-smear targets and cross-fitted station smearing are shown for the locked primary US/Canadian cohorts and the catalogued domestic cohorts elsewhere. R² uses a signed logarithmic color scale while labels show raw values; RMSE is shown on a separate linear scale. T1 is station grouping, T2 spatial blocking, T2 buffer the 50-km buffer, and T3 2022 historical forward prediction. Gray cells distinguish unavailable metrics from ineligible folds in the panel source data; negative R² values are not truncated.')
    save_pub('Figure_1_Historical_Validation',fig,cap,rows,core_claim='Validation performance is country- and task-specific; a country ranking is not the claim.',
       sources=register_catalog('F1','historical_metrics','historical_eligibility'),
       filters={'qa_variant':'strict','estimator_kind':'hgb','arm':'full','target_strategy':'log_smear','backtransform':'smearing','smear_strategy':'crossfit_station','dataset_key':{'USA':'usa_hist','CAN':'can_hist_primary','other countries':'null'}},
       replicate_meaning='Each cell is a saved validation estimate; n is test station-month rows and n_stations is distinct stations. No uncertainty interval added.',
       excluded=['Source rows outside the fixed primary QA/model/target/cohort contract are not drawn; complete historical_metrics and historical_eligibility tables are included in source_data.'])

def plot_f2():
    fut=future_primary(load('future_metrics')); fut=fut[fut.validation_mode.isin(['fixed','rolling'])].copy()
    if fut.duplicated(['country','profile','validation_mode','external_year']).any(): raise RuntimeError('F2 duplicate country/profile/mode/year key')
    if fut.empty: raise RuntimeError('F2 primary future source is empty')
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.3),sharex=True); panels={}
    for ri,country in enumerate(['USA','CAN']):
        for ci,(metric,label) in enumerate([('r2','R²'),('rmse','RMSE (µg m⁻³)')]):
            ax=axs[ri,ci]; sub=fut[fut.country==country].copy(); panel=sub[['country','profile','validation_mode','external_year','train_through','task','dataset_policy','n','n_stations','r2','rmse','source_job_id']].copy(); panel['metric']=metric;panels[f'{country}_{metric}']=panel
            for profile,color,marker in [('with_met',BLUE,'o'),('no_met',ORANGE,'s')]:
                for mode,ls in [('fixed','-'),('rolling','--')]:
                    q=sub[(sub.profile==profile)&(sub.validation_mode==mode)].sort_values('external_year')
                    if len(q): ax.plot(q.external_year,q[metric],color=color,marker=marker,ms=3,lw=.9,ls=ls,label=f'{profile.replace("_"," ")}, {mode}')
            ax.axvline(2023,color=GREY,lw=.65,ls=':');ax.grid(axis='y',color=LIGHT,lw=.45)
            ax.set_title(f'{chr(97+ri*2+ci)}  {country} · {label}',loc='left',fontsize=7.5);ax.set_ylabel(label);ax.set_xlabel('External evaluation year');ax.set_xticks(sorted(fut.external_year.unique()))
            panel_label(ax,chr(97+ri*2+ci))
    handles,labels=[],[]
    for ax in axs.ravel():
        h,l=ax.get_legend_handles_labels()
        for hh,ll in zip(h,l):
            if ll not in labels: handles.append(hh);labels.append(ll)
    fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.01),fontsize=6.2)
    fig.suptitle('Future evaluation depends on year and update policy',y=1.045,fontsize=9);fig.text(.5,.005,'Lines connect observed years only; unavailable values are not imputed.',ha='center',fontsize=6.3)
    fig.tight_layout(rect=[0,.03,1,.93])
    cap=('Figure2 | Future evaluation depends on year, meteorological profile and update policy. Strict-QA HGB full-model R² and RMSE are shown for the primary US and Canadian policies. Solid and dashed lines denote fixed and rolling updates; symbols denote meteorological profile. The dotted reference marks the consulted 2023 evaluation. Lines join only observed years; no Canadian 2025 value is imputed. Lower RMSE is descriptive for the stated evaluation support and is not an unconditional comparison across years.')
    save_pub('Figure_2_Future_Evaluation',fig,cap,panels,core_claim='Future performance varies with evaluation year and model-update policy.',sources=register_catalog('F2','future_metrics'),
       filters={'qa_variant':'strict','estimator_kind':'hgb','arm':'full','target_strategy':'log_smear','backtransform':'smearing','smear_strategy':'crossfit_station','dataset_policy':{'USA':'usa_primary','CAN':'can_primary_all_published'},'validation_mode':['fixed','rolling']},
       replicate_meaning='Saved country/profile/mode/year estimate; n is the evaluation row count and n_stations the distinct evaluation sites. No interval supplied.',excluded=['Unavailable country-years remain absent; 2023 line is a reference, not an imputed observation.'])

def plot_f3():
    raw=load('joint_future')
    required={'country','qa_variant','profile','task','target_strategy','backtransform','rule','fire_month_scope','retained','n','eligible_rows','n_stations','eligible_stations','row_coverage_fraction','station_coverage_fraction','full_rmse','dataset_policy','validation_mode','estimator_kind'}
    if not required.issubset(raw.columns): raise RuntimeError(f'F3 missing fields: {sorted(required-set(raw.columns))}')
    retained=bool_series(raw.retained)
    x=raw[(raw.qa_variant=='strict')&(raw.estimator_kind=='hgb')&(raw.target_strategy=='log_smear')&(raw.backtransform=='smearing')&
          raw.country.isin(['USA','CAN'])&raw.profile.isin(['no_met','with_met'])&raw.task.astype(str).str.startswith('T3_fixed_')&
          (raw.rule=='all')&(raw.validation_mode=='fixed')&(raw.dataset_policy==raw.country.map({'USA':'usa_primary','CAN':'can_primary_all_published'}))&retained&
          raw.fire_month_scope.isin(['P90_fire_proxy','other_months'])].copy()
    key=['country','profile','task','fire_month_scope']
    if x.empty or x.duplicated(key).any(): raise RuntimeError('F3 source empty or duplicated under the fixed primary key')
    x['row_label']=x.country+' '+x.task.str.replace('T3_fixed_','',regex=False)+' '+x.profile.str.replace('_',' ',regex=False)
    x=x.sort_values(['country','task','profile','fire_month_scope']); labels=sorted(x.row_label.unique()); ymap={v:i for i,v in enumerate(labels)}
    fig,axs=plt.subplots(1,2,figsize=(7.2,5.0),gridspec_kw={'width_ratios':[1.25,1]})
    ax=axs[0]; colors={'with_met':BLUE,'no_met':ORANGE};markers={'P90_fire_proxy':'o','other_months':'s'}
    for scope in ['P90_fire_proxy','other_months']:
        q=x[x.fire_month_scope==scope]
        yy=np.array([ymap[v] for v in q.row_label]) + (-.10 if scope=='P90_fire_proxy' else .10)
        ax.scatter(q.full_rmse,yy,c=q.profile.map(colors),marker=markers[scope],s=13,label='P90 fire-proxy months' if scope=='P90_fire_proxy' else 'Other months',edgecolors='white',linewidths=.25)
    ax.set_yticks(range(len(labels)),labels,fontsize=5.4);ax.invert_yaxis();ax.set_xlabel('Full-model RMSE (µg m⁻³)');ax.set_title('a  Held-out error by month stratum',loc='left',fontsize=7.5);ax.grid(axis='x',color=LIGHT,lw=.45)
    # Legend distinguishes profile color from fire-month symbol.
    hs=[plt.Line2D([],[],marker='o',color=BLUE,lw=0,label='with MET'),plt.Line2D([],[],marker='o',color=ORANGE,lw=0,label='no MET'),
        plt.Line2D([],[],marker='o',color='#555555',lw=0,label='P90 fire proxy'),plt.Line2D([],[],marker='s',color='#555555',lw=0,label='other months')]
    ax.legend(handles=hs,fontsize=5.5,loc='lower right',ncol=2)
    # Support panel uses literal nontrivial spatial-support rules from the
    # complete risk summary.  The all-month row is a reference only.
    risk=pd.read_csv(RESULTS/'csv'/'future_risk_summary.csv',low_memory=False)
    p_required={'summary_class','risk_table_included','country','profile','task','target_strategy','backtransform','rule','fire_month_scope','retained','n','eligible_rows','row_coverage_fraction','dataset_policy','validation_mode','estimator_kind'}
    if not p_required.issubset(risk.columns): raise RuntimeError(f'F3 support source missing fields: {sorted(p_required-set(risk.columns))}')
    risk_keep=(bool_series(risk.risk_table_included)&bool_series(risk.retained)&(risk.summary_class=='primary_strict_hgb_log_smear_smearing')&
        (risk.estimator_kind=='hgb')&(risk.target_strategy=='log_smear')&(risk.backtransform=='smearing')&
        risk.country.isin(['USA','CAN'])&risk.profile.isin(['with_met','no_met'])&(risk.task=='T3_fixed_2023')&
        (risk.fire_month_scope=='P90_fire_proxy')&(risk.validation_mode=='fixed')&
        (risk.dataset_policy==risk.country.map({'USA':'usa_primary','CAN':'can_primary_all_published'})))
    p=risk[risk_keep].copy()
    # Preserve exact source rule literals and include all-rule only as a
    # neutral reference.  No all-rule value is interpreted as 100% retention.
    desired_rules=['distance_50km_non_geo_minmax','distance_50km_non_geo_p01_p99','all']
    if set(desired_rules)-set(p.rule.unique()):
        raise RuntimeError(f'F3 expected source rule literals {desired_rules}; found {sorted(p.rule.unique())}')
    p=p[p.rule.isin(desired_rules)].copy()
    p['rule_label']=p.rule.map({'distance_50km_non_geo_minmax':'Distance 50 km + non-geographic min–max','distance_50km_non_geo_p01_p99':'Distance 50 km + non-geographic 1st–99th pct.','all':'All rows (reference)'})
    p['row_label']=p.country+' '+p.task.str.replace('T3_fixed_','',regex=False)+' '+p.profile.str.replace('_',' ',regex=False)
    expected={(c,t,pr,r) for c in ['USA','CAN'] for t in sorted(p.task.unique()) for pr in ['with_met','no_met'] for r in desired_rules}
    pkey=['country','task','profile','rule']
    if p.duplicated(pkey).any() or len(p)!=len(expected): raise RuntimeError(f'F3 risk support key/count mismatch: {len(p)} vs {len(expected)}')
    if not np.allclose(pd.to_numeric(p.row_coverage_fraction,errors='coerce'),pd.to_numeric(p.n,errors='coerce')/pd.to_numeric(p.eligible_rows,errors='coerce'),atol=1e-12,rtol=0):
        raise RuntimeError('F3 row coverage differs from source n / eligible_rows')
    if int(p.loc[(p.country=='CAN')&(p.profile=='with_met')&(p.rule=='distance_50km_non_geo_minmax'),'n'].sum())!=466 or int(p.loc[(p.country=='CAN')&(p.profile=='with_met')&(p.rule=='distance_50km_non_geo_minmax'),'eligible_rows'].sum())!=725:
        raise RuntimeError('F3 expected CAN with-MET distance50+minmax support 466/725')
    if int(p.loc[(p.country=='CAN')&(p.profile=='with_met')&(p.rule=='distance_50km_non_geo_p01_p99'),'n'].sum())!=302 or int(p.loc[(p.country=='CAN')&(p.profile=='with_met')&(p.rule=='distance_50km_non_geo_p01_p99'),'eligible_rows'].sum())!=725:
        raise RuntimeError('F3 expected CAN with-MET distance50+p01p99 support 302/725')
    p['row_coverage_fraction']=pd.to_numeric(p.row_coverage_fraction,errors='coerce')
    p=p.sort_values(['country','task','profile','rule']).reset_index(drop=True)
    p['coverage_label']=p.row_coverage_fraction.map(lambda z:f'{z:.1%}' if pd.notna(z) else 'NA')
    plabels=sorted(p.row_label.unique());pymap={v:i for i,v in enumerate(plabels)}
    rule_style={'distance_50km_non_geo_minmax':('o',BLUE),'distance_50km_non_geo_p01_p99':('s',ORANGE),'all':('x',GREY)}
    for rule,(marker,color) in rule_style.items():
        q=p[p.rule==rule]
        yy=np.array([pymap[v] for v in q.row_label]) + {'distance_50km_non_geo_minmax':-.18,'distance_50km_non_geo_p01_p99':0,'all':.18}[rule]
        axs[1].scatter(q.row_coverage_fraction,yy,marker=marker,s=15,c=color,label={'distance_50km_non_geo_minmax':'Distance 50 km + non-geographic min–max','distance_50km_non_geo_p01_p99':'Distance 50 km + non-geographic 1st–99th pct.','all':'All rows (reference)' }[rule],zorder=3)
    axs[1].set_yticks(range(len(plabels)),plabels,fontsize=5.1);axs[1].invert_yaxis();axs[1].set_xlim(-.03,1.04);axs[1].set_xlabel('Retained row-months / eligible row-months');axs[1].set_title('b  P90-month retained support by rule',loc='left',fontsize=7.5);axs[1].grid(axis='x',color=LIGHT,lw=.45);axs[1].legend(fontsize=5.0,loc='lower right')
    for axx,l in zip(axs,['a','b']):panel_label(axx,l)
    fig.suptitle('Fire-proxy strata differ in held-out error and support',y=1.015,fontsize=9);fig.tight_layout(rect=[0,0,1,.97])
    cap=('Figure4 | Under fixed, strict-QA HGB primary policies, held-out error and P90 fire-proxy support are shown for the United States and Canada. (a) Saved full-model RMSE for P90 fire-proxy and other months. (b) 2023 P90-month row coverage under the source literals distance_50km_non_geo_minmax and distance_50km_non_geo_p01_p99; the all rule is shown only as a reference. CAN with MET retains 466/725 row-months (64.3%) under min–max and 302/725 (41.7%) under 1st–99th percentile. Each point is a country, MET profile and source rule; these conditional descriptive strata are not independent replicates or measures of smoke mass.')
    save_pub('Figure_4_Fire_Support',fig,cap,{'a':x,'b':p},core_claim='P90 fire-proxy row support depends on the explicitly stated distance and range restrictions.',sources=register_catalog('F3','joint_future')+[{'alias':'future_risk_summary','path':RESULTS/'csv'/'future_risk_summary.csv','role':'primary fixed high-fire risk rows by literal support rule'}],
      filters={'panel_a':{'source':'joint_future','qa_variant':'strict','estimator_kind':'hgb','target_strategy':'log_smear','backtransform':'smearing','country':['USA','CAN'],'dataset_policy':{'USA':'usa_primary','CAN':'can_primary_all_published'},'validation_mode':'fixed','task':'T3_fixed_*','rule':'all','retained':True,'fire_month_scope':['P90_fire_proxy','other_months']},'panel_b':{'source':'future_risk_summary.csv','summary_class':'primary_strict_hgb_log_smear_smearing','task':'T3_fixed_2023','validation_mode':'fixed','country':['USA','CAN'],'profile':['with_met','no_met'],'fire_month_scope':'P90_fire_proxy','rule_literals':['distance_50km_non_geo_minmax','distance_50km_non_geo_p01_p99','all reference']}},
      replicate_meaning='Panel a points are saved country/year/MET-profile/month-scope results. Panel b points are exact primary source risk rows; row coverage is source n divided by eligible_rows. Rules and task strata are not pooled.',
      excluded=['Panel a retains the original all-rule held-out RMSE strata. Panel b excludes non-primary/non-fixed/non-P90 and retained-false records; the literal all rule is shown only as a reference. Full risk-summary source is supplied in source_data.'])

def calibration_rows():
    d=pd.read_csv(RESULTS/'csv'/'conformal_primary_fold_summary.csv',low_memory=False)
    required={'normalized_country_for_grouping','task','normalized_profile_for_grouping','fold_id','split_label','arm','normalized_qa_variant_for_grouping','normalized_estimator_for_grouping','q90_status','pointwise_coverage','test_row_n','all_month_station_coverage_recomputed_from_counts','all_month_station_covered_n_source','all_month_station_group_n_source','mean_interval_width','median_interval_width','unique_key','unique_key_record_count'}
    if not required.issubset(d.columns): raise RuntimeError(f'Calibration source missing fields: {sorted(required-set(d.columns))}')
    x=d[(d.normalized_qa_variant_for_grouping=='strict')&(d.normalized_estimator_for_grouping=='hgb')&
        (d.arm=='full')&(d.split_label=='primary_80_20')].copy()
    if len(x)!=64: raise RuntimeError(f'F4 expected 64 verified primary US/CAN fold rows, got {len(x)}')
    if not x.unique_key_record_count.eq(1).all() or x.unique_key.duplicated().any(): raise RuntimeError('F4 primary fold key is not unique')
    x['country']=x.normalized_country_for_grouping
    x['profile']=x.normalized_profile_for_grouping
    x['q90_inf']=x.q90_status.eq('infinite')
    x['all_month_per_station_coverage']=x.all_month_station_coverage_recomputed_from_counts
    x['all_month_per_station_covered_n']=x.all_month_station_covered_n_source
    x['all_month_per_station_group_n']=x.all_month_station_group_n_source
    x['row_label']=x.country+' '+x.task.map(TASK_LABELS).fillna(x.task)+' '+x.profile.str.replace('_',' ',regex=False)+' f'+x.fold_id.astype(str)
    return d,x

def plot_f4():
    full,x=calibration_rows(); task_order=TASKS
    x=x[x.country.isin(['USA','CAN'])&x.task.isin(task_order)].copy()
    if x.empty: raise RuntimeError('F4 selected primary source has no USA/CAN rows')
    key=['country','task','profile','fold_id','analysis_source'] if 'analysis_source' in x else ['country','task','profile','fold_id']
    if x.duplicated(key).any(): raise RuntimeError(f'F4 duplicate primary fold key {key}')
    x=x.sort_values(['country','task','profile','fold_id']).reset_index(drop=True)
    x['pointwise_covered_row_n_derived']=pd.to_numeric(x.pointwise_covered_row_n_derived,errors='coerce')
    x['test_row_n']=pd.to_numeric(x.test_row_n,errors='coerce')
    if not x.q90_status.eq('finite').all(): raise RuntimeError('F4 expected all 64 exact primary folds to have finite q90')
    key_group=['country','task','profile']
    rows=[]
    for key,g in x.groupby(key_group,sort=True):
        expected_fold_n=1 if key[1]=='T3_year_2022' else 5
        if len(g)!=expected_fold_n: raise RuntimeError(f'F4 expected {expected_fold_n} primary folds per country/task/profile group, got {key}: {len(g)}')
        point_n=g.pointwise_covered_row_n_derived.sum(min_count=1); point_den=g.test_row_n.sum(min_count=1)
        station_n=pd.to_numeric(g.all_month_station_covered_n_source,errors='coerce').sum(min_count=1)
        station_den=pd.to_numeric(g.all_month_station_group_n_source,errors='coerce').sum(min_count=1)
        w=pd.to_numeric(g.mean_interval_width,errors='coerce'); w_ok=w.notna()&~g.q90_inf
        width_den=g.loc[w_ok,'test_row_n'].sum(min_count=1)
        width=(w[w_ok]*g.loc[w_ok,'test_row_n']).sum(min_count=1)/width_den if pd.notna(width_den) and width_den>0 else np.nan
        rows.append(dict(country=key[0],task=key[1],profile=key[2],fold_n=len(g),
            pointwise_covered_rows=point_n,pointwise_test_rows=point_den,
            pointwise_coverage=point_n/point_den if pd.notna(point_n) and pd.notna(point_den) and point_den>0 else np.nan,
            all_month_station_covered_n=station_n,all_month_station_group_n=station_den,
            all_month_station_coverage=station_n/station_den if pd.notna(station_n) and pd.notna(station_den) and station_den>0 else np.nan,
            row_weighted_mean_interval_width=width,width_weight_rows=width_den,
            finite_width_folds=int(w_ok.sum()),unbounded_q90_folds=int(g.q90_inf.sum())))
    agg=pd.DataFrame(rows)
    if len(agg)!=16 or agg.duplicated(key_group).any(): raise RuntimeError(f'F4 expected 16 country/task/MET groups; got {len(agg)}')
    if int(agg.finite_width_folds.sum())!=64 or int(agg.unbounded_q90_folds.sum())!=0:
        raise RuntimeError('F4 width summary must retain all 64 finite-q90 primary folds')
    agg['row_label']=agg.country+' · '+agg.task.map(TASK_LABELS).fillna(agg.task)+' · '+agg.profile.str.replace('_',' ',regex=False)
    agg=agg.sort_values(['country','task','profile']).reset_index(drop=True)
    fig,axs=plt.subplots(1,3,figsize=(7.2,5.8),sharey=True,gridspec_kw={'width_ratios':[1,1,1.15]})
    y=np.arange(len(agg));colors=agg.country.map({'USA':BLUE,'CAN':ORANGE});markers=agg.profile.map({'with_met':'o','no_met':'s'})
    for ax,col,lab in zip(axs,['pointwise_coverage','all_month_station_coverage','row_weighted_mean_interval_width'],
                          ['Point coverage','Whole-station coverage','Row-weighted interval width (µg m⁻³)']):
        vals=pd.to_numeric(agg[col],errors='coerce');finite=vals.notna()
        for pr,mark in [('with_met','o'),('no_met','s')]:
            m=(agg.profile==pr).to_numpy()&finite.to_numpy()
            ax.scatter(vals[m],y[m],c=colors[m],marker=mark,s=19,label=pr.replace('_',' '),edgecolors='white',linewidths=.35,zorder=3)
        if col!='row_weighted_mean_interval_width':
            ax.axvline(.9,color=GREY,lw=.6,ls=':');ax.set_xlim(-.03,1.03)
        else:
            for j,r in agg.iterrows():
                if r.unbounded_q90_folds:
                    xpos=(vals[finite].max()*1.05) if finite.any() else 1
                    ax.scatter([xpos],[j],marker='x',c=RED,s=18,zorder=4)
                    ax.annotate(f'{int(r.unbounded_q90_folds)} ∞', (xpos,j), xytext=(2,1),textcoords='offset points',fontsize=5.0,color=RED,va='center')
            ax.set_xlim(left=0)
        ax.grid(axis='x',color=LIGHT,lw=.45);ax.set_xlabel(lab);ax.set_title(lab,loc='left',fontsize=6.8)
    axs[0].set_yticks(y,agg.row_label,fontsize=5.2);axs[0].invert_yaxis();axs[0].set_ylabel('Country · task · profile')
    axs[0].legend(fontsize=5.0,loc='lower right',title='MET profile',title_fontsize=5)
    panel_label(axs[0],'a');panel_label(axs[1],'b');panel_label(axs[2],'c')
    fig.suptitle('Primary conformal performance by country, task and MET profile',y=1.01,fontsize=8.2)
    fig.text(.5,.005,'Sixteen groups retain source fold counts (five for T1/T2 tasks; one for T3 2022). All 64 q90 thresholds are finite; width is test-row weighted over all source folds.',ha='center',fontsize=5.5)
    fig.tight_layout(rect=[0,.02,1,.98])
    cap=('Figure5 | Primary conformal calibration summarized by country, task and MET profile in the United States and Canada. (a) Point coverage is the sum of covered test rows divided by the sum of test rows. (b) Whole-station coverage is the sum of source covered stations divided by the sum of source station groups. (c) Mean interval width is weighted by test-row count across the source folds. All 64 selected primary folds have finite q90. The 16 country–task–profile groups retain the source fold counts: five folds for each T1/T2 task and one for T3 2022. These descriptive summaries are not independent replicates. Feasibility-only rows remain separate in Supplementary Figure S2.')
    save_pub('Figure_5_Calibration',fig,cap,{
      'a':agg[['country','task','profile','fold_n','pointwise_covered_rows','pointwise_test_rows','pointwise_coverage']],
      'b':agg[['country','task','profile','fold_n','all_month_station_covered_n','all_month_station_group_n','all_month_station_coverage']],
      'c':agg[['country','task','profile','fold_n','row_weighted_mean_interval_width','width_weight_rows','finite_width_folds','unbounded_q90_folds']]},
      core_claim='Source-fold pooling yields compact, denominator-based calibration and width summaries across 16 primary country-task-profile groups.',
      sources=[{'alias':'conformal_primary_fold_summary','path':RESULTS/'csv'/'conformal_primary_fold_summary.csv','role':'QA-accepted unique exact-primary US/CAN fold summary; source row bindings retained'},
               {'alias':'results_qa','path':RESULTS/'qa.json','role':'seven-check acceptance record for the verified primary fold summary'}]+register_input('F4','Supplementary_Figure_S2_Panel_a_Source_Data.csv'),
      filters={'source':'QA-accepted exact primary fold summary derived from the formal S3a source','qa_variant':'strict','estimator_kind':'hgb','arm':'full','split_label':'primary_80_20','country':['USA','CAN'],'task_fold_counts':{'T1_station':5,'T2_spatial':5,'T2_spatial_buffer50km':5,'T3_year_2022':1},'country_task_profile_groups':16,'unique_keys':64},
      replicate_meaning='Each main-panel point pools the original source folds within country/task/profile using summed source numerator and denominator; interval width weights finite-q90 fold means by test-row count. Fold counts vary by task (5 for T1/T2; 1 for T3 2022); folds/tasks are not treated as independent replicates.',
      excluded=['Feasibility-only split rows and non-US/Canadian countries are outside this primary summary. The complete exact-primary fold source remains in source_data; feasibility folds remain separate.'])

def plot_f5():
    m=load('multiplicity')
    if len(m)!=78 or m.duplicated(['country','task','profile','contrast']).any(): raise RuntimeError('F5 multiplicity source cardinality/key failed')
    if not (m.qa_variant.eq('strict').all() and m.estimator_kind.eq('hgb').all() and m.target_strategy.eq('log_smear').all()): raise RuntimeError('F5 locked multiplicity scope mismatch')
    expected=['B+AOD - B','full - B+AOD','full - B']; counts={}
    for c in expected:
        q=m[m.contrast==c]
        if len(q)!=26: raise RuntimeError(f'F5 expected 26 rows for {c}; got {len(q)}')
        counts[c]=int(((q.simultaneous_95_low>0)|(q.simultaneous_95_high<0)).sum())
    if list(counts.values())!=[24,12,26]: raise RuntimeError(f'F5 CI-excludes-zero counts differ from contract: {counts}')
    fig,axs=plt.subplots(3,1,figsize=(7.2,9.0),sharex=True); panels={}
    for i,contrast in enumerate(expected):
        ax=axs[i];q=m[m.contrast==contrast].copy();q['row_label']=q.country+' · '+q.task+' · '+q.profile
        q=q.sort_values(['country','task','profile']).reset_index(drop=True);y=np.arange(len(q));colors=q.profile.map({'with_met':BLUE,'no_met':ORANGE})
        ax.axvline(0,color='#222222',lw=.7)
        for j,r in q.iterrows():
            ax.plot([r.simultaneous_95_low,r.simultaneous_95_high],[j,j],color=colors.iloc[j],lw=1)
            ax.plot(r.delta_rmse,j,'o',color=colors.iloc[j],ms=2.8)
        ax.set_yticks(y,q.row_label,fontsize=5.2);ax.invert_yaxis();ax.grid(axis='x',color=LIGHT,lw=.45)
        ax.set_title(f'{chr(97+i)}  {contrast} · simultaneous 95% CI excludes zero: {counts[contrast]}/26',loc='left',fontsize=7.2)
        ax.set_ylabel('Country · task · profile',fontsize=6);panels[chr(97+i)]=q
    axs[-1].set_xlabel('Δ RMSE (µg m⁻³)')
    fig.legend(handles=[plt.Line2D([0],[0],marker='o',color=BLUE,label='with MET',lw=0),plt.Line2D([0],[0],marker='o',color=ORANGE,label='no MET',lw=0)],loc='upper center',ncol=2,bbox_to_anchor=(.5,1.01),fontsize=6)
    fig.suptitle('Feature increments across the locked 26-unit family',y=1.025,fontsize=9);fig.text(.5,.006,'Fixed max-|t| family; conditional on fitted predictions, observed folds and years.',ha='center',fontsize=6)
    fig.tight_layout(rect=[0,.025,1,.97])
    cap=('Figure3 | RMSE feature increments across the locked 26-unit family. Points are saved contrasts and bars are simultaneous 95% max-|t| intervals across all 78 rows (three contrasts × 26 units), using the prespecified 10,000 Exp(1) station-multiplier draws. Intervals exclude zero for 24/26, 12/26 and 26/26 units in the displayed contrast order. The family is conditional/exploratory; no new bootstrap was run.')
    save_pub('Figure_3_Feature_Increments',fig,cap,panels,core_claim='Saved simultaneous intervals distinguish feature increments from the full-model total increment.',sources=register_catalog('F5','multiplicity'),
      filters={'expected_rows':78,'contrasts':expected,'qa_variant':'strict','estimator_kind':'hgb','target_strategy':'log_smear','interval_columns':['simultaneous_95_low','simultaneous_95_high']},
      replicate_meaning='Each row is one country-task-MET-profile unit. Conditional fixed max-|t| simultaneous 95% intervals from 10,000 Exp(1) station-multiplier draws; no new bootstrap was run.')

def plot_s1():
    h=load('historical_metrics')
    h=h[(h.arm=='full')&(h.task=='T1_station')&((~h.country.isin(['USA','CAN']))|h.dataset_key.isin(['usa_hist','can_hist_primary']))].copy()
    qa=h[(h.target_strategy=='log_smear')&(h.backtransform=='smearing')].copy()
    trans=h[(h.qa_variant=='strict')&(h.estimator_kind=='hgb')].copy()
    cs=sorted(set(qa.country)|set(trans.country)); y={c:i for i,c in enumerate(cs)}
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.4),sharey=True)
    qas=[('strict','hgb',BLUE,'strict HGB'),('strict','ridge',GREEN,'strict Ridge'),('sensitivity','hgb',ORANGE,'sensitivity HGB'),('sensitivity','ridge',PURPLE,'sensitivity Ridge')]
    for ci,profile in enumerate(['no_met','with_met']):
        ax=axs[0,ci]
        for j,(qa0,est,col,label) in enumerate(qas):
            q=qa[(qa.qa_variant==qa0)&(qa.estimator_kind==est)&(qa.profile==profile)]
            ax.scatter(q.r2,q.country.map(y)+(j-1.5)*.08,s=10,c=col,marker='o',label=label,alpha=.85)
        ax.set_xscale('symlog',linthresh=.1);ax.axvline(0,color=GREY,lw=.5);ax.grid(axis='x',color=LIGHT,lw=.4)
        ax.set_title(f'{chr(97+ci)}  QA/estimator · {profile.replace("_"," ")}',loc='left',fontsize=7.2);ax.set_xlabel('T1 R² (symlog)')
        ax.legend(fontsize=5.2,ncol=2,loc='lower right')
        ax=axs[1,ci]; q=trans[trans.profile==profile].copy()
        targets=sorted(q.target_strategy.dropna().unique()); colors=[BLUE,ORANGE,GREEN,RED,PURPLE,'#8B7355','#4C8796','#B07C8D']
        markers={'smearing':'o','original_scale':'s','none':'^'}; color_map={t:colors[i%len(colors)] for i,t in enumerate(targets)}
        for target in targets:
            for bt in sorted(q.loc[q.target_strategy==target,'backtransform'].dropna().unique()):
                z=q[(q.target_strategy==target)&(q.backtransform==bt)]
                ax.scatter(z.r2,z.country.map(y),s=10,c=color_map[target],marker=markers.get(bt,'D'),label=f'{target}/{bt}',alpha=.8)
        ax.set_xscale('symlog',linthresh=.1);ax.axvline(0,color=GREY,lw=.5);ax.grid(axis='x',color=LIGHT,lw=.4)
        ax.set_title(f'{chr(99+ci)}  Target/retransformation · {profile.replace("_"," ")}',loc='left',fontsize=7.2);ax.set_xlabel('T1 R² (symlog)')
        ax.legend(fontsize=5.0,ncol=2,loc='lower right')
        ax.set_yticks(range(len(cs)),cs,fontsize=5.2)
    axs[0,0].set_ylabel('Country (PT = Portugal)');axs[1,0].set_ylabel('Country (PT = Portugal)')
    fig.suptitle('Historical sensitivity branches retain MET profile and adverse R²',y=1.01,fontsize=8.8);fig.tight_layout(rect=[0,0,1,.97])
    cap=('Supplementary Figure S3 | Historical T1 station-grouped sensitivity. The upper panels show QA and estimator branches under log-smear/smearing; the lower panels show strict-HGB target/retransformation branches. MET profiles are separated by column, so no-MET and with-MET records are not merged. Each point is a saved country-level estimate; the signed-log x-axis preserves adverse negative R². PT denotes Portugal. No confidence interval or new inference is added.')
    save_pub('Supplementary_Figure_S3',fig,cap,{'a':qa,'b':trans},core_claim='Sensitivity to QA, estimator and target/retransformation is visible without conflating MET profiles.',sources=register_catalog('S1','historical_metrics'),
       filters={'task':'T1_station','arm':'full','country_policy':'primary USA/CAN target keys and catalogued domestic country rows','upper':'target log_smear + smearing; QA×estimator','lower':'strict HGB; all recorded target/backtransform branches','profiles':'faceted no_met/with_met'},
       replicate_meaning='One saved country/profile/branch estimate; no inferential interval; branches are sensitivity analyses, not independent replications.')

def plot_s2():
    loco=load('loco_metrics'); s2a=pd.read_csv(INPUTS/'Supplementary_Figure_S4_Panel_a_Source_Data.csv',low_memory=False)
    thin=pd.read_csv(INPUTS/'Supplementary_Figure_S4_Panel_b_Source_Data.csv',low_memory=False)
    # The source panel is the accepted full LOCO table; make target-country coverage explicit.
    if len(s2a)!=136: raise RuntimeError(f'S2 formal LOCO source expected 136 saved rows, got {len(s2a)}')
    q=s2a[(s2a.scenario=='noMET')&(s2a.arm=='full')&(s2a.qa_variant=='strict')].copy()
    if not {'hgb','ridge'}.issubset(set(q.estimator_kind)) or len(set(q.country))!=17: raise RuntimeError('S2 LOCO does not retain all 17 countries and both estimators')
    if q.duplicated(['country','estimator_kind']).any(): raise RuntimeError('S2 duplicate LOCO country/estimator key')
    thin=thin[(thin.arm=='full')&thin.country.isin(['USA','CAN'])].copy()
    thin['train_n']=pd.to_numeric(thin.sample_size,errors='coerce').fillna(pd.to_numeric(thin.n_train_stations,errors='coerce'))
    if thin.duplicated(['country','profile','seed','sample_size']).any(): raise RuntimeError('S2 duplicate thinning country/profile/seed/sample-size key')
    fig,axs=plt.subplots(1,3,figsize=(7.2,6.0),gridspec_kw={'width_ratios':[1.4,1,1]})
    countries=sorted(q.country.unique());ymap={c:i for i,c in enumerate(countries)}
    for est,col,mark in [('hgb',BLUE,'o'),('ridge',ORANGE,'s')]:
        g=q[q.estimator_kind==est];axs[0].scatter(g.r2,g.country.map(ymap),s=11,c=col,marker=mark,label=est.upper())
    axs[0].set_yticks(range(len(countries)),countries);axs[0].set_xscale('symlog',linthresh=.1);axs[0].axvline(0,color=GREY,lw=.5);axs[0].set_xlabel('LOCO R² (symlog)');axs[0].set_title('a  Strict no-MET transfer',loc='left',fontsize=7.4);axs[0].legend(fontsize=5.5);axs[0].grid(axis='x',color=LIGHT,lw=.4)
    for ax,country,lab in zip(axs[1:],['USA','CAN'],['b','c']):
        z=thin[thin.country==country]
        for profile,color in [('with_met',BLUE),('no_met',ORANGE)]:
            for seed,ls in sorted(zip(z.seed.unique(),['-','--',':'])):
                g=z[(z.profile==profile)&(z.seed==seed)].sort_values('train_n')
                if len(g): ax.plot(g.train_n,g.rmse,marker='o',ms=2.2,lw=.7,ls=ls,color=color,label=f'{profile.replace("_"," ")} · seed {seed}')
            ax.set_xscale('log');ax.set_xlabel('Training stations');ax.set_ylabel('2022 RMSE (µg m⁻³)');ax.set_title(f'{lab}  {country} network thinning',loc='left',fontsize=7.4);ax.grid(axis='y',color=LIGHT,lw=.4);ax.legend(fontsize=5.0,ncol=2)
    fig.suptitle('Transfer and network-thinning sensitivity',y=1.01,fontsize=9);fig.tight_layout(rect=[0,0,1,.98])
    cap=('Supplementary Figure S4 | Transfer and monitoring-network sensitivity. (a) Strict-QA no-MET leave-one-country-out R² for all 17 held-out countries under HGB and Ridge, full feature arm. (b,c) Full-arm 2022 RMSE across nested training-site counts for the three prespecified station-thinning seeds, by MET profile. The shared full-network endpoint is not an independent replicate; lines are descriptive and not confidence intervals.')
    sources=register_catalog('S2','loco_metrics')+register_input('S2','Supplementary_Figure_S4_Panel_a_Source_Data.csv','Supplementary_Figure_S4_Panel_b_Source_Data.csv')
    # Hash-pin the thinning source against its QA acceptance record before delivery.
    thin_hashes=verify_thinning_sources()
    sources += [{'alias':f'thinning_{name}','path':THIN_RUN/name,'role':'hash-pinned accepted thinning output'} for name in thin_hashes]
    save_pub('Supplementary_Figure_S4',fig,cap,{'a':q,'b':thin},core_claim='Geographic transfer and training-network size change performance; fixed seeds are sensitivity replicates.',sources=sources,
      filters={'LOCO':'noMET, arm full, strict QA, HGB/Ridge, all 17 countries','thinning':'USA/CAN, full arm, all three fixed seeds, all sample sizes'},
      replicate_meaning='LOCO destinations are validation targets; thinning seed curves are descriptive split sensitivities, and full-network endpoints are shared.',stats='No new inference; station thinning seeds are fixed sensitivity replicates, not bootstrap confidence intervals.')

def plot_s3():
    formal=pd.read_csv(INPUTS/'Supplementary_Figure_S2_Panel_a_Source_Data.csv',low_memory=False)
    mapping=pd.read_csv(RESULTS/'csv'/'conformal_analysis_source_mapping.csv',low_memory=False)
    primary=pd.read_csv(RESULTS/'csv'/'conformal_primary_fold_summary.csv',low_memory=False)
    required={'normalized_country_for_grouping','task','normalized_profile_for_grouping','fold_id','split_label','arm','normalized_qa_variant_for_grouping','normalized_estimator_for_grouping','q90_status','pointwise_coverage','all_month_station_coverage_recomputed_from_counts','unique_key','unique_key_record_count'}
    if not required.issubset(primary.columns): raise RuntimeError(f'S3 mapped primary source missing fields: {sorted(required-set(primary.columns))}')
    primary=primary[(primary.normalized_qa_variant_for_grouping=='strict')&(primary.normalized_estimator_for_grouping=='hgb')&(primary.arm=='full')&(primary.split_label=='primary_80_20')].copy()
    primary['country']=primary.normalized_country_for_grouping;primary['profile']=primary.normalized_profile_for_grouping
    primary['q90_is_infinite']=primary.q90_status.eq('infinite')
    primary['all_month_per_station_coverage']=primary.all_month_station_coverage_recomputed_from_counts
    if len(formal)!=440 or len(mapping)!=4 or len(primary)!=64: raise RuntimeError(f'S3 source/map/primary counts changed: {len(formal)}/{len(mapping)}/{len(primary)}')
    # Use the four catalogued feasibility files as a separate Brazil external-2023 facet.
    feas_names=[f for f in ['bra2023__0c1a000366a5__BRA_sensitivity_ridge_external_2023__conformal_feasibility_summary',
       'bra2023__27a62e38a97d__BRA_strict_hgb_external_2023__conformal_feasibility_summary',
       'bra2023__7e7cb5461204__BRA_sensitivity_hgb_external_2023__conformal_feasibility_summary',
       'bra2023__4101b95b63c0__BRA_strict_ridge_external_2023__conformal_feasibility_summary']]
    feas=pd.concat([load(n) for n in feas_names],ignore_index=True)
    if len(primary)!=64 or len(feas)!=8: raise RuntimeError(f'S3 mapped-primary/feasible row counts changed: {len(primary)}/{len(feas)}')
    if primary.duplicated(['country','task','profile','fold_id','analysis_source']).any(): raise RuntimeError('S3 duplicate primary country/task/profile/fold/source key')
    if set(primary.country.unique())!={'USA','CAN'} or any(len(g)!=32 for _,g in primary.groupby('country')): raise RuntimeError('S3 expected 32 exact primary folds in each USA/CAN facet')
    # A4-width page: 32 fold rows per country, with pointwise and whole-station
    # coverage adjacent within each facet. Brazil feasibility stays separate.
    primary['row_label']=primary.task.map(TASK_LABELS).fillna(primary.task)+' · '+primary.profile.str.replace('_',' ',regex=False)+' · f'+primary.fold_id.astype(str)
    fig=plt.figure(figsize=(5.90,7.75));gs=fig.add_gridspec(2,2,height_ratios=[5.7,1.7],hspace=.33,wspace=.28)
    axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])];feax=fig.add_subplot(gs[1,:])
    for ax,country,lab in zip(axs,['USA','CAN'],['United States · 32 primary folds','Canada · 32 primary folds']):
        g=primary[primary.country==country].sort_values(['task','profile','fold_id']).reset_index(drop=True);y=np.arange(len(g))
        for metric,mark,off,metric_label in [('pointwise_coverage','o',-.10,'Pointwise'),('all_month_per_station_coverage','D',.10,'Whole-station')]:
            vals=pd.to_numeric(g[metric],errors='coerce');inf=bool_series(g.q90_is_infinite).to_numpy()
            edge=np.where(inf,RED,'white')
            ax.scatter(vals,y+off,c=g.profile.map({'with_met':BLUE,'no_met':ORANGE}),marker=mark,s=10,edgecolors=edge,linewidths=.35,alpha=.88,label=metric_label)
        ax.axvline(.9,color=GREY,lw=.6,ls=':');ax.set_xlim(-.03,1.03);ax.set_yticks(y,g.row_label if country=='USA' else ['']*len(g),fontsize=5.0);ax.invert_yaxis();ax.set_xlabel('Coverage fraction');ax.set_title(lab,loc='left',fontsize=6.7);ax.grid(axis='x',color=LIGHT,lw=.4)
        ax.legend(fontsize=5.0,loc='lower left',ncol=1)
    axs[0].set_ylabel('Task · MET profile · fold')
    task=feas.task.astype(str).str.replace('T4_external_2023_','T4 2023 ',regex=False)
    feas=feas.assign(row_label=feas.qa_variant+' '+feas.estimator_kind.str.upper()+' '+feas.profile.str.replace('_',' ',regex=False))
    yy=np.arange(len(feas))
    for qa0,col in [('strict',BLUE),('sensitivity',ORANGE)]:
        for est,mark in [('hgb','o'),('ridge','s')]:
            m=(feas.qa_variant==qa0)&(feas.estimator_kind==est)
            feax.scatter(feas.loc[m,'q90'],yy[m],c=col,marker=mark,s=17,label=f'{qa0} {est.upper()}')
    feax.set_yticks(yy,feas.row_label,fontsize=5.0);feax.invert_yaxis();feax.set_xlabel('q90 feasibility threshold (source units)');feax.set_title('Brazil external-2023 feasibility only · 8 records',loc='left',fontsize=6.7);feax.grid(axis='x',color=LIGHT,lw=.4)
    for ax,l in zip(axs+[feax],['a','b','c']):panel_label(ax,l)
    fig.suptitle('Fold-level calibration detail by country',y=.995,fontsize=8.2);fig.text(.5,.005,'Dotted lines mark nominal 90% coverage. Brazil feasibility thresholds are a separate split and are not pooled with primary coverage.',ha='center',fontsize=5.3)
    cap=('Supplementary Figure S2 | Fold-level calibration detail, grouped within country. (a) The 32 exact, explicitly mapped strict-HGB full-arm primary 80/20 United States folds show pointwise and whole-station coverage by fold. (b) The 32 corresponding Canadian folds. The dotted lines mark nominal 90% coverage. Blue and orange denote with-MET and no-MET profiles; circles and diamonds denote pointwise and whole-station coverage. (c) Eight Brazil external-2023 feasibility-only thresholds from separate strict/sensitivity HGB/Ridge sources; row labels identify the source classes. Primary empirical coverage and feasibility thresholds are not pooled. The complete 440-row formal source is supplied as a CSV; unmapped rows remain unassigned and unplotted.')
    sources=[{'alias':'conformal_primary_fold_summary','path':RESULTS/'csv'/'conformal_primary_fold_summary.csv','role':'QA-accepted 64 exact-primary fold rows'},
      {'alias':'conformal_analysis_source_mapping','path':RESULTS/'csv'/'conformal_analysis_source_mapping.csv','role':'explicit four-source mapping table'},
      {'alias':'results_qa','path':RESULTS/'qa.json','role':'seven-check acceptance record'}]+register_input('S3','Supplementary_Figure_S2_Panel_a_Source_Data.csv')+register_catalog('S3',*feas_names)
    save_pub('Supplementary_Figure_S2',fig,cap,{'a':primary[primary.country=='USA'],'b':primary[primary.country=='CAN'],'c':feas},core_claim='Fold-level calibration remains country-specific while feasibility-only records remain a separate source class.',sources=sources,
      filters={'panel_a':'strict HGB full arm; primary_80_20; all 64 explicitly mapped USA/CAN country/task/profile/fold rows; 440-row formal input supplied unchanged','panel_b':'all four catalogued Brazil external-2023 conformal feasibility sources; split is feasibility-only'},
      replicate_meaning='Fold/source records are not independent trial replicates; no coverage or width is averaged across folds or source generations.',
      excluded=['Non-strict/non-HGB primary sensitivity branches are not plotted in panel (a); complete formal S3a source is delivered. Brazil feasibility rows are in panel (b) only and are not mixed into panel (a).'])

def plot_s4():
    a=pd.read_csv(INPUTS/'Supplementary_Figure_S8_Panel_a_Source_Data.csv',low_memory=False)
    b=pd.read_csv(INPUTS/'Supplementary_Figure_S8_Panel_b_Source_Data.csv',low_memory=False)
    # Panel a: saved fitted B+CAMS minus full monthly future comparison.
    aa=a[a.contrast=='B+CAMS minus full'].copy();
    # Panel b: prespecified primary exact 12-month annual support only; sensitivity remains in full source CSV.
    bb=b[(b.support_scope=='primary_exact_12_months')&(b.status=='eligible')].copy()
    if len(a)!=28 or len(b)!=176: raise RuntimeError(f'S4 formal input counts changed: {len(a)}/{len(b)}')
    if aa.empty or bb.empty: raise RuntimeError('S4 display branch missing')
    fig,axs=plt.subplots(1,2,figsize=(7.2,7.7),gridspec_kw={'width_ratios':[1,1.55]})
    q=aa.copy();q['label']=q.country+' '+q.task.str.replace('T4_external_','T4 ',regex=False)+' '+q.profile.str.replace('_',' ',regex=False);q=q.sort_values(['country','task','profile']);y=np.arange(len(q))
    for j,r in enumerate(q.itertuples()):
        if pd.notna(r.delta_rmse_ci_low) and pd.notna(r.delta_rmse_ci_high): axs[0].plot([r.delta_rmse_ci_low,r.delta_rmse_ci_high],[j,j],color=BLUE,lw=.7)
        axs[0].scatter(r.delta_rmse,j,c=BLUE if r.profile=='with_met' else ORANGE,s=11)
    axs[0].axvline(0,color=GREY,lw=.6);axs[0].set_yticks(y,q.label,fontsize=5.0);axs[0].invert_yaxis();axs[0].set_xlabel('ΔRMSE: B+CAMS − full (µg m⁻³)');axs[0].set_title('a  Fitted monthly future comparator',loc='left',fontsize=7.2);axs[0].grid(axis='x',color=LIGHT,lw=.4)
    q=bb.copy();q['label']=q.country+' '+q.task.str.replace('T1_station','T1').str.replace('T2_spatial_buffer50km','T2 buffer').str.replace('T2_spatial','T2').str.replace('T3_year_2022','T3 2022').str.replace('T4_external_','T4 ')+' '+q.profile.str.replace('_',' ',regex=False)+' · '+q.benchmark_version
    q=q.sort_values(['country','task','profile','benchmark_version']);y=np.arange(len(q))
    for j,r in enumerate(q.itertuples()):
        if pd.notna(r.delta_rmse_ci_low) and pd.notna(r.delta_rmse_ci_high): axs[1].plot([r.delta_rmse_ci_low,r.delta_rmse_ci_high],[j,j],color=BLUE if r.profile=='with_met' else ORANGE,lw=.65)
        axs[1].scatter(r.delta_rmse,j,c=BLUE if r.profile=='with_met' else ORANGE,s=8)
    axs[1].axvline(0,color=GREY,lw=.6);axs[1].set_yticks(y,q.label,fontsize=5.0);axs[1].invert_yaxis();axs[1].set_xlabel('ΔRMSE in named contrast (µg m⁻³)');axs[1].set_title('b  Annual raw-product comparison · exact 12 months',loc='left',fontsize=7.2);axs[1].grid(axis='x',color=LIGHT,lw=.4)
    panel_label(axs[0],'a');panel_label(axs[1],'b');fig.suptitle('Fitted monthly and annual raw-product comparisons are distinct',y=1.005,fontsize=8.7);fig.text(.5,.003,'For a contrast “A minus B”, positive values mean higher RMSE for A; annual support is not monthly.',ha='center',fontsize=5.8);fig.tight_layout(rect=[0,0,1,.985])
    cap=('Supplementary Figure S8 | Fitted model comparison and annual raw-product benchmarks are shown separately. (a) Saved fixed/rolling future evaluation contrast B+CAMS minus full, with conditional station-resampling intervals. (b) Eligible annual raw-comparator contrasts on the prespecified exact-12-month support; labels retain country, task, MET profile and benchmark. In both panels the signed difference is exactly as named: for “A minus B,” positive values mean higher RMSE for A and negative values mean lower RMSE for A. The annual panel is not a monthly satellite comparison; SatPM calibration overlap prevents treating it as independent truth. Sensitivity support rows remain in the full source table.')
    save_pub('Supplementary_Figure_S8',fig,cap,{'a':aa,'b':bb},core_claim='Fitted monthly and annual raw-product contrasts answer different comparison questions.',sources=register_input('S4','Supplementary_Figure_S8_Panel_a_Source_Data.csv','Supplementary_Figure_S8_Panel_b_Source_Data.csv')+register_catalog('S4','annual_deltas'),
      filters={'panel_a':'all 28 B+CAMS minus full formal future rows','panel_b':'eligible rows; primary_exact_12_months support; full 176-row source retains sensitivity_at_least_10_months and ineligible records'},
      replicate_meaning='Conditional station-resampling intervals from saved files; annual comparison years/supports are not independent monthly samples.',
      excluded=['Annual sensitivity-support and ineligible rows are not plotted but are included in the complete 176-row panel source table.'])

def plot_s5():
    a=pd.read_csv(INPUTS/'Figure_3_Fire_Consistency_Panel_a_Source_Data.csv',low_memory=False)
    b=pd.read_csv(INPUTS/'Figure_3_Fire_Consistency_Panel_b_Source_Data.csv',low_memory=False)
    c=pd.read_csv(INPUTS/'Figure_3_Fire_Consistency_Panel_c_Source_Data.csv',low_memory=False)
    d=pd.read_csv(INPUTS/'Figure_3_Fire_Consistency_Panel_d_Source_Data.csv',low_memory=False)
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.3))
    # HMS consistency by country/profile/year.
    q=a.sort_values(['country','profile','year']); labels=[]
    for prof,col in [('with_met',BLUE),('no_met',ORANGE)]:
        z=q[q.profile==prof]; axs[0,0].errorbar(z.year,z.pearson_r,yerr=[z.pearson_r-z.ci_low,z.ci_high-z.pearson_r],fmt='o-',ms=2.4,lw=.7,color=col,label=prof.replace('_',' '))
    axs[0,0].set_xlabel('Year');axs[0,0].set_ylabel('Pearson r');axs[0,0].set_title('a  PM component vs HMS available-calendar fraction',loc='left',fontsize=7);axs[0,0].legend(fontsize=5)
    # QFED/GFED and OC comparator correlations.
    for ax,z,title,label,idx in [(axs[0,1],b,'2023 fire-product comparisons','b',1),(axs[1,0],c,'EPA OC-tracer correlations','c',2)]:
        if 'scope' in z:
            z=z.copy(); parts=z.scope.astype(str).str.split(':');z['country_label']=parts.str[1].fillna('')
            if 'comparator' in z:z['row_label']=z.country_label+' · '+z.comparator+' · '+z.profile
            else:z['row_label']=z.country_label+' · '+parts.str[-2]+'/'+parts.str[-1]+' · '+z.profile
            x='pearson_r';lo='pearson_r_ci_low';hi='pearson_r_ci_high'
        else: continue
        z=z.sort_values('row_label');yy=np.arange(len(z))
        for j,r in enumerate(z.itertuples()):
            lo_v=getattr(r,lo);hi_v=getattr(r,hi);val=getattr(r,x)
            if pd.notna(lo_v) and pd.notna(hi_v): ax.plot([lo_v,hi_v],[j,j],color=BLUE if getattr(r,'profile','')=='with_met' else ORANGE,lw=.65)
            ax.scatter(val,j,c=BLUE if getattr(r,'profile','')=='with_met' else ORANGE,s=9)
        ax.set_yticks(yy,z.row_label,fontsize=5.0);ax.invert_yaxis();ax.axvline(0,color=GREY,lw=.5);ax.set_xlabel('Pearson r');ax.set_title(f'{label}  {title}',loc='left',fontsize=7);ax.grid(axis='x',color=LIGHT,lw=.4)
    q=d.sort_values(['parameter_code','method_code','alignment','plume_definition']).copy();q['row_label']=q.parameter_code.astype(str)+'/'+q.method_code.astype(str)+' · '+q.alignment.astype(str).str.replace('_',' ')+' · '+q.plume_definition.astype(str).str.replace('_',' '); yy=np.arange(len(q))
    for j,r in enumerate(q.itertuples()):
        axs[1,1].plot([r.ci_low,r.ci_high],[j,j],color=RED,lw=.7);axs[1,1].scatter(r.mean_station_oc_difference,j,c=RED,s=9)
    axs[1,1].set_yticks(yy,q.row_label,fontsize=5.0);axs[1,1].invert_yaxis();axs[1,1].axvline(0,color=GREY,lw=.5);axs[1,1].set_xlabel('Plume − control OC (µg m⁻³)');axs[1,1].set_title('d  Observed OC within-month contrasts',loc='left',fontsize=7);axs[1,1].grid(axis='x',color=LIGHT,lw=.4)
    for ax,l in zip(axs.ravel(),['a','b','c','d']):panel_label(ax,l)
    fig.suptitle('Fire-related agreement is complementary, nonspecific evidence',y=1.01,fontsize=8.7);fig.text(.5,.004,'Correlations and OC contrasts do not identify smoke mass; shared supports are not independent evidence.',ha='center',fontsize=5.8);fig.tight_layout(rect=[0,0,1,.98])
    cap=('Supplementary Figure S5 | Complementary fire-related consistency evidence. (a) Saved PM-component/HMS available-calendar plume-fraction correlations by country, MET profile and year. (b) Saved 2023 GFED/QFED comparator correlations. (c) EPA OC-tracer correlation strata by country, parameter and method. (d) Station-level observed OC plume-minus-control contrasts by parameter/method, date alignment and plume definition. Points and intervals are saved estimates; OC is nonspecific, HMS dates are UTC while OC dates are local, and shared supports are not independent replication. These comparisons do not identify ground-level smoke mass.')
    raw_sources=register_catalog('S5','hms_correlations','smoke_future_comparators','oc_observed')
    raw_sources+=register_input('S5','Figure_3_Fire_Consistency_Panel_a_Source_Data.csv','Figure_3_Fire_Consistency_Panel_b_Source_Data.csv','Figure_3_Fire_Consistency_Panel_c_Source_Data.csv','Figure_3_Fire_Consistency_Panel_d_Source_Data.csv')
    save_pub('Supplementary_Figure_S5',fig,cap,{'a':a,'b':b,'c':c,'d':d},core_claim='Fire, satellite-product, and OC agreement supplies complementary consistency evidence, not a ground-level smoke-mass validation.',sources=raw_sources,
      filters={'panel_a':'strict P90 primary any-plume available-calendar correlations, same named date','panels_b_c_d':'formal saved 2023 fire/OC comparator and OC contrast rows, unchanged'},
      replicate_meaning='Saved station-bootstrap intervals condition on source support and fitted predictions; shared model/support hashes are not independent evidence.')

def plot_s6():
    h=load('hms_correlations'); v=load('hms_coverage')
    q=h[(h.qa_variant=='strict')&(h.target_strategy=='log_smear')&(h.backtransform=='smearing')&(h.HMS_indicator=='plume_fraction_available_calendar')].copy()
    z=v[(v.qa_variant=='strict')&(v.target_strategy=='log_smear')&(v.backtransform=='smearing')].copy()
    den=pd.to_numeric(z.known_days_matched,errors='coerce')+pd.to_numeric(z.unknown_days_matched,errors='coerce')
    z['known_fraction']=pd.to_numeric(z.known_days_matched,errors='coerce')/den.replace(0,np.nan)
    z['retained_fraction']=pd.to_numeric(z.retained_rows,errors='coerce')/pd.to_numeric(z.source_rows,errors='coerce').replace(0,np.nan)
    fig,axs=plt.subplots(1,2,figsize=(7.2,4.4))
    colors={'same_named_date':BLUE,'plusminus1day_window_expansion':ORANGE}
    for al,g in q.groupby('alignment'):
        axs[0].scatter(g.gate_quantile,g.pearson_r,s=9,alpha=.5,c=colors.get(al,GREY),label=al.replace('_',' '))
    axs[0].set_xticks([.8,.9,.95],['P80','P90','P95']);axs[0].set_xlabel('Training-only fire quantile');axs[0].set_ylabel('Pearson r');axs[0].legend(fontsize=5.2);axs[0].set_title('a  Correlation across strict HMS strata',loc='left',fontsize=7.4);axs[0].grid(axis='y',color=LIGHT,lw=.4)
    for al,g in z.groupby('alignment'):
        axs[1].scatter(g.known_fraction,g.retained_fraction,s=8,alpha=.42,c=colors.get(al,GREY),label=al.replace('_',' '))
    axs[1].set_xlim(-.03,1.03);axs[1].set_ylim(-.03,1.03);axs[1].set_xlabel('Known / (known + unknown) matched days');axs[1].set_ylabel('Retained / source rows');axs[1].legend(fontsize=5.2);axs[1].set_title('b  HMS availability and row retention',loc='left',fontsize=7.4);axs[1].grid(color=LIGHT,lw=.4)
    panel_label(axs[0],'a');panel_label(axs[1],'b');fig.suptitle('HMS definition and date alignment change support and association',y=1.01,fontsize=8.7);fig.tight_layout(rect=[0,0,1,.98])
    cap=('Supplementary Figure S6 | HMS definition, date-alignment and fire-gate sensitivity. (a) Strict log-smear/smearing correlations for all recorded model scopes, plume definitions, alignments and P80/P90/P95 gates using the available-calendar plume fraction. (b) Matched-day known fractions and retained-row fractions from the associated coverage records. Points are analytical strata with shared supports, not independent model replications. Unknown archive days remain unknown; same-named dates do not establish OC sampling-window overlap.')
    save_pub('Supplementary_Figure_S6',fig,cap,{'a':q,'b':z},core_claim='HMS availability definitions and date alignment alter the available support and descriptive association.',sources=register_catalog('S6','hms_correlations','hms_coverage'),
      filters={'panel_a':'qa strict, target log_smear, backtransform smearing, HMS_indicator=plume_fraction_available_calendar','panel_b':'qa strict, target log_smear, backtransform smearing','alignments':'all source alignments retained'},
      replicate_meaning='Each point is a saved country/model/year/threshold analytical stratum; shared station supports are not independent replicates.')

def plot_s7():
    import xarray as xr
    fig,axs=plt.subplots(2,4,figsize=(7.2,4.9),sharex=True,sharey=True); allrows=[]; sources=[]; last=None
    months=[1,4,7,10]; names=['Jan','Apr','Jul','Oct']
    for i,country in enumerate(['USA','CAN']):
        p=GRID/f'{country}_strict_with_met_log_smear.nc'
        if not p.exists(): raise FileNotFoundError(p)
        ds=xr.open_dataset(p)
        if 'pm25_smoke_proxy' not in ds: raise RuntimeError(f'{country} grid source lacks pm25_smoke_proxy')
        da=ds['pm25_smoke_proxy']
        for j,mo in enumerate(months):
            if 'month' in da.dims: z=da.sel(month=mo)
            elif 'time' in da.dims: z=da.sel(time=da.time.dt.month==mo).squeeze('time')
            else: raise RuntimeError(f'{country} grid has no month/time dimension')
            frame=z.to_dataframe(name='model_component').reset_index();frame['country']=country;frame['month_panel']=mo
            lat=next((c for c in frame if c.lower() in ['lat','latitude','grid_lat']),None);lon=next((c for c in frame if c.lower() in ['lon','longitude','grid_lon']),None)
            if not lat or not lon: raise RuntimeError(f'Could not identify lon/lat coords in {country} grid')
            frame=frame.rename(columns={lat:'latitude',lon:'longitude'});frame['is_supported']=np.isfinite(pd.to_numeric(frame.model_component,errors='coerce'))
            allrows.append(frame)
            finite=frame[frame.is_supported]
            ax=axs[i,j];last=ax.scatter(finite.longitude,finite.latitude,c=finite.model_component,s=2.5,cmap='YlOrBr',vmin=0,vmax=10,rasterized=True)
            ax.set_title(f'{country} · {names[j]}',fontsize=6.5);ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');ax.tick_params(labelsize=5.2)
        ds.close();sources.append({'alias':f'{country}_strict_with_met_log_smear_grid','path':p,'role':'accepted sparse 2023 reference grid; nodata retained in panel CSV'})
    data=pd.concat(allrows,ignore_index=True)
    cb=fig.colorbar(last,ax=axs.ravel().tolist(),fraction=.025,pad=.025,extend='max');cb.set_label('Model component (µg m⁻³)',fontsize=6.3);cb.ax.tick_params(labelsize=5.5)
    fig.suptitle('Sparse reference-grid integration: selected 2023 quarters',y=1.01,fontsize=8.7);fig.text(.5,.01,'Finite cells plotted; unsupported cells remain in source data and are not interpolated.',ha='center',fontsize=5.8);fig.subplots_adjust(left=.07,right=.91,bottom=.10,top=.91,wspace=.27,hspace=.35)
    cap=('Supplementary Figure S7 | Sparse reference-grid integration for January, April, July and October 2023. The strict-QA with-MET log-smear/smearing P90 model component is shown for the United States and Canada. Finite grid cells are plotted; unsupported cells are retained as nodata in the complete panel source CSV, with no interpolation. This station-supported integration is not a complete national/global exposure release or an identified smoke-mass product.')
    save_pub('Supplementary_Figure_S7',fig,cap,{'a':data},core_claim='The saved 2023 integration grid is sparse and unsupported cells remain missing.',sources=sources,
      filters={'country':['USA','CAN'],'month':[1,4,7,10],'qa':'strict','profile':'with_met','target':'log_smear/smearing','draw':'all finite cells; preserve all nodata rows in source CSV'},
      replicate_meaning='Each row is a source grid cell at one country/month coordinate; cells are spatially dependent and are not independent replicates.',
      excluded=['No unsupported cell is interpolated or rendered as an observed value.'])

def plot_f6():
    qa_path=MONTHLY/'input_manifest'/'qa.json'
    qa=json.loads(qa_path.read_text(encoding='utf-8'))
    if qa.get('status')!='passed': raise RuntimeError('Monthly SatPM input QA did not pass')
    product=qa.get('product',{})
    if product.get('version')!='V6GL03' or product.get('resolution_degrees')!=0.1 or product.get('temporal_resolution')!='monthly' or product.get('months')!=24:
        raise RuntimeError('F6 source is not the QA-accepted native 0.1-degree monthly 2023–24 product')
    expected_files={'monthly_benchmark_metrics.csv':24,'monthly_benchmark_paired_deltas.csv':96,
                    'monthly_match_coverage.csv':24,'monthly_matched_station_months.csv':73116}
    for name,nrows in expected_files.items():
        p=MONTHLY/name
        if not p.exists(): raise FileNotFoundError(p)
        qahash=qa.get('outputs',{}).get('sha256',{}).get(str(p))
        if not qahash or sha(p)!=qahash: raise RuntimeError(f'F6 input hash mismatch: {name}')
        qacount=qa.get('outputs',{}).get('rows',{}).get(str(p))
        if qacount!=nrows: raise RuntimeError(f'F6 QA row-count mismatch for {name}: {qacount}')
    raw_qa_hash=sha(qa_path)
    source_product_hashes={Path(k).name:v for k,v in qa.get('input_sha256',{}).items() if '/raw_monthly_gl/' in k.replace('\\','/')}
    output_hashes={Path(k).name:v for k,v in qa.get('outputs',{}).get('sha256',{}).items()}
    download_hash=next((v for k,v in qa.get('input_sha256',{}).items() if Path(k).name=='satpm_monthly_download_manifest.json'),None)
    qa_summary={'status':qa.get('status'),'product':product,'source_month_file_sha256':source_product_hashes,
       'output_file_sha256':output_hashes,'output_rows':{Path(k).name:v for k,v in qa.get('outputs',{}).get('rows',{}).items()},
       'key_definition':qa.get('key_definition'),'target_match_qa':qa.get('target_match_qa'),'comparisons':qa.get('comparisons'),
       'bootstrap':qa.get('bootstrap'),'limitations':qa.get('limitations'),'raw_qa_sha256':raw_qa_hash,
       'download_manifest_sha256':download_hash}
    qa_summary_path=SOURCE_OUT/'monthly_satpm_qa_summary_input.json';qa_summary_path.parent.mkdir(parents=True,exist_ok=True)
    qa_summary_path.write_text(json.dumps(qa_summary,indent=2,ensure_ascii=False),encoding='utf-8')
    for name,p in [('monthly_benchmark_metrics.csv',MONTHLY/'monthly_benchmark_metrics.csv'),
                   ('monthly_benchmark_paired_deltas.csv',MONTHLY/'monthly_benchmark_paired_deltas.csv'),
                   ('monthly_match_coverage.csv',MONTHLY/'monthly_match_coverage.csv')]:
        if p.stat().st_size>10_000_000: raise RuntimeError(f'F6 summary input unexpectedly large: {name}')
    metrics=pd.read_csv(MONTHLY/'monthly_benchmark_metrics.csv',low_memory=False)
    deltas=pd.read_csv(MONTHLY/'monthly_benchmark_paired_deltas.csv',low_memory=False)
    coverage=pd.read_csv(MONTHLY/'monthly_match_coverage.csv',low_memory=False)
    common=((metrics.qa_variant=='strict')&(metrics.estimator_kind=='hgb')&(metrics.arm=='full')&
        (metrics.target_strategy=='log_smear')&(metrics.backtransform=='smearing')&
        (metrics.comparison=='SatPM_V6GL03_monthly'))
    m=metrics[common&metrics.external_year.isin([2023,2024])&metrics.validation_mode.isin(['fixed','rolling'])].copy()
    dd=deltas[(deltas.qa_variant=='strict')&(deltas.estimator_kind=='hgb')&(deltas.arm=='full')&
        (deltas.target_strategy=='log_smear')&(deltas.backtransform=='smearing')&
        (deltas.comparison=='SatPM_V6GL03_monthly')&(deltas.metric=='rmse')&
        deltas.external_year.isin([2023,2024])&deltas.validation_mode.isin(['fixed','rolling'])].copy()
    cov=coverage[(coverage.qa_variant=='strict')&(coverage.estimator_kind=='hgb')&(coverage.arm=='full')&
        (coverage.target_strategy=='log_smear')&(coverage.backtransform=='smearing')&
        (coverage.comparison=='SatPM_V6GL03_monthly')&coverage.external_year.isin([2023,2024])&coverage.validation_mode.isin(['fixed','rolling'])].copy()
    if len(m)!=12 or len(dd)!=12 or len(cov)!=12: raise RuntimeError(f'F6 expected 12 matched cells in metrics/deltas/coverage, got {len(m)}/{len(dd)}/{len(cov)}')
    keys=['country','validation_mode','external_year','profile','model_alias']
    if m.duplicated(keys).any() or dd.duplicated(keys).any() or cov.duplicated(keys).any(): raise RuntimeError('F6 duplicate country/year/mode/profile/model key')
    if not dd.ci_status.eq('ci_computed').all() or not dd.bootstrap_replicates.eq(500).all() or not dd.bootstrap_seed.eq(20261116).all(): raise RuntimeError('F6 bootstrap/CI metadata differs from the QA-accepted contract')
    merged=m.merge(dd[keys+['delta','ci_low','ci_high','n_matched_pairs','n_matched_stations','ci_status']],on=keys,how='inner',validate='one_to_one',suffixes=('_metric','_delta'))
    if len(merged)!=12 or not np.allclose(merged.full_rmse_ug_m3-merged.comparator_rmse_ug_m3,merged.delta,atol=1e-8,rtol=0): raise RuntimeError('F6 paired delta does not match full minus SatPM RMSE')
    if not ((dd.loc[dd.external_year==2024,'ci_low']<=0)&(dd.loc[dd.external_year==2024,'ci_high']>=0)).all(): raise RuntimeError('F6 2024 interval does not cross zero as expected')
    us23=dd[dd.country.eq('USA')&dd.external_year.eq(2023)];can23=dd[dd.country.eq('CAN')&dd.external_year.eq(2023)]
    if not (len(us23)==2 and (us23.delta>0).all() and len(can23)==2 and (can23.delta<0).all()): raise RuntimeError('F6 2023 USA/CAN adverse direction check failed')
    merged['row_label']=merged.country+' '+merged.external_year.astype(str)+' '+merged.validation_mode+' '+merged.profile.str.replace('_',' ',regex=False)
    merged=merged.sort_values(['country','external_year','validation_mode','profile']).reset_index(drop=True)
    labels=merged.row_label.tolist(); y=np.arange(len(merged)); colors=merged.profile.map({'with_met':BLUE,'no_met':ORANGE})
    fig,axs=plt.subplots(1,2,figsize=(7.2,6.2),gridspec_kw={'width_ratios':[1,1.15]},sharey=True)
    for i,r in merged.iterrows():
        col=colors.iloc[i]
        axs[0].plot([r.full_rmse_ug_m3,r.comparator_rmse_ug_m3],[i,i],color=LIGHT,lw=.8,zorder=1)
        axs[0].scatter(r.full_rmse_ug_m3,i-.08,c=col,marker='o',s=13,label='Full model' if i==0 else None,zorder=3)
        axs[0].scatter(r.comparator_rmse_ug_m3,i+.08,facecolors='white',edgecolors=col,marker='s',s=14,label='SatPM V6GL03' if i==0 else None,zorder=3)
        axs[1].plot([r.ci_low,r.ci_high],[i,i],color=col,lw=.9)
        axs[1].scatter(r.delta,i,marker='o' if r.country=='USA' else 's',c=col,s=13)
    axs[0].set_yticks(y,labels,fontsize=5.2);axs[0].invert_yaxis();axs[0].set_xlabel('RMSE (µg m⁻³)');axs[0].set_title('a  Same-support RMSE',loc='left',fontsize=7.4);axs[0].grid(axis='x',color=LIGHT,lw=.4);axs[0].legend(fontsize=5.2)
    axs[1].axvline(0,color=GREY,lw=.65,ls=':');axs[1].set_xlabel('ΔRMSE: full − SatPM (µg m⁻³)');axs[1].set_title('b  Paired site-bootstrap 95% CI',loc='left',fontsize=7.4);axs[1].grid(axis='x',color=LIGHT,lw=.4)
    axs[0].set_ylabel('Country · year · update mode · profile');panel_label(axs[0],'a');panel_label(axs[1],'b')
    fig.suptitle('Native monthly SatPM comparison on matched station-month support',y=1.01,fontsize=8.7);fig.text(.5,.004,'Negative full−SatPM ΔRMSE favors the full model; SatPM is a calibrated comparator, not independent truth.',ha='center',fontsize=5.8);fig.tight_layout(rect=[0,0,1,.98])
    cap=('Figure6 | Full-model performance versus native monthly SatPM V6GL03 on identical station-month support. (a) Full-model and SatPM RMSE for 12 fixed/rolling country–year–profile cells: fixed 2023, fixed 2024 and rolling 2024. (b) Paired ΔRMSE (full minus SatPM) with conditional station-cluster bootstrap 95% intervals (500 resamples; seed 20261116). Negative differences favor the full model. The 2023 contrast is adverse for the United States and favors the full model for Canada; all 2024 intervals cross zero. SatPM V6GL03 has ground-monitor calibration overlap and is not independent truth. The comparison ends in 2024; no 2025 value is imputed. Source data are provided in the panel and complete matched station-month files.')
    monthly_sources=[{'alias':'monthly_benchmark_metrics','path':MONTHLY/'monthly_benchmark_metrics.csv','role':'QA-passed native 0.1-degree monthly SatPM matched metrics'},
      {'alias':'monthly_benchmark_paired_deltas','path':MONTHLY/'monthly_benchmark_paired_deltas.csv','role':'QA-passed paired bootstrap contrast table'},
      {'alias':'monthly_match_coverage','path':MONTHLY/'monthly_match_coverage.csv','role':'matched support and eligibility denominators'},
      {'alias':'monthly_matched_station_months','path':MONTHLY/'monthly_matched_station_months.csv','role':'all 73,116 station-month match records'},
      {'alias':'monthly_satpm_qa_summary','path':qa_summary_path,'role':'sanitized QA summary; raw QA hash retained','original_source_sha256':raw_qa_hash},
      ]
    save_pub('Figure_6_Monthly_Benchmark',fig,cap,{'a':merged,'b':dd.sort_values(['country','external_year','validation_mode','profile']).reset_index(drop=True)},
      core_claim='On exact matched monthly station support, the full-model advantage reverses by country in 2023 and is uncertain in 2024.',sources=monthly_sources,
      filters={'product':'native V6GL03 0.1-degree monthly SatPM, 2023-01 through 2024-12','comparison':'SatPM_V6GL03_monthly only; CAMS raw not mixed','model':'strict HGB full log_smear/smearing','years_modes':'fixed 2023, fixed 2024, rolling 2024','cells':12},
      replicate_meaning='Exact site-year-month matched support; 500 site-cluster bootstrap resamples, seed 20261116, conditional on saved OOF predictions. No model refit; site-months and profile/mode rows are not independent replicates.',
      stats='RMSE in µg m−3; paired 95% site-cluster bootstrap interval; matched-pair and station denominators are retained in each row.',
      excluded=['The QA-passed analysis is restricted to fixed 2023, fixed 2024 and rolling 2024; 2025 is unavailable under the locked source-support window.'])

def plot_s8():
    source=PROTOCOL/'spatial_split_geometry_summary.csv'
    raw=pd.read_csv(source,low_memory=False)
    if len(raw)!=154: raise RuntimeError(f'S8 expected 154 protocol geometry rows, got {len(raw)}')
    required={'country','task','fold_id','split_rule_code','buffer_km','n_train_stations','n_test_stations','n_buffer_excluded_stations','distance_km_min','distance_km_median','distance_km_p90','test_within_50km_train_pct','geometry_distance_evidence','source_path','source_job_id','status'}
    if not required.issubset(raw.columns): raise RuntimeError(f'S8 source missing fields: {sorted(required-set(raw.columns))}')
    key=['country','task','fold_id','source_job_id']
    if raw.duplicated(key).any(): raise RuntimeError('S8 duplicate country/task/fold/source-job key')
    # Absolute source paths are provenance-only and are replaced in the delivered tables.
    raw['source_path']='[redacted; see spatial geometry source alias]'
    safe_path=SOURCE_OUT/'spatial_split_geometry_summary_sanitized_input.csv'
    safe_path.parent.mkdir(parents=True,exist_ok=True);raw.to_csv(safe_path,index=False,na_rep='')
    original_hash=sha(source)
    tasks=['T2_spatial','T2_spatial_buffer50km'];q=raw[raw.task.isin(tasks)].copy()
    if q.empty: raise RuntimeError('S8 has no spatial task rows')
    countries=sorted(q.country.unique());xpos={c:i for i,c in enumerate(countries)}
    job_levels=sorted(q.source_job_id.astype(str).unique());job_offset={v:o for v,o in zip(job_levels,np.linspace(-.22,.22,max(1,len(job_levels))))}
    q['x_country']=q.country.map(xpos)+q.source_job_id.astype(str).map(job_offset)
    q['n_train_stations']=pd.to_numeric(q.n_train_stations,errors='coerce')
    q['n_test_stations']=pd.to_numeric(q.n_test_stations,errors='coerce')
    for col in ['distance_km_min','distance_km_median','distance_km_p90']:
        q[col]=pd.to_numeric(q[col],errors='coerce')
    if q.loc[q.country.isin(['USA','CAN']),['distance_km_min','distance_km_median','distance_km_p90']].notna().any().any():
        raise RuntimeError('S8 expected US/Canadian distances to remain NA in the source')
    def country_labels(subset):
        labels=[]
        for country in countries:
            n=int(subset.loc[subset.country==country,'distance_km_median'].isna().sum())
            labels.append(country+(f'\nNA n={n}' if n else ''))
        return labels
    fig,axs=plt.subplots(1,2,figsize=(7.2,5.6),gridspec_kw={'width_ratios':[1.15,1]})
    # Paired station counts retain every exact source-job/fold row.
    task_style={'T2_spatial':(BLUE,'o'),'T2_spatial_buffer50km':(ORANGE,'s')}
    for r in q.itertuples():
        color,mark=task_style.get(r.task,(GREY,'o'))
        if pd.notna(r.n_train_stations) and pd.notna(r.n_test_stations):
            axs[0].plot([r.x_country-.035,r.x_country+.035],[r.n_train_stations,r.n_test_stations],color=LIGHT,lw=.35,zorder=1)
            axs[0].scatter(r.x_country-.035,r.n_train_stations,c=color,marker=mark,s=7,alpha=.58,zorder=2)
            axs[0].scatter(r.x_country+.035,r.n_test_stations,facecolors='white',edgecolors=color,marker=mark,s=8,linewidths=.45,alpha=.72,zorder=2)
    axs[0].set_xticks(range(len(countries)),countries,rotation=50,ha='right',fontsize=5.0);axs[0].set_xlabel('Country; each paired mark is a saved fold row');axs[0].set_ylabel('Stations per fold');axs[0].set_title('a  Training and test station counts',loc='left',fontsize=7.0);axs[0].grid(axis='y',color=LIGHT,lw=.4)
    handles=[plt.Line2D([],[],marker=task_style[t][1],color=task_style[t][0],lw=0,label=t.replace('T2_spatial_buffer50km','T2 buffer').replace('T2_spatial','T2 spatial')) for t in task_style]
    handles += [plt.Line2D([],[],marker='o',color=GREY,lw=0,label='Training'),plt.Line2D([],[],marker='o',markerfacecolor='white',markeredgecolor=GREY,color=GREY,lw=0,label='Test')]
    axs[0].legend(handles=handles,fontsize=5.0,ncol=2,loc='upper right')
    t2=q[q.task=='T2_spatial']
    for r in t2.itertuples():
        if pd.notna(r.distance_km_min) and pd.notna(r.distance_km_p90): axs[1].vlines(r.x_country,r.distance_km_min,r.distance_km_p90,color=BLUE,alpha=.30,lw=.65)
        if pd.notna(r.distance_km_median): axs[1].scatter(r.x_country,r.distance_km_median,c=BLUE,s=9,alpha=.62)
    axs[1].set_xticks(range(len(countries)),country_labels(t2),rotation=50,ha='right',fontsize=5.0);axs[1].set_xlabel('Country (NA count is shown for missing distance rows)');axs[1].set_ylabel('Train-to-test station distance (km)');axs[1].set_title('b  T2 distance: minimum–P90, dot = median',loc='left',fontsize=6.8);axs[1].grid(axis='y',color=LIGHT,lw=.4)
    for ax,l in zip(axs,['a','b']):panel_label(ax,l)
    fig.suptitle('Saved spatial-fold geometry by country and task',y=1.01,fontsize=8.5);fig.text(.5,.003,'Offsets separate source jobs; paired marks show saved train/test station counts. Missing distances remain NA. No coordinates, maps or interpolated geometry are inferred.',ha='center',fontsize=5.6);fig.tight_layout(rect=[0,.02,1,.98])
    cap=('Supplementary Figure S1 | Saved spatial-fold geometry summaries, without reconstructing a map. (a) Training (filled) and test (open) station counts are shown as paired values for each country/task/source-job/fold row. (b) T2 train-to-test station distance ranges (minimum to P90; dot, median) are shown by country and fold. Horizontal offsets distinguish saved source jobs and carry no quantitative meaning. Missing United States and Canadian distance values remain NA and are counted in country labels; source rows are retained. No coordinates, block maps or interpolated geometry are inferred. Source paths are redacted from the delivered data table; the raw source hash is retained in the manifest.')
    save_pub('Supplementary_Figure_S1',fig,cap,{'a':q,'b':t2},core_claim='Saved spatial validation geometry is described by exact fold counts and distance summaries; no fold map is inferred without coordinates.',
      sources=[{'alias':'spatial_split_geometry_summary','path':safe_path,'role':'154-row protocol geometry source; absolute source_path redacted','original_source_sha256':original_hash,'redaction':'source_path field replaced with generic source alias; all metric/key rows retained'}],
      filters={'source_rows':154,'drawn_tasks':tasks,'per_fold_rows':'all source_job_id/fold rows; no averaging or deduplication','missing_values':'NA preserved and annotated by country'},
      replicate_meaning='Each point/range is one saved split-summary row; repeated source-job folds are related geometry records, not independent model evidence.',
      excluded=['No map is drawn because verified station-coordinate/block geometry is not included in this summary.'])
    if safe_path.exists(): safe_path.unlink()

def revision_completion_main():
    global VERIFIED
    SOURCE_OUT.mkdir(parents=True,exist_ok=True);VERIFIED=verify_catalog()
    mpath=OUT/'figure_manifest.json'
    if not mpath.exists(): raise RuntimeError('Completed 12-figure base manifest is missing')
    manifest=json.loads(mpath.read_text(encoding='utf-8'))
    existing=manifest.get('figures',[])
    required={'Figure_1_Historical_Validation','Figure_2_Future_Evaluation','Figure_4_Fire_Support','Figure_5_Calibration','Figure_3_Feature_Increments',
      'Supplementary_Figure_S3','Supplementary_Figure_S4','Supplementary_Figure_S2','Supplementary_Figure_S8','Supplementary_Figure_S5','Supplementary_Figure_S6','Supplementary_Figure_S7'}
    if {r['figure'] for r in existing}!=required: raise RuntimeError('Revision completion requires exactly the existing 12-figure base manifest')
    FIGURES.clear()
    plot_f3();plot_f4();plot_s3();plot_f6();plot_s8()
    revised={r['figure']:r for r in FIGURES}
    final=[]
    for rec in existing:
        final.append(revised.pop(rec['figure'],rec))
    final.extend(revised.values())
    if len(final)!=14 or {r['figure'] for r in final}!=required|{'Figure_6_Monthly_Benchmark','Supplementary_Figure_S1'}:
        raise RuntimeError(f'Revision completion must yield the 14 requested figures; got {len(final)}')
    FIGURES[:]=final
    parts=['# Figure captions — STOTEN revision','']
    for r in FIGURES: parts += [f'## {r["figure"]}','',r['caption'],'']
    (OUT/'caption_en.md').write_text('\n'.join(parts),encoding='utf-8')
    manifest['figures']=FIGURES
    manifest.pop('unavailable_figures',None)
    file_names={'caption_en.md'}
    for r in FIGURES:
        file_names.update(r['files']);file_names.add(f'{r["figure"]}_caption_en.md')
        file_names.update(s['filename'] for s in r['source_tables'])
    manifest['file_sha256']={name:sha(OUT/name) for name in sorted(file_names) if (OUT/name).is_file()}
    manifest['notes']=['Revision-only output; unmodified figures in the existing package were retained.','PDF, PNG and editable SVG are delivered; PNG uses 300 dpi for manuscript embedding.']
    mpath.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    qa=json.loads((OUT/'qa.json').read_text(encoding='utf-8')) if (OUT/'qa.json').exists() else {}
    qa.update({'status':'rendered; source and static QA pending','backend':'Python/matplotlib','figures_rendered':[r['figure'] for r in FIGURES],
      'expected_pngs':sorted(p.name for p in OUT.glob('*.png')),
      'pending':['nature-figure static validator','visual inspection of revised PNGs','PDF/SVG text selection and clipping check']})
    (OUT/'qa.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')
    print('REVISION_COMPLETION_RENDERED',len(FIGURES),'figures',flush=True)

def s3_layout_repair_main():
    global VERIFIED
    SOURCE_OUT.mkdir(parents=True,exist_ok=True);VERIFIED=verify_catalog();FIGURES.clear()
    mpath=OUT/'figure_manifest.json'
    manifest=json.loads(mpath.read_text(encoding='utf-8'))
    existing=manifest.get('figures',[])
    if len(existing)!=14 or not any(r['figure']=='Supplementary_Figure_S2' for r in existing):
        raise RuntimeError('S3-only repair requires the completed 14-figure manifest')
    plot_s3();revised=FIGURES[-1]
    final=[revised if r['figure']=='Supplementary_Figure_S2' else r for r in existing]
    parts=['# Figure captions — STOTEN revision','']
    for r in final: parts += [f'## {r["figure"]}','',r['caption'],'']
    (OUT/'caption_en.md').write_text('\n'.join(parts),encoding='utf-8')
    manifest['figures']=final
    file_names={'caption_en.md'}
    for r in final:
        file_names.update(r['files']);file_names.add(f'{r["figure"]}_caption_en.md')
        file_names.update(s['filename'] for s in r['source_tables'])
    manifest['file_sha256']={name:sha(OUT/name) for name in sorted(file_names) if (OUT/name).is_file()}
    mpath.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    qa=json.loads((OUT/'qa.json').read_text(encoding='utf-8')) if (OUT/'qa.json').exists() else {}
    qa.update({'status':'rendered; final S3 layout re-exported; QA pending','figures_rendered':[r['figure'] for r in final],
      'expected_pngs':sorted(p.name for p in OUT.glob('*.png')),
      'pending':['nature-figure static validator','visual inspection of revised S3 PNG','PDF/SVG text selection and clipping check']})
    (OUT/'qa.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')
    print('S3_LAYOUT_REPAIRED',len(final),'figures; only S3 rerendered',flush=True)

def incremental_main():
    global VERIFIED
    SOURCE_OUT.mkdir(parents=True,exist_ok=True);VERIFIED=verify_catalog()
    mpath=OUT/'figure_manifest.json'
    if not mpath.exists(): raise RuntimeError('Run the base revision figure bundle before incremental additions')
    manifest=json.loads(mpath.read_text(encoding='utf-8'))
    existing=manifest.get('figures',[])
    required={'Figure_1_Historical_Validation','Figure_2_Future_Evaluation','Figure_4_Fire_Support','Figure_5_Calibration','Figure_3_Feature_Increments',
              'Supplementary_Figure_S3','Supplementary_Figure_S4','Supplementary_Figure_S2','Supplementary_Figure_S8','Supplementary_Figure_S5','Supplementary_Figure_S6','Supplementary_Figure_S7'}
    if {r['figure'] for r in existing}!=required: raise RuntimeError('Incremental add-on requires the completed 12-figure base manifest')
    FIGURES.extend(existing)
    plot_f6();plot_s8()
    parts=['# Figure captions — STOTEN revision','']
    for r in FIGURES: parts += [f'## {r["figure"]}','',r['caption'],'']
    (OUT/'caption_en.md').write_text('\n'.join(parts),encoding='utf-8')
    manifest['figures']=FIGURES
    manifest.pop('unavailable_figures',None)
    file_names={'caption_en.md'}
    for r in FIGURES:
        file_names.update(r['files']);file_names.add(f'{r["figure"]}_caption_en.md')
        file_names.update(s['filename'] for s in r['source_tables'])
    manifest['file_sha256']={name:sha(OUT/name) for name in sorted(file_names) if (OUT/name).is_file()}
    manifest['notes']=['Revision-only output; no older package was edited.','PDF, PNG and editable SVG delivered; no TIFFs generated by this task.']
    mpath.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    qa=json.loads((OUT/'qa.json').read_text(encoding='utf-8'))
    qa['status']='rendered; static and visual QA pending'
    qa['figures_rendered']=[r['figure'] for r in FIGURES]
    qa['expected_pngs']=sorted(p.name for p in OUT.glob('*.png'))
    qa['pending']=['nature-figure static validator','visual inspection of every PNG','PDF/SVG text selection and clipping check']
    (OUT/'qa.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')
    print('INCREMENTAL_COMPLETE',len(FIGURES),'figures',flush=True)

def main():
    os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    SOURCE_OUT.mkdir(parents=True,exist_ok=True)
    global VERIFIED
    VERIFIED=verify_catalog()
    plot_f1();plot_f2();plot_f3();plot_f4();plot_f5()
    plot_s1();plot_s2();plot_s3();plot_s4();plot_s5();plot_s6();plot_s7()
    caption_path=OUT/'caption_en.md'
    parts=['# Figure captions — STOTEN revision','']
    for r in FIGURES: parts += [f'## {r["figure"]}','',r['caption'],'']
    caption_path.write_text('\n'.join(parts),encoding='utf-8')
    all_files=[p for p in OUT.iterdir() if p.is_file() and p.name not in {'figure_manifest.json','qa.json'}]
    manifest={'figures':FIGURES,'catalog_hashes_verified':VERIFIED,'unavailable_figures':{'Figure_6_Monthly_Benchmark':'Not rendered: the verified native 0.1-degree monthly SatPM 2023–24 result source is not yet available; annual-coordinate sampling data are not substituted.','Supplementary_Figure_S1':'Not rendered: protocol worker has not yet provided verified spatial-fold coordinates or exact geometry.'},'notes':['Revision-only output; no older package was edited.','No TIFFs were generated in this task-owned folder.']}
    manifest['file_sha256']={p.name:sha(p) for p in all_files}
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    qa={'status':'rendered; static and visual QA pending','backend':'Python/matplotlib','catalog_verified_files':len(VERIFIED),'figures_rendered':[r['figure'] for r in FIGURES],
        'exports':['PDF editable TrueType','PNG 300 dpi for manuscript/WPS embedding','SVG editable text'],'expected_pngs':sorted(p.name for p in OUT.glob('*.png')),
        'pending':['nature-figure static validator','visual inspection of every PNG','PDF/SVG text selection and clipping check'],'memory_policy':'sequential source loading; OMP/BLAS threads=1'}
    (OUT/'qa.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')
    print('RENDER_COMPLETE',len(FIGURES),'figures',flush=True)

if __name__=='__main__':
    import sys
    if '--s3-layout-repair' in sys.argv: s3_layout_repair_main()
    elif '--revision-completion' in sys.argv: revision_completion_main()
    elif '--incremental' in sys.argv: incremental_main()
    else: main()
