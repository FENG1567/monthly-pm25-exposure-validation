#!/usr/bin/env python3
"""Build isolated, source-traceable S2/S4/S5 semantic corrections."""
# Publication exports are performed by pf.save_pub: fig.savefig(... .pdf),
# fig.savefig(... .svg), and fig.savefig(... .png, dpi=300); PDF/SVG retain text.
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.update({"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})

V=Path('/external/project/validation')
FIG_DIR=V/'stoten_revision_20261116/figures'
PLOT_MODULE=FIG_DIR/'plot_revision_figures.py'
OUT=Path(__file__).resolve().parent
SOURCE_OUT=OUT/'source_data'
SOURCE_OUT.mkdir(parents=True,exist_ok=True)
os.environ['FIGURE_VALIDATION_ROOT']=str(V)

spec=importlib.util.spec_from_file_location('revision_plot_helpers',PLOT_MODULE)
pf=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)
pf.OUT=OUT
pf.SOURCE_OUT=SOURCE_OUT
pf.V=V
pf.DATA=V/'submission_publication_normalized_v2_20261115'
pf.RESULTS=V/'stoten_revision_20261116/results'
pf.INPUTS=FIG_DIR/'inputs'
pf.THIN_RUN=V/'station_thinning_final_execution_v1_20261115/final_run/run_20261115T020544Z_32c4673b59'
pf.THIN_ACCEPT=V/'station_thinning_postfit_audit_v1_20261115/acceptance.json'
pf.FIGURES.clear()

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE='#3B6C8E'; ORANGE='#D98847'; GREEN='#568F7B'; RED='#A85858'; GREY='#777777'; LIGHT='#D9E2E8'; PURPLE='#81739A'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'svg.fonttype':'none','pdf.fonttype':42,'font.size':6.2,'axes.spines.right':False,
    'axes.spines.top':False,'axes.linewidth':.65,'legend.frameon':False,
    'savefig.facecolor':'white','figure.facecolor':'white','axes.labelsize':6.1,
    'xtick.labelsize':5.5,'ytick.labelsize':5.2})

def plot_s2():
    loco=pf.load('loco_metrics')
    q=loco[(loco.scenario=='noMET')&(loco.arm=='full')&(loco.qa_variant=='strict')].copy()
    if len(q)!=34 or q.duplicated(['country','estimator_kind']).any() or set(q.estimator_kind)!={'hgb','ridge'}:
        raise RuntimeError('S2 LOCO expected 17 countries × 2 estimators with a unique country/estimator key')
    if q.country.nunique()!=17: raise RuntimeError('S2 LOCO must retain all 17 targets')

    input_thin=pd.read_csv(pf.INPUTS/'Supplementary_Figure_S4_Panel_b_Source_Data.csv',low_memory=False)
    accepted=pf.THIN_RUN/'metrics.csv'
    accepted_hashes=pf.verify_thinning_sources()
    accepted_thin=pd.read_csv(accepted,low_memory=False)
    strict=accepted_thin[(accepted_thin.qa_variant=='strict')&(accepted_thin.arm=='full')&accepted_thin.country.isin(['USA','CAN'])].copy()
    thin=input_thin[(input_thin.arm=='full')&(input_thin.qa_variant=='strict')&input_thin.country.isin(['USA','CAN'])].copy()
    key=['dataset_key','country','target_policy','qa_variant','profile','arm','sample_size','seed']
    required=set(key+['rmse','r2','n_train_stations'])
    if not required.issubset(thin.columns) or not required.issubset(strict.columns):
        raise RuntimeError('S2 thinning source lacks accepted key or metric fields')
    if thin.duplicated(key).any() or strict.duplicated(key).any(): raise RuntimeError('S2 thinning duplicated accepted source key')
    if len(strict)!=len(thin): raise RuntimeError(f'S2 formal/accepted strict row count differs: {len(thin)}/{len(strict)}')
    comp=thin[key+['rmse','r2','n_train_stations']].merge(strict[key+['rmse','r2','n_train_stations']],on=key,how='outer',validate='one_to_one',suffixes=('_formal','_accepted'),indicator=True)
    if not comp._merge.eq('both').all(): raise RuntimeError('S2 formal thinning rows do not match accepted metrics keys')
    for col in ['rmse','r2','n_train_stations']:
        if not np.allclose(comp[f'{col}_formal'].astype(float),comp[f'{col}_accepted'].astype(float),equal_nan=True):
            raise RuntimeError(f'S2 formal thinning {col} differs from accepted metrics')
    seeds=sorted(s for s in strict.seed.astype(str).unique() if s!='shared_full')
    formal_seeds=sorted(s for s in thin.seed.astype(str).unique() if s!='shared_full')
    if len(seeds)!=3 or seeds!=formal_seeds: raise RuntimeError(f'S2 accepted source has {len(seeds)} fixed seeds: {seeds}')
    if 'shared_full' not in set(strict.seed.astype(str)): raise RuntimeError('S2 accepted source lacks shared_full endpoints')

    thin['train_n']=pd.to_numeric(thin.sample_size,errors='coerce').fillna(pd.to_numeric(thin.n_train_stations,errors='coerce'))
    # Strictly positive finite training counts are required before the log axis.
    if not np.isfinite(thin.train_n).all() or (thin.train_n<=0).any(): raise RuntimeError('S2 log-scaled training counts must be positive finite values')
    if thin.duplicated(['country','profile','seed','train_n']).any(): raise RuntimeError('S2 plotted thinning key is not unique')
    shared=thin[thin.seed.astype(str)=='shared_full'].copy()
    if len(shared)!=4 or shared.duplicated(['country','profile']).any(): raise RuntimeError('S2 must have one shared_full endpoint per country/profile')
    fig,axs=plt.subplots(1,3,figsize=(7.2,6.25),gridspec_kw={'width_ratios':[1.35,1,1]})
    countries=sorted(q.country.unique()); ymap={c:i for i,c in enumerate(countries)}
    for est,col,mark in [('hgb',BLUE,'o'),('ridge',ORANGE,'s')]:
        g=q[q.estimator_kind==est]
        axs[0].scatter(g.r2,g.country.map(ymap),s=10,c=col,marker=mark,label=est.upper())
    axs[0].set_yticks(range(len(countries)),countries);axs[0].set_xscale('symlog',linthresh=.1);axs[0].axvline(0,color=GREY,lw=.5)
    axs[0].set_xlabel('LOCO R² (symlog)');axs[0].set_title('a  Strict no-MET transfer',loc='left',fontsize=7.1);axs[0].legend(fontsize=5.0);axs[0].grid(axis='x',color=LIGHT,lw=.4)
    seed_styles={seeds[0]:'-',seeds[1]:'--',seeds[2]:':'}
    profile_colors={'with_met':BLUE,'no_met':ORANGE}
    for ax,country,lab in zip(axs[1:],['USA','CAN'],['b','c']):
        z=thin[thin.country==country]
        for profile,color in profile_colors.items():
            for seed in seeds:
                g=z[(z.profile==profile)&(z.seed.astype(str)==seed)].sort_values('train_n')
                if g.empty: raise RuntimeError(f'S2 missing accepted seed curve {country}/{profile}/{seed}')
                ax.plot(g.train_n,g.rmse,marker='o',ms=2.0,lw=.65,ls=seed_styles[seed],color=color)
            endpoint=shared[(shared.country==country)&(shared.profile==profile)]
            ax.scatter(float(endpoint.train_n.iloc[0]),float(endpoint.rmse.iloc[0]),marker='D',s=15,facecolor='white',edgecolor=color,linewidth=.75,zorder=4)
        ax.set_xscale('log');ax.set_xlabel('Training stations');ax.set_ylabel('2022 RMSE (µg m⁻³)')
        ax.set_title(f'{lab}  {country} network thinning',loc='left',fontsize=7.1);ax.grid(axis='y',color=LIGHT,lw=.4)
    handles=[Line2D([],[],color=BLUE,lw=1.2,label='with MET'),Line2D([],[],color=ORANGE,lw=1.2,label='no MET')]
    handles += [Line2D([],[],color=GREY,lw=.8,ls=seed_styles[s],label=f'seed {s}') for s in seeds]
    handles += [Line2D([],[],marker='D',markerfacecolor='white',markeredgecolor=GREY,color='none',label='shared full endpoint')]
    fig.legend(handles=handles,fontsize=5.0,ncol=3,loc='lower center',bbox_to_anchor=(.5,.006),columnspacing=1.1,handlelength=1.5)
    fig.suptitle('Transfer and network-thinning sensitivity',y=.99,fontsize=8.5);fig.tight_layout(rect=[0,.13,1,.96])
    cap=('Supplementary Figure S4 | Transfer and monitoring-network sensitivity. (a) Strict-QA no-MET leave-one-country-out R² for all 17 held-out countries under HGB and Ridge, full feature arm; negative values are retained on a signed-log x-axis. (b,c) Full-arm 2022 RMSE across accepted nested training-site counts for all three prespecified fixed thinning seeds, separated by MET profile. Line style identifies seed; color identifies MET profile. The open diamond is the shared full-network endpoint and is plotted once per country/profile, not joined to the seed curves. Curves are descriptive sensitivity runs, not confidence intervals.')
    sources=pf.register_catalog('S2','loco_metrics')+pf.register_input('S2','Supplementary_Figure_S4_Panel_a_Source_Data.csv','Supplementary_Figure_S4_Panel_b_Source_Data.csv')
    sources += [{'alias':'accepted_thinning_metrics','path':accepted,'role':'128-row accepted thinning result; plot restricted to strict full-arm USA/CAN rows','accepted_artifact_sha256':accepted_hashes['metrics.csv']},
                {'alias':'thinning_acceptance','path':pf.THIN_ACCEPT,'role':'independent accepted status and artifact hashes'}]
    pf.save_pub('Supplementary_Figure_S4',fig,cap,{'a':q,'b':thin},core_claim='Transfer and network-size sensitivity are shown for all fixed seeds while the shared endpoint is clearly marked.',sources=sources,
        filters={'LOCO':'noMET, full arm, strict QA; 17 countries × HGB/Ridge; signed symlog retains negative R²','thinning':'accepted metrics.csv strict full arm, USA/CAN; fixed seeds '+', '.join(seeds)+'; shared_full is endpoint only','formal_vs_accepted':'formal 64-row panel matched accepted strict rows by dataset/country/policy/profile/arm/sample-size/seed and RMSE/R²/station counts'},
        replicate_meaning='LOCO countries are transfer targets. Thinning lines are three saved fixed-seed sensitivity series; shared_full is a common endpoint. No fits or new inference were computed.',
        excluded=['Non-strict accepted thinning rows are not plotted; their complete 128-row artifact remains registered.'])
    return {'figure':'Supplementary_Figure_S4','accepted_thinning_rows':len(accepted_thin),'plotted_thinning_rows':len(thin),'fixed_seeds':seeds,'shared_full_rows':len(shared),'accepted_metrics_sha256':accepted_hashes['metrics.csv']}

def s4_aliases():
    vals=set()
    for c in ['can','usa']:
        for profile in ['no_met','with_met']:
            if c=='can': vals.add(f'can_hist_primary_strict_{profile}_hgb')
            else: vals.add(f'usa_hist_strict_{profile}_hgb')
    for c in ['can','usa']:
        for year in [2023,2024]:
            for profile in ['no_met','with_met']:
                if c=='can': vals.add(f'can_primary_all_published_fixed_{year}_strict_hgb__{profile}')
                else: vals.add(f'usa_primary_fixed_{year}_strict_hgb__{profile}')
    return vals

def plot_s4():
    src=pf.RESULTS/'csv'/'annual_comparator_primary_subtable.csv'
    full_table=pd.read_csv(src,low_memory=False)
    if len(full_table)!=352: raise RuntimeError(f'S4 primary comparator subtable expected 352 lineage rows, got {len(full_table)}')
    if full_table.source_file.value_counts().to_dict().get('annual_deltas.csv')!=176:
        raise RuntimeError('S4 expected a 176-row annual_deltas lineage before selecting its source_file')
    annual=full_table[full_table.source_file=='annual_deltas.csv'].copy()
    aliases=s4_aliases()
    q=annual[(annual.support_scope=='primary_exact_12_months')&(annual.status=='eligible')&
      (annual.qa_variant=='strict')&(annual.estimator_kind=='hgb')&(annual.target_strategy=='log_smear')&
      (annual.backtransform=='smearing')&(annual.arm=='full')&(annual.contrast=='full minus SatPM')&
      (annual.alias.isin(aliases))].copy()
    q['mode']=np.select([q.alias.str.contains('fixed_2023'),q.alias.str.contains('fixed_2024')],['fixed 2023','fixed 2024'],default='historical annual')
    keys=['country','task','profile','mode']
    if len(q)!=24 or q.duplicated(keys).any(): raise RuntimeError(f'S4 expected 24 unique country/task/profile/mode rows; got {len(q)}; duplicate count={q.duplicated(keys,keep=False).sum()}')
    if q.groupby('country').size().to_dict()!={'CAN':12,'USA':12}: raise RuntimeError('S4 expected 12 selected rows per country')
    if set(q.country)!={'CAN','USA'} or set(q.profile)!={'no_met','with_met'}: raise RuntimeError('S4 selected country/profile set is incomplete')
    # Source-key assertions for the requested sign and future-year spot checks.
    def one(c,t,p,m):
        z=q[(q.country==c)&(q.task==t)&(q.profile==p)&(q['mode']==m)]
        if len(z)!=1: raise RuntimeError(f'S4 expected unique row for {c}/{t}/{p}/{m}; got {len(z)}')
        return float(z.iloc[0].delta_rmse)
    checks={'USA T2 with MET':one('USA','T2_spatial','with_met','historical annual'),
      'USA fixed 2023 with MET':one('USA','T4_external_2023_with_met','with_met','fixed 2023'),
      'USA fixed 2024 with MET':one('USA','T4_external_2024_with_met','with_met','fixed 2024')}
    if not (abs(checks['USA T2 with MET']-.719)<.001 and abs(checks['USA fixed 2023 with MET']+.179)<.001 and abs(checks['USA fixed 2024 with MET']-.004)<.001):
        raise RuntimeError(f'S4 signed primary spot checks changed: {checks}')
    task_order=['T1_station','T2_spatial','T2_spatial_buffer50km','T3_year_2022','T4_external_2023_no_met','T4_external_2023_with_met','T4_external_2024_no_met','T4_external_2024_with_met']
    task_labels={'T1_station':'T1 station','T2_spatial':'T2 spatial','T2_spatial_buffer50km':'T2 buffer','T3_year_2022':'T3 2022',
      'T4_external_2023_no_met':'Fixed 2023 · no MET','T4_external_2023_with_met':'Fixed 2023 · with MET',
      'T4_external_2024_no_met':'Fixed 2024 · no MET','T4_external_2024_with_met':'Fixed 2024 · with MET'}
    fig,axs=plt.subplots(1,2,figsize=(7.2,7.6),sharex=True)
    colors={'no_met':ORANGE,'with_met':BLUE}
    for ax,country,lab in zip(axs,['CAN','USA'],['a  Canada','b  United States']):
        g=q[q.country==country].copy()
        g['_task_order']=g.task.map({x:i for i,x in enumerate(task_order)})
        g['_profile_order']=g.profile.map({'no_met':0,'with_met':1})
        g=g.sort_values(['_task_order','_profile_order']).reset_index(drop=True)
        g['label']=[task_labels.get(t,t) if t.startswith('T4_') else task_labels.get(t,t)+' · '+('with MET' if p=='with_met' else 'no MET') for t,p in zip(g.task,g.profile)]
        y=np.arange(len(g))
        for j,r in enumerate(g.itertuples()):
            ax.plot([r.delta_rmse_ci_low,r.delta_rmse_ci_high],[j,j],color=colors[r.profile],lw=.75)
            ax.scatter(r.delta_rmse,j,c=colors[r.profile],s=10,zorder=3)
        ax.set_yticks(y,g.label,fontsize=5.0);ax.invert_yaxis();ax.axvline(0,color=GREY,lw=.55)
        ax.set_title(lab,loc='left',fontsize=7.0);ax.grid(axis='x',color=LIGHT,lw=.4);ax.set_xlabel('ΔRMSE: full − SatPM (µg m⁻³)')
    fig.legend([Line2D([],[],marker='o',color=colors['no_met'],lw=0),Line2D([],[],marker='o',color=colors['with_met'],lw=0)],['no MET','with MET'],ncol=2,fontsize=5.2,loc='lower center',bbox_to_anchor=(.5,.015))
    fig.suptitle('Annual exact-12-month primary comparison with SatPM',y=.99,fontsize=8.2)
    fig.text(.5,.055,'Positive full − SatPM favors SatPM; negative values favor the full model.',ha='center',fontsize=5.5)
    fig.tight_layout(rect=[0,.09,1,.955])
    cap=('Supplementary Figure S8 | Annual SatPM comparator contrasts on primary exact-12-month support. The two country facets each show 12 source rows: T1 station, T2 spatial, T2 buffer and T3 2022 historical tasks under both MET profiles, plus fixed 2023 and fixed 2024 external tasks under both profiles. Rows are selected from `annual_deltas.csv` by the primary strict-HGB/log-smear/smearing full-model aliases; repeated lineage copies tagged as Supplementary Figure source data, common-support records and rolling evaluations are excluded by explicit source/alias rules, not by score. For ΔRMSE = full − SatPM, positive values mean lower RMSE for SatPM and negative values favor the full model. Each selected annual comparison requires 12 exact matched months. The full annual CSV, including raw-CAMS comparison rows, is retained in the source package; the fitted-CAMS comparison is reported in the main Table 7 rather than duplicated here.')
    sources=[{'alias':'annual_comparator_primary_subtable','path':src,'role':'352-row lineage table; plotted rows selected only where source_file=annual_deltas.csv'},
      {'alias':'annual_deltas','path':pf.DATA/'annual_deltas.csv','role':'complete 176-row annual source retained, including CAMS comparisons'}]
    pf.save_pub('Supplementary_Figure_S8',fig,cap,{'a':q},core_claim='The prespecified strict-HGB full-model annual contrast against SatPM is shown on exact 12-month support by country and deployment task.',sources=sources,
      filters={'source_file':'annual_deltas.csv preferred over duplicate Supplementary_Figure_S8_Panel_b_Source_Data.csv lineage rows','support_scope':'primary_exact_12_months; eligible only','model':'strict / HGB / log_smear / smearing / full','contrast':'full minus SatPM only; CAMS rows remain in complete source CSV','alias_policy':'historical primary aliases + fixed 2023/2024 primary aliases; common_* and rolling_* excluded by alias','selected_rows':24,'row_key':'country × task × profile × mode, asserted unique; 12 rows per country'},
      replicate_meaning='Each row is a saved annual task/profile comparator estimate with its source interval; annual years and tasks are descriptive saved results, not independent monthly replicates.',
      excluded=['No annual CAMS comparison is plotted; complete annual_deltas.csv remains registered and supplied. No 2025 point is imputed because exact shared support is unavailable.'])
    return {'figure':'Supplementary_Figure_S8','source_registry_rows':len(full_table),'source_file_counts':full_table.source_file.value_counts().to_dict(),'selected_rows':len(q),'rows_per_country':q.groupby('country').size().to_dict(),'unique_key':['country','task','profile','mode'],'spot_checks_delta_rmse':checks,'selected_aliases':sorted(q.alias.unique())}

def _ci_plot(ax,z,label_col,value,lo,hi,profile_col='profile',colors=None,fontsize=5.0):
    z=z.sort_values(label_col).reset_index(drop=True)
    yy=np.arange(len(z))
    for j,r in enumerate(z.itertuples()):
        color=(colors or {'with_met':BLUE,'no_met':ORANGE}).get(getattr(r,profile_col,''),GREY)
        lv=getattr(r,lo);hv=getattr(r,hi);val=getattr(r,value)
        if pd.notna(lv) and pd.notna(hv): ax.plot([lv,hv],[j,j],color=color,lw=.65)
        ax.scatter(val,j,c=color,s=8,zorder=3)
    ax.set_yticks(yy,z[label_col],fontsize=fontsize);ax.invert_yaxis();ax.axvline(0,color=GREY,lw=.5);ax.grid(axis='x',color=LIGHT,lw=.4)

def plot_s5():
    a=pd.read_csv(pf.INPUTS/'Figure_3_Fire_Consistency_Panel_a_Source_Data.csv',low_memory=False)
    b=pd.read_csv(pf.INPUTS/'Figure_3_Fire_Consistency_Panel_b_Source_Data.csv',low_memory=False)
    c=pd.read_csv(pf.INPUTS/'Figure_3_Fire_Consistency_Panel_c_Source_Data.csv',low_memory=False)
    d=pd.read_csv(pf.INPUTS/'Figure_3_Fire_Consistency_Panel_d_Source_Data.csv',low_memory=False)
    if len(a)!=16 or set(a.country)!={'CAN','USA'} or set(a.profile)!={'no_met','with_met'}: raise RuntimeError('S5a source must retain 16 CAN/USA × MET-profile saved rows')
    if len(c)!=6 or len(d)!=12: raise RuntimeError(f'S5 c/d row count changed: {len(c)}/{len(d)}')
    c=c.copy();parts=c.scope.astype(str).str.split(':')
    c['country_label']=parts.str[0].str.split('_').str[0]
    c['parameter_code']=parts.str[-2];c['method_code']=parts.str[-1]
    c['row_label']=c.parameter_code+'/'+c.method_code+' · '+c.profile.map({'with_met':'with MET','no_met':'no MET'})
    if c.row_label.duplicated().any(): raise RuntimeError('S5c parameter/method/MET label is not unique')
    d=d.copy()
    d['alignment_short']=d.alignment.map({'same_named_date_EPA_local_HMS_UTC':'same date','plusminus1day_window_expansion':'±1 day'}).fillna(d.alignment)
    d['plume_short']=d.plume_definition.map({'any_plume':'any plume','medium_or_heavy':'medium/heavy'}).fillna(d.plume_definition)
    d['row_label']=d.parameter_code.astype(str)+'/'+d.method_code.astype(str)+' · '+d.alignment_short+' · '+d.plume_short
    if d.row_label.duplicated().any(): raise RuntimeError('S5d parameter/method/alignment/plume key is not unique')
    b=b.copy()
    if {'scope','comparator','profile'}.issubset(b.columns):
        parts_b=b.scope.astype(str).str.split(':');b['country_label']=parts_b.str[1].fillna('')
        b['row_label']=b.country_label+' · '+b.comparator.astype(str)+' · '+b.profile.map({'with_met':'with MET','no_met':'no MET'}).fillna(b.profile.astype(str))
    else: raise RuntimeError('S5b source lacks saved country/comparator/profile labels')
    if b.row_label.duplicated().any(): raise RuntimeError('S5b comparator labels are not unique')

    fig=plt.figure(figsize=(7.2,7.65));gs=fig.add_gridspec(3,2,height_ratios=[1.05,1.2,1.55],hspace=.60,wspace=.34)
    aa=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])]
    bb=fig.add_subplot(gs[1,0]);cc=fig.add_subplot(gs[1,1]);dd=fig.add_subplot(gs[2,:])
    profile_colors={'with_met':BLUE,'no_met':ORANGE}
    for ax,country in zip(aa,['USA','CAN']):
        z=a[a.country==country]
        for profile,color in profile_colors.items():
            for mode,marker,offset in [('fixed','o',-.08),('rolling','D',.08)]:
                g=z[(z.profile==profile)&z.task.astype(str).str.contains(mode,case=False)]
                if g.empty: continue
                x=pd.to_numeric(g.year,errors='coerce')+offset
                ax.errorbar(x,g.pearson_r,yerr=[g.pearson_r-g.ci_low,g.ci_high-g.pearson_r],fmt=marker,ms=2.6,lw=.55,capsize=1.4,color=color,markerfacecolor=color if mode=='fixed' else 'white',markeredgecolor=color)
        ax.set_xlim(2022.65,2025.35);ax.set_xticks([2023,2024,2025]);ax.set_xlabel('External evaluation year');ax.set_ylabel('Pearson r')
        ax.set_title(country,loc='left',fontsize=6.8);ax.grid(axis='y',color=LIGHT,lw=.4)
    aa[0].text(-.15,1.10,'a',transform=aa[0].transAxes,fontsize=8,fontweight='bold')
    aa[0].legend(handles=[Line2D([],[],marker='o',color=BLUE,lw=0,label='with MET'),Line2D([],[],marker='o',color=ORANGE,lw=0,label='no MET')],fontsize=5.0,loc='lower left',ncol=2)
    aa[1].legend(handles=[Line2D([],[],marker='o',color=GREY,lw=0,label='fixed'),Line2D([],[],marker='D',markerfacecolor='white',markeredgecolor=GREY,color='none',label='rolling')],fontsize=5.0,loc='lower left',ncol=2)
    _ci_plot(bb,b,'row_label','pearson_r','pearson_r_ci_low','pearson_r_ci_high')
    bb.set_xlabel('Pearson r');bb.set_title('2023 fire-product comparisons',loc='left',fontsize=6.8);pf.panel_label(bb,'b')
    _ci_plot(cc,c,'row_label','pearson_r','pearson_r_ci_low','pearson_r_ci_high')
    cc.set_xlabel('Pearson r');cc.set_title('EPA OC-tracer correlations',loc='left',fontsize=6.8);pf.panel_label(cc,'c')
    _ci_plot(dd,d,'row_label','mean_station_oc_difference','ci_low','ci_high',profile_col='__none__',colors={})
    dd.set_xlabel('Plume − control OC (µg m⁻³)');dd.set_title('Observed OC within-month contrasts',loc='left',fontsize=6.8);pf.panel_label(dd,'d')
    fig.suptitle('Fire and OC records provide complementary consistency evidence',y=.995,fontsize=8.2)
    fig.text(.5,.008,'HMS dates are UTC; EPA OC dates are local. “Same date” uses equal date labels; “±1 day” expands the local-date window. Correlations and OC contrasts are not smoke-mass measurements.',ha='center',fontsize=5.1)
    cap=('Supplementary Figure S5 | Complementary fire-related consistency evidence. (a) All 16 saved country/profile/year HMS plume-fraction correlations are shown as points with their saved intervals, faceted by country; color identifies MET profile and marker fill/shape distinguishes fixed from rolling evaluations. United States and Canadian records are never connected across countries. (b) Saved 2023 fire-product comparator correlations. (c) EPA OC-tracer correlation estimates labeled by the recorded parameter/method codes (for example, 88320/805) and MET profile. (d) Station-level observed OC plume-minus-control contrasts retain parameter/method, plume and date-alignment labels. “Same date” means matching date labels although HMS is UTC and EPA dates are local; “±1 day” means the expanded local-date window. Points and intervals are saved estimates; shared supports are not independent replication, OC is nonspecific, and these comparisons do not identify ground-level smoke mass.')
    sources=pf.register_catalog('S5','hms_correlations','smoke_future_comparators','oc_observed')
    sources+=pf.register_input('S5','Figure_3_Fire_Consistency_Panel_a_Source_Data.csv','Figure_3_Fire_Consistency_Panel_b_Source_Data.csv','Figure_3_Fire_Consistency_Panel_c_Source_Data.csv','Figure_3_Fire_Consistency_Panel_d_Source_Data.csv')
    pf.save_pub('Supplementary_Figure_S5',fig,cap,{'a':a,'b':b,'c':c,'d':d},core_claim='Fire, product, and OC records give complementary consistency evidence while country, MET profile, parameter, method and date alignment remain explicit.',sources=sources,
      filters={'panel_a':'all 16 formal saved rows; faceted CAN/USA, color by MET profile, fixed/rolling marker; no lines or cross-country connecting','panel_b':'all formal 2023 comparator rows; source ordering retained in labelled points','panel_c':'all 6 rows; parameter/method parsed from scope suffix and explicitly labeled with MET profile','panel_d':'all 12 rows; actual parameter/method codes retained with short date-alignment and plume labels'},
      replicate_meaning='Saved station/bootstrap intervals condition on source supports and fitted predictions; repeated country/profile/year modes and shared support hashes are not independent evidence.',
      excluded=[])
    return {'figure':'Supplementary_Figure_S5','panel_rows':{'a':len(a),'b':len(b),'c':len(c),'d':len(d)},'s5a_2023_CAN':a[(a.country=='CAN')&(a.year==2023)][['profile','pearson_r']].to_dict('records'),'s5a_2023_USA':a[(a.country=='USA')&(a.year==2023)][['profile','pearson_r']].to_dict('records'),'s5c_labels':c.row_label.tolist(),'s5d_labels':d.row_label.tolist()}

def main():
    verified=pf.verify_catalog()
    r2=plot_s2();r4=plot_s4();r5=plot_s5()
    captions=['# Supplementary figure captions — semantic corrections','']
    for r in pf.FIGURES: captions += [f"## {r['figure']}",'',r['caption'],'']
    (OUT/'captions_en.md').write_text('\n'.join(captions),encoding='utf-8')
    products=[r2,r4,r5]
    evidence={'backend':'Python/matplotlib','catalog_hashes_verified':len(verified),'figures':products,
      'figure_records':pf.FIGURES,'source_package':'complete source CSV copies are under source_data/','no_model_fit':True}
    (OUT/'manifest_records.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print(json.dumps({'status':'complete','output_dir':str(OUT),'figure_names':[r['figure'] for r in pf.FIGURES],
      'records':products,'manifest':'manifest_records.json','captions':'captions_en.md'},indent=2,ensure_ascii=False,default=str),flush=True)

if __name__=='__main__': main()
