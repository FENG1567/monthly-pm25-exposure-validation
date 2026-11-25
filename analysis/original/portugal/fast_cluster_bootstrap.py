"""Algebraically equivalent station-cluster bootstrap using group statistics.

This changes computation cost, not station draws, metric definitions or seeds.
Active frozen runs are not patched by importing this module.
"""
import numpy as np


def paired_bootstrap(group,y,pa,pb,replicates,seed):
    groups=np.asarray(group).astype(str)
    y,pa,pb=[np.asarray(values,dtype=float) for values in [y,pa,pb]]
    if not (len(groups)==len(y)==len(pa)==len(pb)) or not len(y):
        raise ValueError('Bootstrap arrays must have equal nonzero lengths')
    if not np.isfinite(np.column_stack([y,pa,pb])).all():
        raise ValueError('Bootstrap requires finite observations and predictions')
    if not isinstance(replicates,(int,np.integer)) or replicates<0:
        raise ValueError('Bootstrap replicates must be nonnegative integers')
    unique,codes=np.unique(groups,return_inverse=True)
    n_groups=len(unique)
    count=np.bincount(codes).astype(float)
    # Center once, then use within/between-group variance decomposition. This
    # avoids subtracting two large uncentered second moments for the R² metric.
    centered=y-y.mean()
    total_y=np.bincount(codes,weights=centered,minlength=n_groups)
    mean_y=total_y/count
    within_y=np.bincount(codes,weights=(centered-mean_y[codes])**2,minlength=n_groups)
    ea,eb=pa-y,pb-y
    sse_a=np.bincount(codes,weights=ea**2,minlength=n_groups)
    sse_b=np.bincount(codes,weights=eb**2,minlength=n_groups)
    abs_a=np.bincount(codes,weights=np.abs(ea),minlength=n_groups)
    abs_b=np.bincount(codes,weights=np.abs(eb),minlength=n_groups)
    sum_a=np.bincount(codes,weights=ea,minlength=n_groups)
    sum_b=np.bincount(codes,weights=eb,minlength=n_groups)
    rng=np.random.default_rng(seed)
    deltas={name:[] for name in ['rmse','mae','bias','r2']}
    reference_rows=None
    variance_fallback_scale=32*np.finfo(float).eps*max(float(np.mean(y**2)),1.)
    for _ in range(replicates):
        draw=rng.choice(n_groups,size=n_groups,replace=True)
        n=count[draw].sum()
        a,b=sse_a[draw].sum(),sse_b[draw].sum()
        mean=total_y[draw].sum()/n
        sst=(within_y[draw]+count[draw]*(mean_y[draw]-mean)**2).sum()
        # Match the reference row-level variance in numerically degenerate
        # cases. Such cases are cheap/small; ordinary cohorts use only O(G).
        if sst <= variance_fallback_scale*n:
            if reference_rows is None:
                reference_rows=[np.flatnonzero(codes==i) for i in range(n_groups)]
            yy=y[np.concatenate([reference_rows[i] for i in draw])]
            sst=np.sum((yy-yy.mean())**2)
        values={'rmse':np.sqrt(a/n)-np.sqrt(b/n),
                'mae':abs_a[draw].sum()/n-abs_b[draw].sum()/n,
                'bias':sum_a[draw].sum()/n-sum_b[draw].sum()/n,
                'r2':(1-a/sst)-(1-b/sst) if sst>0 else np.nan}
        for name,value in values.items():
            if np.isfinite(value):deltas[name].append(float(value))
    result={'bootstrap_replicates':replicates,'bootstrap_unit':'station',
            'bootstrap_independent_units':n_groups}
    for name,values in deltas.items():
        result[f'delta_{name}_lo']=float(np.percentile(values,2.5)) if values else np.nan
        result[f'delta_{name}_hi']=float(np.percentile(values,97.5)) if values else np.nan
    return result
