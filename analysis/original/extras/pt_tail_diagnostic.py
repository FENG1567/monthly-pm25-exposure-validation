#!/usr/bin/env python3
import os
for n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[n]="1"
import hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np,pandas as pd,sklearn
from sklearn.model_selection import GroupKFold
BASE=Path("/external/project/validation")
RUN=BASE/"ready_global_additional_models_20261113"; RESULTS=RUN/"results"
SOURCE=RUN/"inputs/PT/strict.parquet"; CODE=RUN/"source_snapshot/validation"
DIAG=BASE/"stoten_revision_20261116/results/pt_diagnostic"
sys.path.insert(0,str(CODE))
from validation_core import ModelConfig,estimator,feature_matrix
def fsha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def jsha(v):
 b=json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,default=str,allow_nan=False).encode()
 return hashlib.sha256(b).hexdigest()
def asha(v):
 a=np.ascontiguousarray(np.asarray(v)); h=hashlib.sha256()
 h.update(f"{a.dtype.str}|{a.shape}".encode("ascii")); h.update(b"\0"); h.update(a.tobytes()); return h.hexdigest()
def rowsha(df):
 k=[(str(c).upper(),str(s),int(y),int(m)) for c,s,y,m in df[["country","site_id","year","month"]].itertuples(index=False,name=None)]
 if len(set(k))!=len(k): raise ValueError("duplicate row keys")
 return jsha(sorted(k))
def trainfp(t,features):
 cols=list(dict.fromkeys(["country","site_id","year","month","lat","lon","pm25",*features,*(["aod_raw"] if "aod_used" in features else [])]))
 z=t[cols].copy(); z.insert(0,"_source_row_id",pd.to_numeric(t["_row_id"],errors="raise").to_numpy())
 return hashlib.sha256(z.to_csv(index=False,na_rep="<NA>",float_format="%.17g",lineterminator="\n").encode()).hexdigest()
def arr(d):
 a=np.asarray(d,dtype=float); e=np.exp(a)
 return {"resid_q50_q90_q95_q99_max":[float(x) for x in np.quantile(a,[.5,.9,.95,.99,1])],
         "exp_resid_q50_q90_q95_q99_max":[float(x) for x in np.quantile(e,[.5,.9,.95,.99,1])],
         "exp_resid_mean":float(e.mean()),"top_1_contribution_pct":float(100*e.max()/e.sum()),
         "top_5_contribution_pct":float(100*np.sort(e)[-min(5,len(e)):].sum()/e.sum()),"residual_n":len(a)}
def main():
 start=time.time(); probe=json.loads((DIAG/"pt_saved_diag_probe.json").read_text())
 source_sha=fsha(SOURCE); raw=pd.read_parquet(SOURCE); d=raw.copy(); d["country"]="PT"
 for c in ("year","month","pm25"): d[c]=pd.to_numeric(d[c],errors="coerce")
 good=np.isfinite(pd.to_numeric(d.lat,errors="coerce"))&np.isfinite(pd.to_numeric(d.lon,errors="coerce"))&np.isfinite(d.pm25)&(d.pm25>=0)
 d=d.loc[good].copy(); d["year"]=d.year.astype(int); d["month"]=d.month.astype(int); d["site_id"]=d.site_id.astype(str)
 d.reset_index(drop=True,inplace=True); d["_row_id"]=np.arange(len(d),dtype=np.int64)
 expected=probe["profiles"][0]["source_manifest"]["PT"]
 source_ok=(source_sha==expected["feature_sha256"][0] and len(raw)==434 and len(d)==434 and rowsha(d)==expected["analysis_row_key_sha256"])
 out={"purpose":"Saved PT HGB smear/clip validation and bounded train-only residual-tail reproduction; diagnostic only.",
      "created_utc":pd.Timestamp.now(tz="UTC").isoformat(),"source":{"path":str(SOURCE),"sha256":source_sha,
       "raw_rows":len(raw),"analysis_rows":len(d),"duplicate_key_rows":int(d.duplicated(["country","site_id","year","month"],keep=False).sum()),
       "source_manifest_sha256_match":source_sha==expected["feature_sha256"][0],
       "analysis_row_key_sha256":rowsha(d),"analysis_row_key_sha256_match":rowsha(d)==expected["analysis_row_key_sha256"]},
      "code_runtime":{"run_validation_sha256":fsha(CODE/"run_validation.py"),"validation_core_sha256":fsha(CODE/"validation_core.py"),
       "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__},
      "profiles":{},"preflight_equivalence_gates_pass":False}
 contexts=[]; assign_hashes=[]
 for pr in probe["profiles"]:
  name=pr["profile"]; root=RESULTS/f"PT_strict_{name}_hgb"
  run=json.loads((root/"run_manifest.json").read_text()); sm=json.loads((root/"outer_fold_split_manifest.json").read_text())
  schemas=json.loads((root/"schema_manifest.json").read_text()); ass_path=root/"outer_fold_assignments.parquet"
  ass=pd.read_parquet(ass_path); oof=pd.read_parquet(root/"oof_predictions.parquet"); ah=fsha(ass_path); assign_hashes.append(ah)
  ac={"sha256":ah,"matches_run_manifest":ah==run["outer_fold_split_manifest"]["assignment_sha256"],
      "matches_probe":ah==pr["split_assignment_sha256"],"rows":len(ass),
      "role_values":sorted(ass.role.astype(str).unique().tolist()),"row_ids_match_source":True}
  src_idx=d.set_index("_row_id")
  chk=ass.copy(); chk["row_id"]=chk.row_id.astype(int)
  merged=chk.merge(src_idx[["site_id","year","month"]].reset_index(),left_on="row_id",right_on="_row_id",how="left",suffixes=("_a","_s"),validate="many_to_one")
  ac["row_ids_match_source"]=bool(merged.site_id_a.astype(str).eq(merged.site_id_s.astype(str)).all() and merged.year_a.astype(int).eq(merged.year_s.astype(int)).all() and merged.month_a.astype(int).eq(merged.month_s.astype(int)).all())
  features=schemas["PT:full"]["base_features"]; cfg=ModelConfig(**run["model_config"])
  mrec={(x["task"],str(x["fold_id"])):x for x in pr["source_manifest"]["PT"]["outer_fold_models"] if x.get("task") in ("T1_station","T3_year_2022")}
  srec={(x["task"],str(x["fold_id"])):x for x in sm["splits"] if x.get("country")=="PT" and x.get("task") in ("T1_station","T3_year_2022")}
  p={"model_config":run["model_config"],"saved_runtime":run.get("runtime"),"feature_count":len(features),
     "assignment_check":ac,"folds":{},"saved_oof":{},"equivalence_gates_pass":False}
  p_gate=bool(source_ok and ac["matches_run_manifest"] and ac["matches_probe"] and ac["row_ids_match_source"])
  prepared=[]
  for (task,fold_id),r in sorted(srec.items(),key=lambda kv:(kv[0][0],int(kv[0][1]))):
   mr=mrec[(task,fold_id)]; sub=ass[(ass.country.astype(str)=="PT")&(ass.task.astype(str)==task)&(ass.fold_id.astype(str)==fold_id)&(ass.role.astype(str)=="test")]
   ids=sorted(sub.row_id.astype(int).tolist())
   if r["test_ids_unit"]=="site_id":
    test=d[d.site_id.isin(set(map(str,r["test_ids"])))].copy(); train=d[d.site_id.isin(set(map(str,r["train_ids"])))].copy()
    units_ok=set(sub.site_id.astype(str))==set(map(str,r["test_ids"]))
   else:
    test=d[d._row_id.isin(set(map(int,r["test_ids"])))].copy(); train=d[d._row_id.isin(set(map(int,r["train_ids"])))].copy()
    units_ok=set(ids)==set(map(int,r["test_ids"]))
   assignment_ok=sorted(test._row_id.astype(int))==ids and units_ok
   trs=asha(np.sort(train._row_id.to_numpy(dtype=np.int64))); tes=asha(np.sort(test._row_id.to_numpy(dtype=np.int64)))
   trk=rowsha(train); tek=rowsha(test); tfp=trainfp(train,features)
   row_ok=bool(assignment_ok and trs==r["train_row_ids_sha256"] and tes==r["test_row_ids_sha256"] and trk==r["train_row_key_sha256"] and tek==r["test_row_key_sha256"] and rowsha(d)==r["analysis_row_key_sha256"])
   fp_ok=tfp==mr["training_support_fingerprint"]; config_ok=run["model_config"]==mr["config"]
   fold_result={"train_rows":len(train),"train_stations":int(train.site_id.nunique()),"test_rows":len(test),
         "assignment_matches_manifest":assignment_ok,"train_row_ids_sha256_match":trs==r["train_row_ids_sha256"],
         "test_row_ids_sha256_match":tes==r["test_row_ids_sha256"],"train_row_keys_sha256_match":trk==r["train_row_key_sha256"],
         "test_row_keys_sha256_match":tek==r["test_row_key_sha256"],"training_support_fingerprint":tfp,
         "training_support_fingerprint_match":fp_ok,"configuration_matches_fold_manifest":config_ok,
         "saved_smear":float(mr["smear"])}
   p["folds"][f"{task}/{fold_id}"]=fold_result; p_gate=bool(p_gate and row_ok and fp_ok and config_ok)
   prepared.append((task,fold_id,train,schemas["PT:full"],cfg,mr,fold_result))
  s=oof[(oof.country.astype(str)=="PT")&(oof.arm.astype(str)=="full")&(oof.target_strategy.astype(str)=="log_smear")&
        (oof.smear_strategy.astype(str)=="crossfit_station")&oof.task.astype(str).isin(["T1_station","T3_year_2022"])]
  ratio_ok=True
  for (task,fold),g in s.groupby(["task","fold_id"],sort=True):
   sm=g[g.backtransform.astype(str)=="smearing"]; orig=g[g.backtransform.astype(str)=="original"]
   key=["country","task","fold_id","row_id","site_id","year","month"]
   pair=sm.merge(orig,on=key,suffixes=("_smear","_original"),validate="one_to_one")
   pos=(pair.pred_smear>0)&(pair.pred_original>0)
   ratio=(pair.loc[pos,"pred_smear"].to_numpy(float)+.1)/(pair.loc[pos,"pred_original"].to_numpy(float)+.1)
   factors=sorted(set(float(x) for x in pair.smear_smear.dropna().unique()))
   delta=float(np.max(np.abs(ratio-factors[0]))) if len(ratio) and len(factors)==1 else None
   mr=mrec[(str(task),str(fold))]; ratio_fold_ok=bool(len(pair)>0 and len(factors)==1 and factors[0]==float(mr["smear"]) and delta is not None and delta<=1e-8)
   ratio_ok=bool(ratio_ok and ratio_fold_ok)
   p["saved_oof"][f"{task}/{fold}"]={"smearing_rows":len(sm),"original_rows":len(orig),"paired_rows":len(pair),
      "positive_both_rows":int(pos.sum()),"smearing_clipped_zero_rows":int((pair.pred_smear==0).sum()),
      "original_clipped_zero_rows":int((pair.pred_original==0).sum()),"stored_smear_values":factors,
      "matches_saved_fold_model_smear":len(factors)==1 and factors[0]==float(mr["smear"]),
      "max_adjusted_ratio_minus_factor":delta,"ratio_check_pass":ratio_fold_ok}
  p["saved_oof_ratio_clip_checks_pass"]=ratio_ok
  p_gate=bool(p_gate and ratio_ok); p["equivalence_gates_pass"]=p_gate
  out["profiles"][name]=p; contexts.append((name,prepared,p))
 out["assignment_sha256_identical_across_profiles"]=len(set(assign_hashes))==1
 gates=bool(source_ok and len(set(assign_hashes))==1 and all(x[2]["equivalence_gates_pass"] for x in contexts))
 out["preflight_equivalence_gates_pass"]=gates
 if gates:
  out["fit_scope"]={"profiles":["no_met","with_met"],"tasks":["T1_station","T3_year_2022"],"arm":"full",
    "method":"Recompute the exact 3-split station-group OOF training residuals used for each saved smear; no outer test targets enter.",
    "fit_count":sum(3 for _,prep,_ in contexts for _ in prep)}
  for name,prepared,p in contexts:
   for task,fold_id,train,schema,cfg,mr,foldout in prepared:
    X=feature_matrix(train,schema); z=np.log(train.pm25.to_numpy(dtype=np.float64)+.1); groups=train.site_id.astype(str).to_numpy()
    resid=np.full(len(train),np.nan)
    for ti,ci in GroupKFold(n_splits=3).split(X,groups=groups):
     est=estimator(cfg); est.fit(X[ti],z[ti]); resid[ci]=z[ci]-est.predict(X[ci])
    if not np.isfinite(resid).all(): raise ValueError(f"nonfinite residuals {name} {task}/{fold_id}")
    tail=arr(resid); delta=tail["exp_resid_mean"]-float(mr["smear"])
    foldout["recomputed_training_only_residual_tail"]=tail
    foldout["recomputed_smear"]=tail["exp_resid_mean"]
    foldout["smear_abs_difference"]=abs(delta)
    foldout["smear_reproduction_match_1e-8"]=abs(delta)<=1e-8
  out["tail_reproduction_pass"]=all(f["smear_reproduction_match_1e-8"] for p in out["profiles"].values() for f in p["folds"].values())
 else:
  out["fit_scope"]={"status":"not_run","reason":"One or more source/split/fingerprint/config/saved-Oof algebra gates failed."}
 out["elapsed_seconds"]=time.time()-start
 (DIAG/"actual_evidence.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
 print(json.dumps({"status":"complete","output":str(DIAG/"actual_evidence.json"),
    "preflight_equivalence_gates_pass":gates,"tail_reproduction_pass":out.get("tail_reproduction_pass"),
    "elapsed_seconds":out["elapsed_seconds"]}))
if __name__=="__main__": main()

