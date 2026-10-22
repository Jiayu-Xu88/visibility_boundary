#!/usr/bin/env python3
from pathlib import Path
import os
import json,hashlib,csv
import numpy as np,pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from PIL import Image
REPO_ROOT=Path(__file__).resolve().parents[1]; PROJECT_ROOT=Path(os.environ.get("PROJECT_ROOT",REPO_ROOT)).resolve(); R=REPO_ROOT/"results"; OUT=Path(os.environ.get("FIGURE_OUTPUT_DIR",REPO_ROOT/"generated_figures")).resolve(); OUT.mkdir(parents=True,exist_ok=True)
def apply_figure_style(*,frame="open",font=None,sizes=(8,7,6),grid=False):
 base,secondary,tick=sizes
 mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":[font or "DejaVu Sans","DejaVu Sans"],"font.size":base,"axes.labelsize":base,"axes.titlesize":base,"legend.fontsize":secondary,"xtick.labelsize":tick,"ytick.labelsize":tick,"axes.linewidth":0.6,"xtick.direction":"out","ytick.direction":"out","xtick.major.size":3,"ytick.major.size":3,"xtick.major.width":0.6,"ytick.major.width":0.6,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":bool(grid),"legend.frameon":False,"figure.dpi":200,"savefig.dpi":300,"savefig.bbox":"tight","pdf.fonttype":42,"ps.fonttype":42})
apply_figure_style(frame="open",font="Arial",sizes=(8,7,6),grid=False)
mpl.rcParams.update({"savefig.dpi":600,"savefig.pad_inches":0.02,"svg.fonttype":"none"})
MM=1/25.4; W=178*MM; BLUE="#2C6FBB"; NAVY="#174A7E"; ORANGE="#D97904"; GREY="#858585"; LIGHT="#D8D8D8"; INK="#1A1A1A"; PALE="#DCE9F5"
J=lambda n:json.loads((R/n).read_text())
e2c=J("e2c_mil_projection_manifest.json"); audit=J("e2_representation_audit_manifest.json"); lad=J("e1_ladder_manifest.json")["levels"]; cv=J("m4_internal_external_cv_manifest.json"); surgen=J("surgen_state0_bootstrap_ci.json")["by_label_view"]["dmmr_msi_composite"]["auroc"]; assoc=pd.read_csv(R/"gi_morphology_state_associations.tsv",sep="\t")
EX=PROJECT_ROOT/"Experiment/analysis/medical_morphology_atlas/state_exemplars"
def clean(ax,left=True):
 ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.spines["left"].set_visible(left)
 if not left: ax.tick_params(axis="y",length=0)
def lab(ax,s,x=-.12,y=1.03): ax.text(x,y,s,transform=ax.transAxes,fontweight="bold",fontsize=9,ha="left",va="bottom")
def save(fig,n):
 fig.savefig(OUT/f"{n}.pdf",dpi=600); fig.savefig(OUT/f"{n}.png",dpi=600); fig.savefig(OUT/f"{n}.tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"}); plt.close(fig)
def box(ax,x,y,w,h,t,ec=INK,fc="white",bold=False,fs=6.8):
 ax.add_patch(Rectangle((x,y),w,h,ec=ec,fc=fc,lw=.8)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal",linespacing=1.3)
def arrow(ax,x1,y1,x2,y2,c=INK): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=7,lw=.8,color=c))
fig=plt.figure(figsize=(W,92*MM)); gs=fig.add_gridspec(2,1,height_ratios=[.92,1.08],hspace=.38,left=.08,right=.97,top=.95,bottom=.13); a=fig.add_subplot(gs[0]); b=fig.add_subplot(gs[1]); a.set_axis_off(); a.set_xlim(0,1); a.set_ylim(0,1)
box(a,.01,.34,.18,.38,"CRC H&E\n534 patients\n71 MSI-H",fc="#F7F7F7"); box(a,.25,.34,.20,.38,"frozen UNI\npatch features",ec=NAVY,fc=PALE,bold=True); box(a,.52,.58,.20,.28,"supervised\nMIL score",ec=BLUE); box(a,.52,.12,.20,.28,"patient-level\nrepresentations",ec=ORANGE); box(a,.80,.58,.19,.28,"gastric MSI\nAUROC 0.727",ec=BLUE,fc=PALE,bold=True); box(a,.80,.12,.19,.28,"localise shared\nMSI component",ec=ORANGE,fc="#FBEBD8",bold=True)
arrow(a,.19,.53,.25,.53,NAVY); arrow(a,.45,.53,.52,.72,BLUE); arrow(a,.45,.53,.52,.26,ORANGE); arrow(a,.72,.72,.80,.72,BLUE); arrow(a,.72,.26,.80,.26,ORANGE); a.text(.62,.94,"prediction",color=BLUE,ha="center",fontweight="bold",fontsize=7); a.text(.62,.02,"representation analysis",color=ORANGE,ha="center",fontweight="bold",fontsize=7); lab(a,"a",x=-.075,y=1.02)
rows=[("CRC, within lineage",.940,.914,.966,NAVY),("gastric, within lineage",.849,.790,.903,GREY),("CRC → gastric",e2c["crc_trained_mil_on_stad_auroc"],*e2c["crc_trained_mil_on_stad_boot95"],BLUE)]
for y,(name,v,lo,hi,c0) in zip(np.arange(3)[::-1],rows): b.plot([lo,hi],[y,y],color=c0,lw=1.5); b.plot(v,y,"o",ms=6,mfc=c0,mec="white",mew=.7); b.text(hi+.008,y,f"{v:.3f}",ha="left",va="center",fontsize=7,color=c0,fontweight="bold" if c0==BLUE else "normal")
b.axvline(.5,color=GREY,lw=.7,ls=(0,(2,2))); b.set_yticks(np.arange(3)[::-1]); b.set_yticklabels([r[0] for r in rows]); b.set_xlim(.48,1.01); b.set_ylim(-.55,2.55); b.set_xlabel("MSI discrimination (AUROC)"); b.text(.485,-.48,"chance",fontsize=6,color=GREY); b.text(.99,-.38,"CRC interval: mean ± SD; others: 95% CI",fontsize=6,color=GREY,ha="right"); clean(b,left=False); lab(b,"b"); save(fig,"Figure_1")
fig=plt.figure(figsize=(W,102*MM))
gs=fig.add_gridspec(2,2,height_ratios=[1,1.02],hspace=.58,wspace=.42,left=.075,right=.97,top=.95,bottom=.12)
a=fig.add_subplot(gs[0,0]); b=fig.add_subplot(gs[0,1]); c=fig.add_subplot(gs[1,:])
E=audit["E2a"]; flo=E["cross"]["floor_2p5_97p5"]
vals=[E["CRC"]["splithalf_cos_median"],E["STAD"]["splithalf_cos_median"],E["cross"]["cos_wSTAD_wCRC"]]
a.axhspan(flo[0],flo[1],color="#EEEEEE")
for i,(v,c0) in enumerate(zip(vals,[GREY,GREY,BLUE])):
 a.bar(i,v,width=.60,color=c0); a.text(i,v+.022,f"{v:.3f}",ha="center",fontsize=7,color=c0,fontweight="bold" if i==2 else "normal")
a.axhline(0,color=INK,lw=.7); a.set_xticks(range(3)); a.set_xticklabels(["CRC\nsplit-half","gastric\nsplit-half","cross-organ"]); a.set_ylabel("cosine similarity"); a.set_ylim(-.23,.72)
a.text(2.45,np.mean(flo),"permutation\n95% interval",fontsize=6,color=GREY,ha="left",va="center")
a.text(2,E["cross"]["cos_wSTAD_wCRC"]*.43,f'p = {E["cross"]["floor_p"]:.3f}',fontsize=6.5,color="white",fontweight="bold",ha="center",va="center"); clean(a); lab(a,"a")
names=["CRC nested CV","gastric nested CV","gastric → CRC","CRC → gastric"]
vs=[E["CRC"]["within_cv_auroc"],E["STAD"]["within_cv_auroc"],E["cross"]["transfer_auroc_STAD_to_CRC"],E["cross"]["transfer_auroc_CRC_to_STAD"]]
for y,n,v,c0 in zip(np.arange(4)[::-1],names,vs,[GREY,GREY,BLUE,BLUE]):
 b.plot([.5,v],[y,y],color=LIGHT,lw=1.1); b.plot(v,y,"o",ms=6,mfc=c0,mec="white",mew=.7); b.text(v+.012,y,f"{v+1e-12:.3f}",va="center",fontsize=7,color=c0)
b.axvline(.5,color=GREY,lw=.7,ls=(0,(2,2))); b.set_yticks(np.arange(4)[::-1]); b.set_yticklabels(names); b.set_xlim(.48,.91); b.set_ylim(-.55,3.55); b.set_xlabel("MSI direction-score AUROC"); clean(b,left=False); lab(b,"b")
r=e2c["ridge_cv_r2_of_mil_logit"]; rn=["1,024D patient mean","K = 128 states","K = 64 states","K = 24 states"]; rv=[r["mean1024"],r["K128"],r["K64"],r["K24"]]; rc=[BLUE,"#8EB6D8","#A9A9A9",ORANGE]
c.bar(range(4),rv,color=rc,width=.68)
for x,v,c0 in zip(range(4),rv,rc): c.text(x,v+.025,f"{v:.3f}",ha="center",fontsize=7,color=c0,fontweight="bold" if x in (0,3) else "normal")
c.set_xticks(range(4)); c.set_xticklabels(rn); c.set_ylim(0,.91); c.set_ylabel("cross-validated $R^2$"); c.set_xlabel("representation used to reproduce the transferred model logit"); clean(c); lab(c,"c",x=-.055)
c.text(.995,.96,"model-score fidelity, not MSI-label discrimination",transform=c.transAxes,ha="right",va="top",fontsize=6.3,color=GREY)
save(fig,"Figure_2")
fig=plt.figure(figsize=(W,158*MM)); gs=fig.add_gridspec(3,2,height_ratios=[.72,.76,1.08],width_ratios=[1.12,1],hspace=.62,wspace=.40,left=.08,right=.97,top=.965,bottom=.075); a=fig.add_subplot(gs[0,:]); b=fig.add_subplot(gs[1,0]); d=fig.add_subplot(gs[1,1])
mat=[]; sig=[]
for key in ("CRC_MSI_H_vs_MSS","STAD_MSI_H_vs_MSS"):
 sub=assoc[assoc.analysis==key].sort_values("state"); mat.append(sub.cliffs_delta.to_numpy()); sig.append(sub.bh_fdr.to_numpy()<.10)
mat=np.vstack(mat); sig=np.vstack(sig); im=a.imshow(mat,aspect="auto",cmap="RdBu_r",vmin=-.45,vmax=.45); a.set_yticks([0,1]); a.set_yticklabels(["CRC","gastric"]); a.set_xticks(range(24)); a.set_xticklabels(range(24),fontsize=5.3); a.set_xlabel("morphology state")
for i in range(2):
 for j in range(24):
  if sig[i,j]: a.text(j,i,"*",ha="center",va="center",fontsize=6)
a.add_patch(Rectangle((-.5,-.5),1,2,fill=False,ec=INK,lw=1.2))
for sp in a.spines.values(): sp.set_visible(False)
cb=fig.colorbar(im,ax=a,fraction=.020,pad=.012); cb.outline.set_linewidth(.5); cb.set_label("Cliff's delta",fontsize=7); lab(a,"a",x=-.06)
for i,coh in enumerate(["CRC","STAD"]):
 rnd=cv[coh]["MSI_H"]["random_5fold_cv_auroc"]; loco=cv[coh]["MSI_H"]["loco_pooled_auroc"]; ci=cv[coh]["MSI_H"]["loco_boot95"]; c0=BLUE if coh=="STAD" else GREY; b.plot([i-.22,i+.22],[rnd,loco],"-o",color=c0,lw=1.2,ms=5); b.plot([i+.22,i+.22],ci,color=c0,lw=1.1)
b.plot(2.1,surgen["mean"],"D",ms=5,mfc="white",mec=INK); b.plot([2.1,2.1],[surgen["ci_low_2_5"],surgen["ci_high_97_5"]],color=INK,lw=1.1); b.axhline(.5,color=GREY,lw=.7,ls=(0,(2,2))); b.set_xticks([-.22,.22,.78,1.22,2.1]); b.set_xticklabels(["random","LOCO","random","LOCO","SurGen"],fontsize=6); b.set_ylim(.48,.79); b.set_ylabel("MSI AUROC"); b.set_xlabel("within-lineage and external validation"); b.text(0,1.03,"CRC",transform=b.get_xaxis_transform(),ha="center",fontsize=6.5); b.text(1,1.03,"gastric",transform=b.get_xaxis_transform(),ha="center",fontsize=6.5); clean(b); lab(b,"b")
msi1=lad["R3_MSI_H"]["transfer"]["CRC_to_STAD"]; msi2=lad["R3_MSI_H"]["transfer"]["STAD_to_CRC"]; muc1=lad["R2_mucinous_icdo"]["transfer"]["CRC_to_STAD"]; muc2=lad["R2_mucinous_icdo"]["transfer"]["STAD_to_CRC"]
for i,(mi,mu) in enumerate([(msi1,muc1),(msi2,muc2)]):
 for off,obj,c0,label0 in [(-.12,mi,ORANGE,"MSI"),(.12,mu,BLUE,"mucinous")]:
  lo,hi=obj["boot95"]; d.plot([i+off,i+off],[lo,hi],color=c0,lw=1.3); d.plot(i+off,obj["auroc"],"o",ms=6,mfc=c0,mec="white",mew=.7,label=label0 if i==0 else None); d.text(i+off,obj["auroc"]+.025,f'{obj["auroc"]:.3f}',ha="center",fontsize=6.5,color=c0)
d.axhline(.5,color=GREY,lw=.7,ls=(0,(2,2))); d.set_xticks([0,1]); d.set_xticklabels(["CRC → gastric","gastric → CRC"]); d.set_ylim(.43,.91); d.set_ylabel("cross-organ AUROC"); d.set_xlabel("same K = 24 state representation"); d.legend(loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,borderaxespad=0); clean(d); lab(d,"c")
axc=fig.add_subplot(gs[2,:]); axc.set_axis_off(); lab(axc,"d",x=-.06,y=1.05)
def choose(folder,k=4):
 out=[]
 for f in sorted(folder.glob("rank*.png")):
  im0=Image.open(f).convert("RGB")
  if np.asarray(im0.convert("L")).mean()<235: out.append(im0)
  if len(out)>=k: break
 return out
sets=[("CRC state 0\nMSI-H candidate pool",choose(EX/"CRC_MSI_state00")),("gastric state 0\nlabel-blind candidate pool",choose(EX/"STAD_state00_labelblind"))]; pos=axc.get_position(); gap=.012; labelw=.055; tw=(pos.width-labelw*2-gap*7)/8; th=tw*fig.get_size_inches()[0]/fig.get_size_inches()[1]; y=pos.y0+(pos.height-th)/2
for gi,(t,ims) in enumerate(sets):
 start=pos.x0+gi*(4*tw+3*gap+labelw)+labelw; fig.text(start-labelw+.005,y+th/2,t,ha="left",va="center",fontsize=7,fontweight="bold",rotation=90)
 for j,im0 in enumerate(ims):
  axi=fig.add_axes([start+j*(tw+gap),y,tw,th]); axi.imshow(im0); axi.set_xticks([]); axi.set_yticks([])
  for sp in axi.spines.values(): sp.set_linewidth(.6); sp.set_color(INK)
axc.text(.5,-.02,"CRC candidates were restricted to MSI-H; gastric candidates were label-blind. Both were centroid-ranked and not pathologist-adjudicated.",transform=axc.transAxes,ha="center",va="top",fontsize=6,color=GREY); save(fig,"Figure_3")
files=["e2c_mil_projection_manifest.json","e2_representation_audit_manifest.json","e1_ladder_manifest.json","m4_internal_external_cv_manifest.json","surgen_state0_bootstrap_ci.json","gi_morphology_state_associations.tsv"]
with open(R/"source_data_index.csv","w",newline="") as f:
 w=csv.writer(f); w.writerow(["figure","panel","source_path","sha256","role"])
 for fi,pa,fn,ro in [("F1","b",files[0],"supervised transfer"),("F2","a,b,d",files[1],"continuous space and ladder"),("F2","c",files[0],"score reconstruction"),("F3","a",files[5],"state associations"),("F3","b",files[3],"centre validation"),("F3","b",files[4],"SurGen validation"),("F3","d",files[2],"MSI and mucinous transfer")]:
  p=R/fn; w.writerow([fi,pa,str(p.relative_to(REPO_ROOT)),hashlib.sha256(p.read_bytes()).hexdigest(),ro])
with open(OUT/"color_registry.csv","w",newline="") as f:
 w=csv.writer(f); w.writerow(["semantic_role","hex","usage"]); w.writerows([["shared cross-organ or positive-control",BLUE,"F1-F3"],["within-lineage continuous",NAVY,"F1"],["morphology-state MSI",ORANGE,"F2-F3"],["neutral/reference",GREY,"all"]])
print("generated",OUT)
