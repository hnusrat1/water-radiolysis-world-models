"""Draw manuscript figures 1-4 from the archived display data. No model evaluation."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / 'plot_inputs.json').read_text(encoding='utf-8'))
ASSETS = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

COLORS = {'counts': '#606870', 'refit_compact32_active': '#3374a7', 'refit_compact64_active': '#bd5064',
          'refit_moment_learned': '#177757', 'moment_physics': '#997c21'}
NAMES = {'counts': 'Counts only', 'refit_compact32_active': 'C32', 'refit_compact64_active': 'C64',
         'refit_moment_learned': 'Learned moments', 'moment_physics': 'Physical moments'}
INK = '#203035'
GRAY = '#52636a'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.4, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'axes.spines.right': False, 'axes.spines.top': False,
                     'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
                     'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.edgecolor': GRAY, 'axes.titlecolor': INK})


def save(fig, name, title):
    fig.savefig(ASSETS / f'{name}.pdf', metadata={'Title': title, 'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def source_state():
    geometry = DATA['source_geometry']
    row = geometry['row']
    assert (row['case_id'], row['family_id'], row['action']) == (6009605, 60096, 5)

    fig = plt.figure(figsize=(7.12, 3.02))
    fig.text(.006, .946, 'A   Observed and incoming clouds', fontsize=9.4, weight='bold')
    fig.text(.49, .946, 'B   Two encodings of the initial cloud', fontsize=9.4, weight='bold')
    ax = fig.add_axes([.084, .425, .367, .394])
    for key, marker, color, label, size in [('incoming', '+', '#bd6525', 'Incoming at 10 ns', 10.5),
                                            ('initial', 'o', '#286580', 'Initial at 0.02 ns', 5.4)]:
        xz = np.asarray(row[key])
        style = dict(color=color, linewidths=.58) if marker == '+' else dict(facecolors='none', edgecolors=color, linewidths=.50)
        ax.scatter(xz[:, 0], xz[:, 1], s=size, marker=marker, alpha=.85, label=label, **style)
    ax.set(xlim=geometry['xlim'], ylim=geometry['zlim'], aspect='equal', xticks=[-50, 0, 50],
           yticks=[-30, 0, 20], xlabel='x (nm)', ylabel='z (nm)')
    ax.grid(alpha=.13, lw=.6)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc='center', bbox_to_anchor=(.240,.845), ncol=2, frameon=False,
               handlelength=.96, handletextpad=.25, columnspacing=1., fontsize=7.5, markerscale=1.5)

    # Schematic glyphs illustrate the geometry supplied to the encounter factor. They are not fitted values.
    for ry, method, details in [(.750, 'Learned moments', 'Species centers and covariances\nset the overlap of each reacting pair.'),
                               (.435, 'Pooled C64', 'The first pooled tensor sets a shared\nisotropic scale for encounters.')]:
        icon = fig.add_axes([.49, ry-.055, .076, .16])
        icon.set(xlim=(-1.35,1.35), ylim=(-1.20,1.20), aspect='equal')
        icon.axis('off')
        if method == 'Learned moments':
            for xy, wide, high, angle, color in [((-.25,.16),1.80,1.03,30,'#417d71'),((.27,-.12),1.70,.78,-20,'#87608d')]:
                icon.add_patch(Ellipse(xy,wide,high,angle=angle,facecolor=color,edgecolor='none',alpha=.12))
                icon.add_patch(Ellipse(xy,wide,high,angle=angle,facecolor='none',edgecolor=color,lw=1.0))
                icon.plot(*xy,'o',color=color,ms=2.0)
        else:
            icon.add_patch(Circle((0,0),.90,facecolor='#e3e9ec',edgecolor='#586871',lw=1))
            icon.plot([0,.64],[0,.64],color='#586871',lw=.9)
            icon.plot(0,0,'o',color='#586871',ms=2.1)
        fig.text(.581,ry+.050,method,fontsize=9.1,weight='bold')
        fig.text(.581,ry+.005,details,va='top',fontsize=8.2,linespacing=1.30)

    fig.add_artist(Line2D([.006,.994],[.265,.265],transform=fig.transFigure,color='#b4bfc1',lw=.65))
    fig.text(.006,.177,'C   From either state',fontsize=9.1,weight='bold')
    fig.text(.257,.180,'Evolve to 10 ns',fontsize=8.4)
    fig.text(.481,.204,'Continue without addition',fontsize=8.0)
    fig.text(.481,.071,'Add incoming molecules',fontsize=8.0)
    fig.text(.815,.180,'Response at 20 ns',fontsize=8.4,weight='bold')
    fig.text(.815,.070,'added − passive',fontsize=8.2)
    connector=fig.add_axes([0,0,1,1],zorder=-1)
    connector.axis('off')
    for x,y in [([.399,.426,.458],[.182,.182,.220]),([.426,.458],[.182,.086]),
                ([.723,.749,.781],[.220,.182,.182]),([.723,.749],[.086,.182])]:
        connector.plot(x,y,transform=fig.transFigure,color='#779098',lw=.9)
    save(fig,'source_state_query_schematic','Observed molecular cloud, compact state and response to a delayed addition')


def transfer():
    rows = DATA['transfer']
    splits = [('energy100','100 eV'),('energy200_control','200 eV (control)'),('energy400','400 eV')]
    fig, axes = plt.subplots(1,3,figsize=(7.10,2.40),sharey=True,sharex=True)
    fig.subplots_adjust(left=.163,right=.993,top=.775,bottom=.240,wspace=.11)
    fig.text(.015,.934,'Source transfer at a fixed training budget',weight='bold',fontsize=9.3)
    for ax,(split,title) in zip(axes,splits):
        values = [r for r in rows if r['split']==split]
        assert [r['group'] for r in values] == list(NAMES)
        for yi,row in enumerate(values):
            key=row['group']
            point=row['interval']['mean']
            lo,hi=row['interval']['ci95']
            ax.hlines(yi,lo,hi,color=COLORS[key],lw=1.25)
            ax.vlines([lo,hi],yi-.072,yi+.072,color=COLORS[key],lw=.7)
            ax.plot(point,yi,'o',mfc=COLORS[key],mec='white',mew=.45,ms=5.0,zorder=3)
            fits=row['per_fit']
            if len(fits)>1:
                assert len(fits)==5
                ax.scatter(fits,np.full(len(fits), yi+.19),s=6,color=COLORS[key],alpha=.63,linewidths=0)
        ax.set_xscale('log')
        ax.set_title(title,fontsize=8.6,pad=4)
        ax.set(xlim=(.009,1.56),ylim=(4.53,-.48))
        ax.set_yticks(range(len(NAMES)),labels=list(NAMES.values()))
        ax.tick_params(axis='y',length=0,labelsize=8.1)
        ax.xaxis.set_major_locator(FixedLocator([.01,.1,1.]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='x',labelsize=7.9)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x',alpha=.19,lw=.6)
        ax.set_axisbelow(True)
    fig.text(.565,.045,'20 ns response NMSE against SBS reference means',ha='center',fontsize=8.5)
    save(fig,'fresh_transfer','Source transfer at a fixed training budget')


def representative():
    rows=DATA['representative']
    assert len(rows)==12
    methods=list(NAMES)[1:]
    species_labels={'OH':'OH','e_aq':r'$e^-_{\mathrm{aq}}$','H2O2':r'H$_2$O$_2$'}
    styles=['--','-.','-',':']
    fig,axes=plt.subplots(3,4,figsize=(7.12,3.90),sharex=True,sharey='row')
    fig.subplots_adjust(left=.105,right=.986,bottom=.135,top=.837,wspace=.22,hspace=.39)
    for r in rows:
        ax=axes[r['row'],r['column']]
        times=np.array(r['time'])
        assert r['family_id']==60096 and len(times)==11
        before=times<r['injection_ns']
        delta=np.array(r['native_mean']); se=np.array(r['native_MCSE'])
        for seg in [before,~before]:
            ax.fill_between(times[seg],(delta-se)[seg],(delta+se)[seg],color='.68',alpha=.52,lw=0)
            ax.plot(times[seg],delta[seg],color='#252829',lw=1.05,marker='.',ms=2.3,zorder=5)
            for name,ls in zip(methods,styles):
                ax.plot(times[seg],np.asarray(r['models'][name])[seg],ls,color=COLORS[name],
                        lw=1.05,marker='.',ms=1.9)
        ax.axvline(r['injection_ns'],color='.55',ls=':',lw=.65)
        ax.set_xscale('log')
        ax.grid(alpha=.15,lw=.5)
        ax.tick_params(labelsize=7.7,pad=2)
        if r['column']==0:
            ax.set_ylabel(species_labels[r['species']],weight='bold',fontsize=9.1)
        if r['row']==0:
            ax.set_title(fr"$\tau$ = {r['injection_ns']:g} ns, $r$ = {r['spacing_nm']:g} nm",fontsize=8.1)
    lines=[Line2D([],[],color='#252829',lw=1.15)]+[Line2D([],[],color=COLORS[g],ls=st,lw=1.3) for g,st in zip(methods,styles)]
    fig.legend(lines,['SBS mean']+[NAMES[g] for g in methods],ncol=5,loc='upper center',
               bbox_to_anchor=(.52,.997),fontsize=7.8,frameon=False,columnspacing=1.13,handlelength=1.9,
               handletextpad=.38)
    fig.text(.008,.52,'Added − passive (molecules)',rotation='vertical',va='center',fontsize=8.4)
    fig.text(.545,.038,'Time (ns)',ha='center',fontsize=8.5)
    save(fig,'representative_B400_layout','Intervention response for four actions in a fixed 400 eV source family')


def accuracy_cost():
    rows = DATA['accuracy_cost']
    fit_summary = DATA['fit_inset']
    fits = dict(zip(fit_summary['seeds'], fit_summary['fits'], strict=True))
    low, high = fit_summary['min_max']
    avg = fit_summary['mean']
    assert (min(fits.values()),max(fits.values()))==(low,high) and np.isclose(np.mean(list(fits.values())),avg)
    fig,ax=plt.subplots(figsize=(7.1,3.37))
    fig.subplots_adjust(left=.089,right=.974,bottom=.170,top=.954)
    ax.set(xscale='log',xlim=(.94,103),ylim=(-.025,1.48))
    ax.set_xlabel('Complete worker CPU for 32 paired cases (s)')
    ax.set_ylabel('20 ns response NMSE')
    ax.xaxis.set_major_locator(FixedLocator([1,2,4,8,16,32,64]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=.18,lw=.55)
    ax.set_axisbelow(True)
    for row in rows:
        key=row['group']; native=bool(row['R_per_arm'])
        color=COLORS.get(key,'#505354')
        x=float(row['CPU_median']); y=float(row['MSE_mean'])
        ax.hlines(y,float(row['CPU_q25']),float(row['CPU_q75']),color=color,lw=1.1)
        ax.vlines(x,float(row['CI_low']),float(row['CI_high']),color=color,lw=1.)
        ax.plot(x,y,'s' if native else 'o',color=color,ms=4.15)
        label=f'IRT {row["R_per_arm"]}' if native else NAMES[key]
        offset = (4,3); ha='left'
        if key=='refit_compact32_active': offset=(-4,-7);ha='right'
        if key=='refit_moment_learned':
            assert y==fits[1701] and int(row['timing_cases'])==32
            offset=(5,-4)
        if key=='paired_IRT_R64': offset=(0,6);ha='center'
        ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',ha=ha,fontsize=7.55)
    inset=ax.inset_axes([.458,.542,.535,.44])
    inset.set(xlim=(.063,.121),ylim=(0,1),xticks=[],yticks=[])
    inset.set_facecolor('#f7faf8')
    for spine in inset.spines.values():
        spine.set_visible(True);spine.set_color('#b5c6bd');spine.set_linewidth(.63)
    inset.text(.03,.82,'Learned moments at 400 eV · five fits',transform=inset.transAxes,fontsize=8.1,weight='bold')
    mid=.35
    inset.hlines(mid,low,high,color=GRAY,lw=1)
    inset.vlines([low,high],mid-.06,mid+.055,color=GRAY,lw=.9)
    inset.plot(fits[1701],mid,'o',ms=4.1,mfc='white',mec=INK,mew=.9)
    inset.plot(avg,mid,'D',ms=4.5,color=COLORS['refit_moment_learned'])
    inset.annotate(f'Timed {fits[1701]:.4f}',(fits[1701],mid+.03),xytext=(.067,.565),textcoords='data',
                   ha='left',fontsize=7.3,arrowprops=dict(arrowstyle='-',lw=.6,color=GRAY,shrinkA=1,shrinkB=2))
    inset.annotate(f'Mean {avg:.4f}',(avg,mid+.03),xytext=(.091,.565),textcoords='data',
                   ha='left',fontsize=7.3,color=COLORS['refit_moment_learned'],
                   arrowprops=dict(arrowstyle='-',lw=.6,color=COLORS['refit_moment_learned'],shrinkA=1,shrinkB=2))
    inset.text(low,.09,f'Min {low:.4f}',ha='left',fontsize=7.3)
    inset.text(high,.09,f'Max {high:.4f}',ha='right',fontsize=7.3)
    save(fig,'fresh_IRT_cost_profile_with_fit_range','Conditional forecast error and measured CPU at 400 eV')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('rendered'), help='target directory for four PDFs')
    ASSETS = parser.parse_args().output_dir
    ASSETS.mkdir(parents=True, exist_ok=True)
    source_state()
    transfer()
    representative()
    accuracy_cost()
