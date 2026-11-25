"""Shared typography for the manuscript figure layouts (project code: MIT)."""
from pathlib import Path
import json,os

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.text import Text


def configure_style():
    font_dir = os.environ.get('PM25_FIGURE_FONT_DIR')
    if font_dir:
        for name in ('arial.ttf', 'arialbd.ttf'):
            path = Path(font_dir) / name
            if path.is_file():
                font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.weight': 'bold', 'font.size': 8.5,
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'axes.labelsize': 9.0, 'axes.titlesize': 10.0,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial:bold',
        'mathtext.it': 'Arial:bold', 'mathtext.bf': 'Arial:bold',
        'mathtext.sf': 'Arial:bold', 'mathtext.default': 'regular',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def style_figure(fig):
    """Keep authored positions/sizes while making every text artist legible."""
    for text in fig.findobj(match=Text):
        text.set_fontfamily('Arial')
        text.set_fontweight('bold')
        if text.get_fontsize() < 8.2:
            text.set_fontsize(8.2)
    fig.canvas.draw()


def write_layout_qa(fig, outdir, stem):
    """Record text bounds for inspection; never remove or reposition content."""
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();canvas=fig.bbox
    texts=[];outside=[];undrawn_ticks=set()
    # Locators retain tick artists beyond the view interval; Matplotlib does
    # not draw those ticks, so they are not clipping or collision defects.
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc()<lo-1e-10 or tick.get_loc()>hi+1e-10:
                    undrawn_ticks.update((id(tick.label1),id(tick.label2)))
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip() or id(artist) in undrawn_ticks:
            continue
        bbox=artist.get_window_extent(renderer)
        item={'text':artist.get_text(),'font_size_pt':artist.get_fontsize(),
              'weight':str(artist.get_fontweight()),'bbox_px':list(bbox.extents)}
        texts.append(item)
        if bbox.x0<canvas.x0-1 or bbox.y0<canvas.y0-1 or bbox.x1>canvas.x1+1 or bbox.y1>canvas.y1+1:
            outside.append(item)
    collisions=[]
    for i,a in enumerate(texts):
        x0,y0,x1,y1=a['bbox_px'];area=max(0,x1-x0)*max(0,y1-y0)
        if area==0:continue
        for b in texts[i+1:]:
            u0,v0,u1,v1=b['bbox_px'];other=max(0,u1-u0)*max(0,v1-v0)
            overlap=max(0,min(x1,u1)-max(x0,u0))*max(0,min(y1,v1)-max(y0,v0))
            if other and overlap/min(area,other)>.18:
                collisions.append({'first':a['text'],'second':b['text'],'overlap_fraction':overlap/min(area,other)})
    result={'figure':stem,'canvas_px':list(canvas.extents),'outside_text':outside,
            'collision_candidates':collisions,'text_artists':texts}
    (Path(outdir)/(stem+'_layout_qa.json')).write_text(json.dumps(result,indent=2)+'\n')


configure_style()
