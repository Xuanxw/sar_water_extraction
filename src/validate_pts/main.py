"""独立水体核验窗口：python src/validate_pts/main.py，无需QGIS。"""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import ImageTk

try:
    from .review_store import ReviewStore
    from .map_layers import map_bounds, render_local, render_tianditu, WORLD
except ImportError:
    from review_store import ReviewStore
    from map_layers import map_bounds, render_local, render_tianditu, WORLD

# 路径集中在入口；也可在窗口中打开其他统一核验GPKG及GeoTIFF。
DEFAULT_POINTS = Path(r'E:\InSAR\水体提取\北京密云区\outputs\manual_review\manual_review.gpkg')
TILE_CACHE = Path(r'E:\InSAR\水体提取\北京密云区\data_cache\validation_tiles')
DEFAULT_IMAGE = ''
DEFAULT_ZOOM = 17

LABELS = {None:'未核验', 0:'非水体', 1:'水体', -1:'无法确定'}


class PointReviewer:
    """左侧选点和填写，右侧地图；影像读取在工作线程，保存始终在主线程。"""
    def __init__(self, root, points_path=None):
        self.root=root;root.title('SAR水体 · 人工点位核验');root.geometry('1420x900');root.minsize(1100,760)
        self.store=None;self.current=None;self.visible=[];self.center=(0,0);self.zoom=DEFAULT_ZOOM
        self.executor=ThreadPoolExecutor(max_workers=1);self.future=None;self.generation=0
        self.photo=None;self.display_bounds=None;self.drag=None;self.closed=False;self.resize_job=None
        self.images=[];self.form_initial=None;self.loading=False
        self.scene=tk.StringVar(value='全部影像');self.filter=tk.StringVar(value='未核验')
        self.mode=tk.StringVar(value='本地影像');self.local=tk.StringVar(value=DEFAULT_IMAGE)
        self.bands=tk.StringVar(value='1,2,3');self.stretch=tk.BooleanVar(value=True)
        self.token=tk.StringVar(value=os.environ.get('TIANDITU_TOKEN',''))
        self.reviewer=tk.StringVar();self.reference=tk.StringVar();self.ref_date=tk.StringVar()
        self.confidence=tk.StringVar(value='中');self.auto_next=tk.BooleanVar(value=True)
        self.show_pred=tk.BooleanVar(value=False);self.status=tk.StringVar(value='打开核验点，并加载本地影像或填写天地图Key。')
        self.info=tk.StringVar();self.progress=tk.StringVar();self.image_notice=tk.StringVar(value='本地影像需带坐标系；天地图默认底图不能假定为SAR同日影像。')
        self.build_ui()
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.bind('<KeyPress>',self.key)
        root.after(120,self.poll)
        if points_path and Path(points_path).exists():self.load_points(points_path)

    def build_ui(self):
        style=ttk.Style();style.theme_use('vista' if 'vista' in style.theme_names() else 'clam')
        style.configure('TLabel',font=('Microsoft YaHei UI',10))
        bar=ttk.Frame(self.root,padding=8);bar.pack(fill='x')
        ttk.Button(bar,text='打开核验GPKG',command=self.choose_points).pack(side='left')
        ttk.Button(bar,text='导出核验CSV',command=self.export_csv).pack(side='left',padx=6)
        ttk.Label(bar,textvariable=self.progress).pack(side='left',padx=10)
        ttk.Button(bar,text='撤销上次保存',command=self.undo).pack(side='right')
        body=ttk.Panedwindow(self.root,orient='horizontal');body.pack(fill='both',expand=True)
        left=ttk.Frame(body,padding=10,width=370);right=ttk.Frame(body,padding=(0,8,8,8))
        body.add(left,weight=0);body.add(right,weight=1)
        self.root.after(100,lambda:body.sashpos(0,380))
        ttk.Label(left,text='场景（按SAR日期）').pack(anchor='w')
        self.scene_combo=ttk.Combobox(left,textvariable=self.scene,state='readonly',width=34)
        self.scene_combo.pack(fill='x');self.scene_combo.bind('<<ComboboxSelected>>',lambda e:self.refresh_list())
        line=ttk.Frame(left);line.pack(fill='x',pady=6)
        self.filter_combo=ttk.Combobox(line,textvariable=self.filter,values=['未核验','全部','无法确定','已确定'],state='readonly',width=10)
        self.filter_combo.pack(side='left');self.filter_combo.bind('<<ComboboxSelected>>',lambda e:self.refresh_list())
        ttk.Checkbutton(line,text='保存后下一点',variable=self.auto_next).pack(side='left',padx=8)
        tree_frame=ttk.Frame(left);tree_frame.pack(fill='both',expand=True)
        self.tree=ttk.Treeview(tree_frame,columns=('number','pixel','label'),show='headings',height=6,selectmode='browse')
        for key,title,width in [('number','序号',50),('pixel','点位行/列',170),('label','参考类别',90)]:
            self.tree.heading(key,text=title);self.tree.column(key,width=width,stretch=key=='pixel')
        scroll=ttk.Scrollbar(tree_frame,orient='vertical',command=self.tree.yview);self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left',fill='both',expand=True);scroll.pack(side='right',fill='y')
        self.tree.bind('<<TreeviewSelect>>',self.on_select)
        nav=ttk.Frame(left);nav.pack(fill='x',pady=5)
        ttk.Button(nav,text='上一点 ←',command=lambda:self.move(-1)).pack(side='left')
        ttk.Button(nav,text='下一点 →',command=lambda:self.move(1)).pack(side='left',padx=4)
        ttk.Button(nav,text='定位点',command=self.locate).pack(side='right')
        ttk.Label(left,textvariable=self.info,wraplength=340,justify='left').pack(fill='x',pady=5)
        ttk.Checkbutton(left,text='显示预测类别（建议独立判读后再看）',variable=self.show_pred,command=self.update_info).pack(anchor='w')
        for title,var in [('核验人',self.reviewer),('参考资料',self.reference),('参考影像日期（未知可留空）',self.ref_date)]:
            ttk.Label(left,text=title).pack(anchor='w',pady=(5,0));ttk.Entry(left,textvariable=var).pack(fill='x')
        row=ttk.Frame(left);row.pack(fill='x',pady=5)
        ttk.Label(row,text='判读置信度').pack(side='left')
        ttk.Combobox(row,textvariable=self.confidence,values=['高','中','低'],state='readonly',width=6).pack(side='left',padx=8)
        ttk.Label(left,text='备注（时相差、阴影、混合像元等）').pack(anchor='w')
        self.note=tk.Text(left,height=2,width=34,wrap='word',font=('Microsoft YaHei UI',10));self.note.pack(fill='x')
        controls=ttk.Frame(left);controls.pack(fill='x',pady=8)
        for text,value in [('1 水体',1),('0 非水体',0),('U 无法确定',-1)]:
            ttk.Button(controls,text=text,command=lambda v=value:self.save(v)).pack(side='left',expand=True,fill='x')
        ttk.Button(left,text='清空参考类别（恢复未核验）',command=lambda:self.save(None,advance=False)).pack(fill='x')
        source=ttk.Frame(right);source.pack(fill='x')
        for name in ('本地影像','天地图'):
            ttk.Radiobutton(source,text=name,value=name,variable=self.mode,command=self.source_changed).pack(side='left')
        ttk.Button(source,text='添加光学影像',command=self.choose_images).pack(side='left',padx=7)
        ttk.Button(source,text='放大 +',command=lambda:self.change_zoom(1)).pack(side='right')
        ttk.Button(source,text='缩小 −',command=lambda:self.change_zoom(-1)).pack(side='right')
        self.image_combo=ttk.Combobox(right,textvariable=self.local,state='readonly')
        self.image_combo.pack(fill='x',pady=5);self.image_combo.bind('<<ComboboxSelected>>',lambda e:self.source_changed())
        opts=ttk.Frame(right);opts.pack(fill='x')
        ttk.Label(opts,text='RGB波段').pack(side='left');ttk.Entry(opts,textvariable=self.bands,width=8).pack(side='left',padx=4)
        ttk.Checkbutton(opts,text='显示P2–P98拉伸',variable=self.stretch,command=self.request_render).pack(side='left')
        ttk.Label(opts,text='天地图Key').pack(side='left',padx=(10,3));ttk.Entry(opts,textvariable=self.token,show='*',width=22).pack(side='left',fill='x',expand=True)
        ttk.Button(opts,text='应用 / 刷新',command=self.source_changed).pack(side='right',padx=4)
        self.canvas=tk.Canvas(right,bg='#262c34',highlightthickness=0,cursor='crosshair')
        self.canvas.pack(fill='both',expand=True,pady=6)
        self.canvas.bind('<Configure>',self.resized)
        self.canvas.bind('<MouseWheel>',lambda e:self.change_zoom(1 if e.delta>0 else -1))
        self.canvas.bind('<ButtonPress-1>',self.begin_drag)
        self.canvas.bind('<ButtonRelease-1>',self.pan)
        ttk.Label(right,textvariable=self.image_notice,wraplength=850).pack(fill='x')
        ttk.Label(self.root,textvariable=self.status,padding=8,wraplength=1350).pack(fill='x')

    def choose_points(self):
        path=filedialog.askopenfilename(title='选择统一核验点',filetypes=[('GeoPackage','*.gpkg')])
        if path:self.load_points(path)

    def load_points(self,path):
        try:
            if not self.flush_edits() and not messagebox.askyesno('重新加载','是否放弃本窗口未保存改动并重新读取文件？'):return
            self.store=ReviewStore(path);self.current=None;self.form_initial=None
            self.scene.set('全部影像')
            self.scene_combo['values']=['全部影像']+sorted({p['scene_id'] for p in self.store.points})
            saved={}
            session=self.store.path.with_name(self.store.path.stem+'.viewer.json')
            if session.exists():
                saved=json.loads(session.read_text(encoding='utf8'))
                self.reviewer.set(saved.get('reviewer',''))
                self.images=[p for p in saved.get('images',[]) if Path(p).is_file()]
                self.image_combo['values']=self.images
                if self.images:self.local.set(saved.get('image') if saved.get('image') in self.images else self.images[0])
                self.ref_date.set(saved.get('reference_date',''))
                self.zoom=int(saved.get('zoom',DEFAULT_ZOOM))
            if saved.get('scene') in self.scene_combo['values']:self.scene.set(saved['scene'])
            if saved.get('filter') in ('未核验','全部','无法确定','已确定'):self.filter.set(saved['filter'])
            self.refresh_list(prefer=saved.get('point_id'))
            self.status.set(f'已加载{len(self.store.points)}点。原几何和抽样权重不会被修改。')
        except Exception as exc:messagebox.showerror('无法加载',str(exc))

    def refresh_list(self,prefer=None):
        if not self.store or not self.flush_edits():return
        self.loading=True
        points=self.store.points
        scene=self.scene.get();mode=self.filter.get()
        self.visible=[p for p in points if (scene=='全部影像' or p['scene_id']==scene) and
                      (mode=='全部' or mode=='未核验' and p['ref_class'] is None or
                       mode=='无法确定' and p['ref_class']==-1 or mode=='已确定' and p['ref_class'] in (0,1))]
        self.tree.delete(*self.tree.get_children())
        for i,p in enumerate(self.visible,1):
            self.tree.insert('', 'end',iid=p['point_id'],values=(i,f"{p.get('row','')}/{p.get('col','')}",LABELS[p['ref_class']]))
        known=sum(p['ref_class'] in (0,1) for p in points);uncertain=sum(p['ref_class']==-1 for p in points)
        self.progress.set(f'已确定 {known}/{len(points)} | 无法确定 {uncertain} | 当前列表 {len(self.visible)}')
        self.loading=False
        if self.visible:
            key=prefer if prefer in {p['point_id'] for p in self.visible} else self.visible[0]['point_id']
            self.tree.selection_set(key);self.tree.see(key);self.select_point(key)
        else:
            self.current=None;self.form_initial=None;self.info.set('当前筛选下没有点，请切换场景或状态。')
            self.canvas.delete('all');self.generation+=1

    def on_select(self,event=None):
        selected=self.tree.selection()
        if not self.loading and selected:self.select_point(selected[0])

    def select_point(self,key):
        if self.current and self.current['point_id']==key:return
        if not self.flush_edits():return
        self.current=self.store.by_id[key];p=self.current
        self.confidence.set(p['confidence'] or '中')
        self.note.delete('1.0','end');self.note.insert('1.0',p['note'] or '')
        if p['reviewer']:self.reviewer.set(p['reviewer'])
        if p['reference']:
            parts=p['reference'].split(' | 影像日期=',1)
            self.reference.set(parts[0]);self.ref_date.set(parts[1] if len(parts)>1 else '')
        else:self.reference.set(self.reference_name())
        self.form_initial=self.form_values()
        self.update_info();self.locate()

    def update_info(self):
        if not self.current:return
        p=self.current;lon,lat=p['lonlat']
        text=f"{p['scene_id']}\n经纬度 {lon:.6f}, {lat:.6f}\n参考：{LABELS[p['ref_class']]}"
        if self.show_pred.get():text+=f" | 预测：{LABELS[p['pred']]}\n抽样层：{p['stratum']}"
        self.info.set(text)

    def reference_name(self):
        if self.mode.get()=='天地图':return '天地图影像（默认底图，拍摄日期未知）'
        return self.local.get() or ''

    def form_values(self):
        return (self.reviewer.get(),self.reference.get(),self.ref_date.get(),self.confidence.get(),self.note.get('1.0','end-1c'))

    def flush_edits(self):
        if self.current and self.form_initial is not None and self.form_values()!=self.form_initial:
            return self.save(self.current['ref_class'],advance=False,refresh=False)
        return True

    def save(self,value,advance=None,refresh=True):
        if not self.current:return False
        try:
            point_id=self.current['point_id'];i=next((i for i,p in enumerate(self.visible) if p['point_id']==point_id),0)
            next_id=self.visible[i+1]['point_id'] if i+1<len(self.visible) else None
            reviewer,reference,date,confidence,note=self.form_values()
            if not reference.strip() and value in (0,1):raise ValueError('请填写参考资料，便于追溯人工判读依据')
            # 日期独立显示，同时写入已有reference字段，保持原统计脚本的数据接口。
            reference=reference.split(' | 影像日期=')[0]
            if date.strip():reference+=' | 影像日期='+date.strip()
            values=dict(ref_class=value,confidence=confidence,note=note,reviewer=reviewer,
                        reference=reference,review_date=datetime.now().astimezone().isoformat(timespec='seconds'))
            self.store.save(point_id,values)
            self.form_initial=self.form_values()
            self.status.set(f'已保存：{point_id} → {LABELS[value]}；首次保存前备份：{self.store.backup_path.name}')
            try:self.save_session()
            except OSError:pass
            self.update_info()
            if refresh:
                advance=self.auto_next.get() if advance is None else advance
                target=next_id if advance else point_id
                self.current=None;self.form_initial=None
                self.refresh_list(prefer=target)
            return True
        except Exception as exc:
            messagebox.showerror('保存失败',str(exc));return False

    def undo(self):
        if not self.store:return
        try:
            if not self.flush_edits():return
            key=self.store.undo()
            if key:
                self.current=None;self.form_initial=None;self.scene.set('全部影像');self.filter.set('全部')
                self.refresh_list(prefer=key);self.status.set('已恢复上次保存前的人工字段。')
        except Exception as exc:messagebox.showerror('撤销失败',str(exc))

    def move(self,step):
        if not self.current or not self.visible:return
        i=next(i for i,p in enumerate(self.visible) if p['point_id']==self.current['point_id'])
        key=self.visible[max(0,min(len(self.visible)-1,i+step))]['point_id']
        self.tree.selection_set(key);self.tree.see(key);self.select_point(key)

    def choose_images(self):
        paths=filedialog.askopenfilenames(title='选择带地理坐标的光学影像',filetypes=[('GeoTIFF / VRT','*.tif *.tiff *.vrt'),('所有文件','*.*')])
        if paths:
            self.images=list(dict.fromkeys(self.images+list(paths)));self.image_combo['values']=self.images
            self.local.set(paths[0]);self.mode.set('本地影像');self.source_changed()

    def source_changed(self):
        # 换底图不改历史人工结论；下一次主动保存时才采用新的参考来源。
        if self.current and self.current['ref_class'] is None:
            self.reference.set(self.reference_name())
            # 仅底图切换产生的默认来源不触发自动写入；主动编辑的备注仍会保存。
            if self.form_initial is not None:
                old=list(self.form_initial);old[1]=self.reference.get();self.form_initial=tuple(old)
        self.request_render()

    def locate(self):
        if self.current:self.center=self.current['map_xy'];self.request_render()

    def change_zoom(self,step):
        self.zoom=max(3,min(20 if self.mode.get()=='本地影像' else 18,self.zoom+step));self.request_render()

    def begin_drag(self,event):
        self.canvas.focus_set();self.drag=(event.x,event.y,self.center)

    def pan(self,event):
        if self.drag:
            x,y,old=self.drag;resolution=2*WORLD/(256*2**self.zoom)
            self.center=(old[0]-(event.x-x)*resolution,old[1]+(event.y-y)*resolution)
            self.drag=None;self.request_render()

    def resized(self,event):
        if self.resize_job:self.root.after_cancel(self.resize_job)
        self.resize_job=self.root.after(200,self.request_render)

    def request_render(self):
        self.resize_job=None;self.generation+=1
        self.canvas.delete('all');self.display_bounds=None
        if not self.current:return
        self.canvas.create_text(20,20,text='正在读取当前视窗…',fill='white',anchor='nw')
        if self.future is None:self.start_render()

    def start_render(self):
        if not self.current or self.closed:return
        width=max(100,self.canvas.winfo_width());height=max(100,self.canvas.winfo_height())
        center=self.center;zoom=self.zoom;bounds=map_bounds(center,zoom,width,height)
        generation=self.generation;mode=self.mode.get();local=self.local.get()
        token=self.token.get().strip();stretch=self.stretch.get();band_text=self.bands.get()
        def work():
            if mode=='天地图':image,message=render_tianditu(center,zoom,width,height,token,TILE_CACHE)
            else:
                if not local:raise ValueError('请先添加本地光学影像，或选择天地图并填写Key')
                bands=tuple(int(x.strip()) for x in band_text.split(','))
                image,message=render_local(local,bounds,width,height,bands,stretch)
            return generation,image,message,bounds
        self.future=self.executor.submit(work);self.future_generation=generation

    def poll(self):
        if self.closed:return
        if self.future is not None and self.future.done():
            old=self.future_generation
            try:
                generation,image,message,bounds=self.future.result()
                if generation==self.generation and self.current:
                    self.photo=ImageTk.PhotoImage(image);self.canvas.delete('all')
                    self.canvas.create_image(0,0,image=self.photo,anchor='nw');self.display_bounds=bounds
                    self.draw_points();self.image_notice.set(message)
            except Exception as exc:
                if old==self.generation:
                    self.canvas.delete('all');self.canvas.create_text(25,25,text=str(exc),fill='#ffd27f',anchor='nw',width=max(200,self.canvas.winfo_width()-50))
                    self.image_notice.set('底图未加载；点位记录未改变。')
            self.future=None
            if old!=self.generation:self.start_render()
        self.root.after(100,self.poll)

    def draw_points(self):
        if not self.current or not self.display_bounds:return
        left,bottom,right,top=self.display_bounds;width=self.canvas.winfo_width();height=self.canvas.winfo_height()
        # 仅叠加同景核验点，不把不同日期的点混在一起。
        for p in self.store.points:
            if p['scene_id']!=self.current['scene_id']:continue
            x,y=p['map_xy'];sx=(x-left)/(right-left)*width;sy=(top-y)/(top-bottom)*height
            if not (0<=sx<=width and 0<=sy<=height):continue
            selected=p['point_id']==self.current['point_id'];color='#ffcc33' if selected else '#ffffff'
            r=12 if selected else 3
            self.canvas.create_oval(sx-r,sy-r,sx+r,sy+r,outline=color,width=2)
            if selected:
                for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                    self.canvas.create_line(sx+dx*15,sy+dy*15,sx+dx*23,sy+dy*23,fill=color,width=2)
        self.canvas.create_text(12,height-15,text=f'滚轮缩放 · 拖动平移 · Z{self.zoom} · 黄色圈中心为核验像元中心',fill='white',anchor='sw')

    def key(self,event):
        if isinstance(self.root.focus_get(),(tk.Entry,ttk.Entry,tk.Text,ttk.Combobox)):return
        if event.keysym=='Right':self.move(1)
        elif event.keysym=='Left':self.move(-1)
        elif event.char=='1':self.save(1)
        elif event.char=='0':self.save(0)
        elif event.char.lower()=='u':self.save(-1)

    def export_csv(self):
        if not self.store or not self.flush_edits():return
        path=filedialog.asksaveasfilename(title='导出人工核验表',defaultextension='.csv',initialfile='manual_reference.csv',filetypes=[('CSV','*.csv')])
        if not path:return
        fields=['point_id','scene_id','date','stratum','pred','ref_class','confidence','note','reviewer','reference','review_date','weight','longitude','latitude']
        try:
            with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for p in self.store.points:
                    row={k:p.get(k,'') for k in fields};row.update(longitude=p['lonlat'][0],latitude=p['lonlat'][1]);writer.writerow(row)
            self.status.set(f'核验表已导出：{path}')
        except Exception as exc:messagebox.showerror('导出失败',str(exc))

    def save_session(self):
        if self.store:
            state={'reviewer':self.reviewer.get(),'images':self.images,'image':self.local.get(),
                   'reference_date':self.ref_date.get(),'zoom':self.zoom,'scene':self.scene.get(),
                   'filter':self.filter.get(),'point_id':self.current['point_id'] if self.current else None}
            # 天地图Key不持久保存；可通过TIANDITU_TOKEN环境变量提供。
            self.store.path.with_name(self.store.path.stem+'.viewer.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')

    def close(self):
        if not self.flush_edits():return
        try:self.save_session()
        except OSError:pass
        self.closed=True;self.executor.shutdown(wait=False,cancel_futures=True);self.root.destroy()


def main():
    if os.name=='nt':
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    root=tk.Tk();PointReviewer(root,DEFAULT_POINTS);root.mainloop()


if __name__=='__main__':main()
