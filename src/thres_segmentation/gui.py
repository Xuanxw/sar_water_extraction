"""从原 SAR GUI 脚本迁移，保留已有处理逻辑。"""
import sys
import os
import glob
import xml.etree.ElementTree as ET
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QTabWidget, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QTextEdit, QProgressBar, QMessageBox, QComboBox, QRadioButton, QButtonGroup
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from thres_segmentation.gis_worker import SARWorker

class SARWaterExtractApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TY-SARWater Extraction")
        self.worker = None
        #  设置窗口图标（放在 __init__ 开头）
        if os.path.exists("favicon.ico"):
            self.setWindowIcon(QIcon("favicon.ico"))
        else:
            # 如果打包后找不到，尝试从临时目录找
            if getattr(sys, 'frozen', False):
                icon_path = os.path.join(sys._MEIPASS, "favicon.ico")
                if os.path.exists(icon_path):
                    self.setWindowIcon(QIcon(icon_path))
        self.resize(950, 750)
        self.setStyleSheet(self.get_qss_style())
        self.init_ui()

    def get_qss_style(self):
        return """
            QMainWindow { background-color: #f0f2f5; }
            QTabWidget::pane { border: 1px solid #dcdfe6; background: white; border-radius: 4px; }
            QTabBar::tab { background: #e4e7ed; color: #606266; padding: 10px 20px; border: 1px solid #dcdfe6; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }
            QTabBar::tab:selected { background: #409eff; color: white; font-weight: bold; }
            QGroupBox { font-weight: bold; border: 1px solid #dcdfe6; border-radius: 6px; margin-top: 10px; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #303133; }
            QPushButton { background-color: #409eff; color: white; border: none; padding: 8px 15px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #66b1ff; }
            QPushButton#btn_browse { background-color: #e6a23c; }
            QPushButton#btn_browse:hover { background-color: #f0c78a; }
            QPushButton#btn_start { background-color: #67c23a; font-size: 16px; padding: 12px; }
            QPushButton#btn_start:hover { background-color: #85ce61; }
            QPushButton#btn_start:disabled { background-color: #a8abb2; }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox { border: 1px solid #dcdfe6; border-radius: 4px; padding: 6px; background: white; }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus { border: 1px solid #409eff; }
            QTextEdit { border: 1px solid #dcdfe6; border-radius: 4px; background: #2b2b2b; color: #a9b7c6; font-family: Consolas; }
            QProgressBar { border: 1px solid #dcdfe6; border-radius: 4px; text-align: center; background: #ebeef5; }
            QProgressBar::chunk { background-color: #409eff; border-radius: 3px; }
        """

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        title_label = QLabel("🛰️ SAR 水体自动化提取与地形畸变分析系统")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #303133; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        self.create_data_tab()
        self.create_param_tab()
        self.create_analysis_tab()
        self.create_run_tab()

    def create_data_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("📁 路径配置")
        form = QFormLayout()
        h1 = QHBoxLayout()
        self.input_folder = QLineEdit()
        btn1 = QPushButton("浏览..."); btn1.setObjectName("btn_browse")
        btn1.clicked.connect(lambda: self.browse_folder(self.input_folder))
        h1.addWidget(self.input_folder); h1.addWidget(btn1)
        form.addRow("SAR 影像文件夹:", h1)

        self.terrain_source_group = QButtonGroup(self)
        self.rb_local_layers = QRadioButton("采用本地坡度/坡向数据")
        self.rb_local_dem = QRadioButton("采用本地DEM并计算坡度/坡向")
        self.rb_local_dem.setChecked(True)
        self.terrain_source_group.addButton(self.rb_local_layers)
        self.terrain_source_group.addButton(self.rb_local_dem)
        source_layout = QVBoxLayout()
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(self.rb_local_layers)
        source_layout.addWidget(self.rb_local_dem)
        form.addRow("地形数据来源:", source_layout)

        form.addRow(QLabel('辅助数据请通过 main.py 准备；此界面只选择本地数据并运行处理。'))

        h3 = QHBoxLayout()
        self.dem_folder = QLineEdit()
        self.dem_folder.setPlaceholderText("包含 *slope.tif 与 *aspect.tif 的文件夹")
        self.btn_slope_aspect = QPushButton("浏览..."); self.btn_slope_aspect.setObjectName("btn_browse")
        self.btn_slope_aspect.clicked.connect(lambda: self.browse_folder(self.dem_folder))
        h3.addWidget(self.dem_folder); h3.addWidget(self.btn_slope_aspect)
        form.addRow("本地坡度/坡向文件夹:", h3)

        h4 = QHBoxLayout()
        self.local_dem_path = QLineEdit()
        self.local_dem_path.setPlaceholderText("选择单个DEM GeoTIFF或DEM分幅文件夹")
        self.btn_local_dem = QPushButton("浏览文件..."); self.btn_local_dem.setObjectName("btn_browse")
        self.btn_local_dem.clicked.connect(lambda: self.browse_raster_file(self.local_dem_path))
        self.btn_local_dem_folder = QPushButton("浏览文件夹...")
        self.btn_local_dem_folder.setObjectName("btn_browse")
        self.btn_local_dem_folder.clicked.connect(lambda: self.browse_folder(self.local_dem_path))
        h4.addWidget(self.local_dem_path)
        h4.addWidget(self.btn_local_dem)
        h4.addWidget(self.btn_local_dem_folder)
        form.addRow("本地DEM数据:", h4)

        h5 = QHBoxLayout()
        self.output_folder = QLineEdit()
        btn3 = QPushButton("浏览..."); btn3.setObjectName("btn_browse")
        btn3.clicked.connect(lambda: self.browse_folder(self.output_folder))
        h5.addWidget(self.output_folder); h5.addWidget(btn3)
        form.addRow("结果输出目录:", h5)
        btn_parse = QPushButton("🔍 自动扫描并解析 XML 参数")
        btn_parse.setStyleSheet("background-color: #909399; margin-top: 10px;")
        btn_parse.clicked.connect(self.auto_parse_xml)
        form.addRow("", btn_parse)
        for button in (self.rb_local_layers, self.rb_local_dem):
            button.toggled.connect(self.update_terrain_source_controls)
        self.update_terrain_source_controls()
        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()
        self.tabs.addTab(tab, "📁 数据与路径")

    def create_param_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        g1 = QGroupBox("🛰️ SAR 几何参数")
        f1 = QFormLayout()
        self.use_ui_params = QCheckBox("使用下方手动配置的参数 (否则自动从XML解析)")
        self.alpha_spin = QDoubleSpinBox(); self.alpha_spin.setRange(0, 360); self.alpha_spin.setValue(191.58)
        self.theta_spin = QDoubleSpinBox(); self.theta_spin.setRange(0, 90); self.theta_spin.setValue(25.0)
        self.look_dir = QComboBox(); self.look_dir.addItems(["left", "right"])
        f1.addRow("", self.use_ui_params)
        f1.addRow("航向角 (Heading °):", self.alpha_spin)
        f1.addRow("入射角 (Incidence °):", self.theta_spin)
        f1.addRow("观测方向 (Look Side):", self.look_dir)
        g1.setLayout(f1)
        layout.addWidget(g1)
        g2 = QGroupBox("💧 水体提取参数")
        f2 = QFormLayout()
        self.fill_holes = QCheckBox("开启内部孔洞填充"); self.fill_holes.setChecked(True)
        self.hole_thresh = QSpinBox(); self.hole_thresh.setRange(10, 1000); self.hole_thresh.setValue(200)
        self.hole_thresh.setToolTip("填充小于此像素数的内部孔洞")
        self.water_pct = QSpinBox(); self.water_pct.setRange(1, 100); self.water_pct.setValue(25)
        self.min_area = QSpinBox(); self.min_area.setRange(0, 10000); self.min_area.setValue(200)
        self.min_area.setToolTip(
            "在填孔、道路/建筑/地形掩膜和形态学处理全部完成后，删除小于此像素数的最终水体碎片。")
        self.enable_morph = QCheckBox("开启形态学开闭运算"); self.enable_morph.setChecked(True)
        self.enable_morph.setToolTip(
            "在道路、建筑和地形掩膜之后清理毛刺；运算后会再次强制应用全部掩膜。")
        self.smooth_vector = QCheckBox("开启水体矢量边界快速平滑")
        self.smooth_vector.setChecked(True)
        self.smooth_vector.setToolTip(
            "栅格转矢量后先按半个像元简化，再执行2轮Chaikin平滑；不改变水体栅格。")

        # 🔑 新增：高反射强度水体提取模式选择
        self.extraction_mode_group = QButtonGroup()
        self.rb_bottom = QRadioButton("正向提取 (低反射水体，默认)")
        self.rb_top = QRadioButton("反向提取 (高反射水体)")
        self.rb_bottom.setChecked(True)  # 默认底部提取
        self.extraction_mode_group.addButton(self.rb_bottom)
        self.extraction_mode_group.addButton(self.rb_top)

        f2.addRow("水体阈值百分比 (%):", self.water_pct)
        f2.addRow("最终最小连通域面积 (像素):", self.min_area)
        f2.addRow("", self.enable_morph)
        f2.addRow("", self.fill_holes)  # 新增
        f2.addRow("孔洞填充阈值 (像素):", self.hole_thresh)  # 新增
        f2.addRow("", self.smooth_vector)
        f2.addRow("提取模式:", self.rb_bottom)  # 添加单选按钮组
        f2.addRow("", self.rb_top)

        g2.setLayout(f2)
        layout.addWidget(g2)
        g3 = QGroupBox("🏔️ 地形畸变与高坡度滤除")
        f3 = QFormLayout()
        self.enable_terrain = QCheckBox("开启地形掩膜与高坡度滤除"); self.enable_terrain.setChecked(True)
        self.slope_thresh = QDoubleSpinBox(); self.slope_thresh.setRange(0, 90); self.slope_thresh.setValue(15.0)
        self.dem_resolution = QDoubleSpinBox()
        self.dem_resolution.setRange(1.0, 100.0); self.dem_resolution.setDecimals(1)
        self.dem_resolution.setValue(30.0); self.dem_resolution.setSuffix(" m")
        self.dem_resolution.setToolTip(
            "下载DEM和本地DEM都会先投影到该米制网格再计算坡度坡向；"
            "Copernicus GLO-30建议保持30米。")
        f3.addRow("", self.enable_terrain)
        f3.addRow("高坡度滤除阈值 (°):", self.slope_thresh)
        f3.addRow("DEM地形计算分辨率:", self.dem_resolution)
        self.enable_terrain.toggled.connect(self.update_terrain_source_controls)
        g3.setLayout(f3)
        layout.addWidget(g3)
        layout.addStretch()
        self.tabs.addTab(tab, "⚙️ 参数设置")

    def create_analysis_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        note = QLabel('始终统计当前水体面积。选择历史数据后计算新增、消退及持续水体；水深为岸线高程约束的几何估算。')
        note.setWordWrap(True)
        form.addRow(note)

        def path_field(title, filters):
            field = QLineEdit()
            button = QPushButton('浏览...')
            def browse():
                filename, _ = QFileDialog.getOpenFileName(self, title, '', filters)
                if filename:
                    field.setText(filename)
            button.clicked.connect(browse)
            row = QHBoxLayout(); row.addWidget(field); row.addWidget(button)
            form.addRow(title, row)
            return field

        self.analysis_reference = path_field('历史水体栅格/矢量', 'Water (*.tif *.tiff *.shp *.gpkg *.geojson)')
        self.analysis_encoding = QComboBox()
        self.analysis_encoding.addItem('提取结果或main下载：1水体 / 0非水体', ((1,), (0,)))
        self.analysis_encoding.addItem('旧GUI的ESA结果：0水体 / 1非水体', ((0,), (1,)))
        self.analysis_encoding.addItem('原始WorldCover：80水体', ((80,), (10,20,30,40,50,60,70,90,95,100)))
        form.addRow('历史栅格编码', self.analysis_encoding)
        self.analysis_coverage = path_field('历史有效覆盖面（可选）', 'Polygons (*.shp *.gpkg *.geojson)')
        self.analysis_complete = QCheckBox('确认水体矢量完整覆盖当前影像，矢量外可视为非水体')
        form.addRow(self.analysis_complete)
        self.enable_depth = QCheckBox('计算水深估算')
        form.addRow(self.enable_depth)
        self.analysis_elevation = path_field('DEM/DSM高程栅格', 'Elevation (*.tif *.tiff)')
        self.analysis_elevation.setPlaceholderText('留空可复用本次地形处理的DEM/DSM，不能使用坡度/坡向')
        self.analysis_elevation_kind = QComboBox()
        self.analysis_elevation_kind.addItems(['DEM', 'DSM'])
        form.addRow('高程类型', self.analysis_elevation_kind)
        self.analysis_scale = QDoubleSpinBox(); self.analysis_scale.setDecimals(6)
        self.analysis_scale.setRange(0.000001, 10000); self.analysis_scale.setValue(1)
        form.addRow('高程转米系数', self.analysis_scale)
        self.analysis_scope = QComboBox()
        self.analysis_scope.addItem('仅新增水体（需要历史数据）', 'new_water')
        self.analysis_scope.addItem('全部水体（需确认水下地形适用性）', 'all_water')
        form.addRow('水深计算范围', self.analysis_scope)
        self.analysis_distance = QDoubleSpinBox(); self.analysis_distance.setRange(1, 100000)
        self.analysis_distance.setValue(5000); self.analysis_distance.setSuffix(' m')
        form.addRow('最大岸线插值距离', self.analysis_distance)
        note = QLabel('DSM含建筑/植被表面；普通DEM/DSM通常不含永久水体水底。水深输出含质量标记，不能视为实测精度。默认最多2500万像元进行整景水深计算。')
        note.setWordWrap(True); form.addRow(note)
        self.tabs.addTab(tab, '面积、变化与水深')

    def create_run_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        control_group = QGroupBox("🎮 运行控制")
        ctrl_layout = QHBoxLayout()
        self.btn_start = QPushButton(" 开始处理"); self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self.start_processing)
        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0)
        ctrl_layout.addWidget(self.btn_start, 1); ctrl_layout.addWidget(self.progress_bar, 3)
        control_group.setLayout(ctrl_layout)
        layout.addWidget(control_group)
        log_group = QGroupBox("📝 实时处理日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        self.tabs.addTab(tab, "▶️ 运行与监控")

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            line_edit.setText(folder)
            # 如果修改的是SAR影像文件夹，自动更新输出路径与DEM缓存路径
            if line_edit == self.input_folder:
                extraction_path = os.path.join(folder, "Extraction")
                # 更新水体提取结果输出路径
                if not self.output_folder.text() or "Extraction" in self.output_folder.text():
                    self.output_folder.setText(extraction_path)

    def browse_raster_file(self, line_edit):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择本地DEM", "", "GeoTIFF (*.tif *.tiff);;所有文件 (*)")
        if path:
            line_edit.setText(path)

    def update_terrain_source_controls(self):
        terrain_enabled = (self.enable_terrain.isChecked()
                           if hasattr(self, "enable_terrain") else True)
        use_layers = terrain_enabled and self.rb_local_layers.isChecked()
        use_local_dem = terrain_enabled and self.rb_local_dem.isChecked()
        self.dem_folder.setEnabled(use_layers)
        self.btn_slope_aspect.setEnabled(use_layers)
        self.local_dem_path.setEnabled(use_local_dem)
        self.btn_local_dem.setEnabled(use_local_dem)
        self.btn_local_dem_folder.setEnabled(use_local_dem)
        if hasattr(self, "dem_resolution"):
            self.dem_resolution.setEnabled(
                terrain_enabled and use_local_dem)

    def auto_parse_xml(self):
        folder = self.input_folder.text()
        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, "提示", "请先选择有效的 SAR 影像文件夹！"); return
        xml_files = glob.glob(os.path.join(folder, "*.xml"))
        if not xml_files:
            QMessageBox.warning(self, "提示", "未在文件夹中找到 .xml 参数文件！"); return
        try:
            tree = ET.parse(xml_files[0]); root = tree.getroot()
            self.alpha_spin.setValue(float(root.find('.//platformHeading').text))
            self.theta_spin.setValue(float(root.find('.//incidenceAngleMidSwath').text))
            self.look_dir.setCurrentText(root.find('.//look_side').text.strip().lower())
            self.log_text.append(f"✅ 成功解析 XML 参数并填充至界面。")
        except Exception as e:
            QMessageBox.critical(self, "解析失败", f"XML 解析出错: {str(e)}")

    def start_processing(self):
        # 🔑 中断逻辑：如果正在运行，则停止
        if self.worker and self.worker.isRunning():
            self.log_text.append("⛔ 正在请求停止 SAR 处理任务...")
            self.worker.requestInterruption()
            self.worker.wait()
            self.btn_start.setText("🚀 开始处理")
            self.btn_start.setStyleSheet("background-color: #67c23a; font-size: 16px; padding: 12px;")
            self.log_text.append("✅ SAR 处理任务已停止。")
            return
        if not self.input_folder.text():
            QMessageBox.warning(self, "提示", "请先配置 SAR 影像文件夹！"); return
        reference = self.analysis_reference.text().strip() or None
        coverage = self.analysis_coverage.text().strip() or None
        elevation = self.analysis_elevation.text().strip() or None
        for filename in (reference, coverage, elevation if self.enable_depth.isChecked() else None):
            if filename and not os.path.isfile(filename):
                QMessageBox.warning(self, '提示', f'分析输入文件不存在：{filename}'); return
        if reference and os.path.splitext(reference)[1].lower() in ('.shp','.gpkg','.geojson'):
            if not coverage and not self.analysis_complete.isChecked():
                QMessageBox.warning(self, '提示', '水体矢量需要有效覆盖面，或明确确认完整覆盖当前影像。'); return
        if self.enable_depth.isChecked():
            if self.analysis_scope.currentData() == 'new_water' and not reference:
                QMessageBox.warning(self, '提示', '新增水体水深需要历史数据。'); return
            if not elevation and (not self.enable_terrain.isChecked() or self.rb_local_layers.isChecked()):
                QMessageBox.warning(self, '提示', '请提供DEM/DSM高程，或启用DEM地形处理以复用其高程。'); return
        if self.enable_terrain.isChecked():
            if self.rb_local_layers.isChecked():
                local_layers = self.dem_folder.text().strip()
                if local_layers and not os.path.isdir(local_layers):
                    QMessageBox.warning(
                        self, "提示", f"本地坡度/坡向文件夹不存在：\n{local_layers}")
                    return
            elif self.rb_local_dem.isChecked():
                local_dem = self.local_dem_path.text().strip()
                if not local_dem or not os.path.exists(local_dem):
                    QMessageBox.warning(
                        self, "提示", "请选择有效的本地DEM GeoTIFF或DEM分幅文件夹！")
                    return
        self.btn_start.setText("⛔ 停止任务")
        self.btn_start.setStyleSheet("background-color: #f56c6c; font-size: 16px; padding: 12px;")
        self.log_text.clear(); self.progress_bar.setValue(0)

        # 🔑 新增：判断提取模式
        extraction_mode = 'top' if self.rb_top.isChecked() else 'bottom'
        if self.rb_local_dem.isChecked():
            terrain_source_mode = "local_dem"
        else:
            terrain_source_mode = "local_layers"

        config = {
            "folder_path": self.input_folder.text(),
            "enable_depth": self.enable_depth.isChecked(),
            "analysis_options": dict(
                reference_path=reference,
                reference_water_values=self.analysis_encoding.currentData()[0],
                reference_dry_values=self.analysis_encoding.currentData()[1],
                reference_coverage_path=coverage,
                vector_assume_complete=self.analysis_complete.isChecked(),
                elevation_path=elevation if self.enable_depth.isChecked() else None,
                elevation_kind=self.analysis_elevation_kind.currentText(),
                elevation_scale=self.analysis_scale.value(),
                depth_scope=self.analysis_scope.currentData(),
                depth_max_distance_m=self.analysis_distance.value()),
            "terrain_source_mode": terrain_source_mode,
            "dem_folder": self.dem_folder.text().strip(),
            "local_dem_path": self.local_dem_path.text().strip(),
            "dem_buffer_m": 1000.0,
            "dem_resolution_m": self.dem_resolution.value(),
            "output_dir": self.output_folder.text() if self.output_folder.text() else os.path.join(self.input_folder.text(), "Extraction"),
            "use_ui_params": self.use_ui_params.isChecked(),
            "alpha": self.alpha_spin.value(), "theta": self.theta_spin.value(), "look_dir": self.look_dir.currentText(),
            "water_pct": self.water_pct.value(), "min_area": self.min_area.value(), "enable_morphology": self.enable_morph.isChecked(),
            "morph_kernel_size": 3, "lee_win": 5, "noise_ratio": 0.12, "p_low": 1.0, "p_high": 99.0,
            "enable_terrain": self.enable_terrain.isChecked(), "slope_thresh": self.slope_thresh.value(),
            "enable_building_mask": False,
            "enable_road_mask": False,
            "enable_farmland_mask": False,
            "fill_holes": self.fill_holes.isChecked(),  # 新增
            "hole_thresh": self.hole_thresh.value(),  # 新增
            "smooth_vector": self.smooth_vector.isChecked(),
            "smooth_iterations": 2,
            "extraction_mode": extraction_mode  # 🔑 新增：传递提取模式
        }
        self.worker = SARWorker(config)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def append_log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def on_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("🚀 开始处理")
        self.btn_start.setStyleSheet("background-color: #67c23a; font-size: 16px; padding: 12px;")
        if success:
            QMessageBox.information(self, "成功", msg)
        else:
            if "停止" in msg or "中断" in msg:
                pass
            else:
                QMessageBox.critical(self, "失败", msg)


def run_gui():
    """原始地理编码 SAR 工作流，依赖 GDAL、OpenCV、GeoPandas 和 PySide6。"""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("SimSun", 10))
    window = SARWaterExtractApp()
    window.show()
    return app.exec()
