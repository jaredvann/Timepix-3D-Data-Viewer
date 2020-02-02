#! /usr/local/bin/python3
"""
ARIADNE Experiment, Department of Physics, University of Liverpool

Usage: python main.py [data-file]

Authors: Jared Vann
"""

import enum
import math
import multiprocessing
import os.path
import struct
import sys
from typing import List, Tuple

import PyQt5 as Qt
from PyQt5 import uic
import PyQt5.QtCore
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QEvent, QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QFileDialog, \
    QLabel, QMainWindow, QErrorMessage, QMessageBox, QWidget, QVBoxLayout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters
import toml

import constants
from tpx3dview import TPX3DView, DEFAULT_PLANE_SEPARATION

HIT_DATATYPE = np.dtype([("col", "u2"), ("row", "u2"), ("toa", "u8"), ("tot", "u4")])
FLOAT_HIT_DATATYPE = np.dtype([("col", "f8"), ("row", "f8"), ("toa", "f8"), ("tot", "f8")])
FLOAT_HIT_DATATYPE_DIMS = np.dtype([("x", "f8"), ("y", "f8"), ("z", "f8"), ("tot", "f8"), ("toa", "f8")])



if len(sys.argv) > 2:
    READOUT_DIMS = (int(sys.argv[2]), int(sys.argv[3]))
else:
    READOUT_DIMS = (256, 256)

MAX_WINDOW_SIZE = 100000 #us
MAX_STEP_SIZE = 100000 #us

MAX_ALLOWED_TOT = 25550 #ns

PROJECTION_WIDGETS_SIZE = 400

FRAMES_PER_SECOND = 30

COLORMAPS = ["viridis", "hot", "gray", "hsv"]


def set_value_silently(widget, value):
    widget.blockSignals(True)
    widget.setValue(value)
    widget.blockSignals(False)


def normalise_hits(hits: np.ndarray, window_start: float, window_size: float) -> np.ndarray:
    hits2 = np.copy(hits)

    if hits.size > 0:
        hits2 = np.zeros(hits.shape, dtype=FLOAT_HIT_DATATYPE_DIMS)

        hits2["x"] = hits["col"] * constants.XY_CONVERSION_FACTOR
        hits2["y"] = hits["row"] * constants.XY_CONVERSION_FACTOR
        hits2["z"] = hits["toa"] * constants.Z_CONVERSION_FACTOR
        hits2["tot"] = hits["tot"]

        hits2["x"] -= READOUT_DIMS[0] * constants.XY_CONVERSION_FACTOR / 2
        hits2["y"] -= READOUT_DIMS[1] * constants.XY_CONVERSION_FACTOR / 2

        hits2["z"] -= (window_start + window_size/2) * constants.Z_CONVERSION_FACTOR
        hits2["z"] *= (DEFAULT_PLANE_SEPARATION*1000)/window_size

        np.clip(hits2["tot"], 0, MAX_ALLOWED_TOT)
        hits2["tot"] /= MAX_ALLOWED_TOT

    return hits2


def normalise_event_hits(hits: np.ndarray) -> np.ndarray:
    hits2 = np.zeros(hits.shape, dtype=FLOAT_HIT_DATATYPE_DIMS)

    hits2["x"] = hits["col"] * constants.XY_CONVERSION_FACTOR
    hits2["y"] = hits["row"] * constants.XY_CONVERSION_FACTOR
    hits2["z"] = hits["toa"] * constants.Z_CONVERSION_FACTOR
    hits2["tot"] = hits["tot"]
    hits2["toa"] = hits["toa"] - hits["toa"].min()
    
    hits2["x"] -= READOUT_DIMS[0] * constants.XY_CONVERSION_FACTOR / 2
    hits2["y"] -= READOUT_DIMS[1] * constants.XY_CONVERSION_FACTOR / 2
    hits2["z"] -= hits2["z"].min() + hits2["z"].ptp()/2

    np.clip(hits2["tot"], 0, MAX_ALLOWED_TOT)
    hits2["tot"] /= MAX_ALLOWED_TOT

    return hits2


class Projection(enum.Enum):
    XY = 1
    XZ = 2
    YZ = 3


class Mode(enum.Enum):
    Default = 1
    Events = 2
    Rolling = 3


class MainWindow(QMainWindow):
    def __init__(self, data_file: str = None):
        super().__init__()
        uic.loadUi("mainwindow.ui", self)

        self.mode = Mode.Default

        self.data_file = None

        # Hit buffers
        self.c_hits = []
        self.c_hits_cut = []
        self.c_hits_normed = []
        self.c_hits_normed_cut = []

        # Cluster mode state variables
        self.events_metadata = None
        self.current_event = None
        self.current_event_metadata = None

        # Segment mode state variables
        self.hits_data = None

        self.time_range = None
        self.window_start = None
        self.window_size = config["default_window_size"] * 1000
        self.step_size = config["default_step_size"] * 1000
        self.window_start_hit_index = None
        self.window_end_hit_index = None

        # Timers
        self.auto_scroll_timer = QTimer()
        self.auto_scroll_timer.timeout.connect(self.scroll)

        # Additional windows
        self.xy_projection_window = None
        self.xz_projection_window = None
        self.yz_projection_window = None
        self.heatmap_window = None

        # Widget setup
        self.colormap_combo_box.addItems(COLORMAPS)

        self.window_size_spin_box.setValue(config["default_window_size"])
        self.window_size_spin_box.setMaximum(MAX_WINDOW_SIZE)
        self.step_size_spin_box.setValue(config["default_step_size"])
        self.step_size_spin_box.setMaximum(MAX_STEP_SIZE)

        self.hit_size_spin_box.setValue(config["default_hit_size"])

        self.auto_scroll_speed_widget.setVisible(False)

        self.plane_spacing_label.setVisible(False)
        self.plane_spacing_spin_box.setVisible(False)

        # Signals - Header
        self.open_file_btn.clicked.connect(self.open_file)
        self.save_as_image_btn.clicked.connect(self.save_as_image)
        self.show_sidebar_btn.clicked.connect(self.toggle_show_sidebar)
        self.show_stats_btn.clicked.connect(self.toggle_show_stats)
        self.show_xy_projection_btn.clicked.connect(self.show_xy_projection)
        self.show_xz_projection_btn.clicked.connect(self.show_xz_projection)
        self.show_yz_projection_btn.clicked.connect(self.show_yz_projection)
        self.show_heatmap_btn.clicked.connect(self.show_heatmap)

        # Signals - Footer
        self.previous_event_btn.clicked.connect(self.previous_event)
        self.next_event_btn.clicked.connect(self.next_event)

        self.event_spin_box.valueChanged.connect(self.event_changed)

        self.jump_backward_btn.clicked.connect(self.jump_backward)
        self.jump_forward_btn.clicked.connect(self.jump_forward)

        self.previous_window_btn.clicked.connect(self.previous_window)
        self.next_window_btn.clicked.connect(self.next_window)

        self.current_position_spin_box.valueChanged.connect(self.window_changed)
        self.window_size_spin_box.valueChanged.connect(self.window_size_changed)
        self.step_size_spin_box.valueChanged.connect(self.step_size_changed)

        # Signals - Sidebar
        self.quick_jump_slider.valueChanged.connect(self.quick_jump_slider_changed)

        self.min_tot_cut_spin_box.valueChanged.connect(self.min_tot_cut_spin_box_changed)
        self.min_tot_cut_slider.valueChanged.connect(self.min_tot_cut_slider_changed)
        self.min_tot_cut_lock_checkbox.stateChanged.connect(self.toggle_min_tot_cut_lock)

        self.colormap_combo_box.currentIndexChanged.connect(self.colormap_changed)
        self.hit_size_spin_box.valueChanged.connect(self.hit_size_changed)
        self.light_background_check_box.stateChanged.connect(self.toggle_background_color)
        self.additive_view_mode_check_box.stateChanged.connect(self.toggle_additive_view_mode)
        self.show_planes_check_box.stateChanged.connect(self.toggle_show_planes)
        self.plane_spacing_spin_box.valueChanged.connect(self.plane_spacing_changed)

        self.rotate_camera_btn.clicked.connect(self.toggle_rotate_camera)
        self.auto_scroll_btn.clicked.connect(self.toggle_auto_scroll)
        self.auto_scroll_speed_spin_box.valueChanged.connect(self.change_auto_scroll_speed)

        self.plot_time_series_histogram_btn.clicked.connect(self.plot_time_series_histogram)
        self.plot_time_series_histogram_all_hits_btn.clicked.connect(self.plot_time_series_histogram_all_hits)

        # self.keyPressed.connect(self.on_key)

        # Main Plot Setup
        self.view_widget = TPX3DView(plane_size=READOUT_DIMS)
        self.view_widget_container.addWidget(self.view_widget)

        # Projection Plots Setup
        pg.setConfigOption("background", 0.1)

        # Initialisation
        self.set_mode(Mode.Default)

        # If have data file try and load data
        if data_file and not self.load_data(data_file):
            sys.exit()

        self.show()


    def set_mode(self, mode: Mode):
        self.rolling_mode_controls_widget.setVisible(mode == Mode.Rolling)
        self.event_mode_controls_widget.setVisible(mode == Mode.Events)

        self.event_info_group_box.setVisible(mode == Mode.Events)

        self.plot_time_series_histogram_all_hits_btn.setVisible(mode == Mode.Rolling)

        self.label_13.setVisible(mode == Mode.Events)
        self.label_14.setVisible(mode == Mode.Events)
        self.stats__min_toa_label.setVisible(mode == Mode.Events)
        self.stats__max_toa_label.setVisible(mode == Mode.Events)

        self.plane_spacing_label.setVisible(mode == Mode.Events)
        self.plane_spacing_spin_box.setVisible(mode == Mode.Events)

        self.auto_scroll_btn.setVisible(mode == Mode.Events)

        self.mode = mode

        self.view_widget.mark_first_hit = (mode == Mode.Events)


    def load_data(self, filename: str, mode: Mode = None) -> bool:
        msg_box = QErrorMessage()

        if not os.path.exists(filename):
            msg_box.showMessage(f"'{filename}' does not exist!")
            msg_box.exec_()
            return False

        if not os.path.isfile(filename):
            msg_box.showMessage(f"'{filename}' is not a file!")
            msg_box.exec_()
            return False

        stem = os.path.basename(filename).split(".")[0]

        if mode == Mode.Rolling or stem == "hits":
            hits = np.memmap(filename, dtype=HIT_DATATYPE, mode="r")

            if hits.size == 0:
                msg_box.showMessage("No hits found in file")
                msg_box.exec_()
                return False

            self.hits_data = hits

            self.min_time = self.hits_data["toa"].min()
            self.max_time = self.hits_data["toa"].max()
            self.time_range = self.max_time - self.min_time

            self.window_start = self.min_time

            self.quick_jump_slider.setValue(self.min_time/1000)
            self.quick_jump_slider.setMinimum(self.min_time/1000)
            self.quick_jump_slider.setMaximum(self.max_time/1000)

            self.current_position_spin_box.setValue(self.min_time/1000)
            self.current_position_spin_box.setMinimum(self.min_time/1000)
            self.current_position_spin_box.setMaximum((self.max_time - self.window_size)/1000)

            self.set_mode(Mode.Rolling)

        else:
            metadata = pd.read_csv(filename.replace(".bin", ".csv"), index_col="event")

            if len(metadata) == 0:
                msg_box.showMessage("Could not read metadata file")
                msg_box.exec_()
                return False

            self.events_metadata = metadata
            self.current_event = 0

            self.quick_jump_slider.setValue(0)
            self.quick_jump_slider.setMinimum(0)
            self.quick_jump_slider.setMaximum(len(metadata))

            set_value_silently(self.event_spin_box, metadata.index[0])
            self.event_spin_box.setMinimum(metadata.index[0])
            self.event_spin_box.setMaximum(metadata.index[-1])

            if np.all(metadata["duration"] == metadata.iloc[0].duration):
                self.view_widget.set_plane_separation(metadata.iloc[0].duration)
                self.plane_spacing_spin_box.setValue(metadata.iloc[0].duration)
                self.plane_spacing_spin_box.setEnabled(False)
            else:
                self.plane_spacing_spin_box.setEnabled(True)

            self.set_mode(Mode.Events)

        self.save_as_image_btn.setEnabled(True)
        self.show_xy_projection_btn.setEnabled(True)
        self.show_xz_projection_btn.setEnabled(True)
        self.show_yz_projection_btn.setEnabled(True)
        self.show_heatmap_btn.setEnabled(True)

        self.data_file = filename
        self.update()

        return True


    def update(self):
        if self.mode == Mode.Events:
            self.current_event_metadata = self.events_metadata.iloc[self.current_event]            
            self.c_hits = np.fromfile(self.data_file, dtype=HIT_DATATYPE, count=int(self.current_event_metadata["hits"]), offset=int(self.current_event_metadata["offset"]))

            if not self.validate_data(self.c_hits):
                sys.exit(-1)

            # Temporary fix for Simulation data
            self.c_hits = self.c_hits[self.c_hits["toa"] != 0]

            self.c_hits_normed = normalise_event_hits(self.c_hits)

            self.header_info_label.setText(f"Event {self.current_event+1}/{len(self.events_metadata)}")

            self.event_info__start_label.setText(f"{self.current_event_metadata.time/1000}us")
            self.event_info__length_label.setText(f"{self.current_event_metadata.duration/1000}us")

            set_value_silently(self.quick_jump_slider, self.current_event)
            set_value_silently(self.event_spin_box, self.current_event)

            self.previous_event_btn.setEnabled(self.current_event != 0)
            set_value_silently(self.event_spin_box, self.current_event_metadata.name)
            self.next_event_btn.setEnabled(self.current_event < len(self.events_metadata)-1)

        elif self.mode == Mode.Rolling:
            start_indices = np.where(self.hits_data["toa"] >= self.window_start)[0]
            end_indices = np.where(self.hits_data["toa"] > self.window_start + self.window_size)[0]

            self.window_start_hit_index = 0 if len(start_indices) == 0 else start_indices[0]
            self.window_end_hit_index = len(self.hits_data) if len(end_indices) == 0 else end_indices[0]

            self.header_info_label.setText(f"Showing hits {self.window_start_hit_index}-{self.window_end_hit_index}/{len(self.hits_data)}")

            self.jump_backward_btn.setEnabled(self.window_start_hit_index > 1)
            self.previous_window_btn.setEnabled(self.window_start_hit_index > 1)
            self.next_window_btn.setEnabled(self.window_start + self.window_size <= self.max_time)
            self.jump_forward_btn.setEnabled(self.window_end_hit_index < len(self.hits_data))

            set_value_silently(self.quick_jump_slider, self.window_start/1000)

            self.current_position_spin_box.blockSignals(True)
            set_value_silently(self.current_position_spin_box, self.window_start/1000)
            self.current_position_spin_box.blockSignals(False)

            self.c_hits = self.hits_data[self.window_start_hit_index:self.window_end_hit_index]
            self.c_hits_normed = normalise_hits(self.c_hits, self.window_start, self.window_size)

        else:
            return

        cut_mask = self.c_hits["tot"] > self.min_tot_cut_spin_box.value()

        self.c_hits_cut = self.c_hits[cut_mask]
        self.c_hits_normed_cut = self.c_hits_normed[cut_mask]

        total_n_hits = len(self.c_hits)
        n_hits = len(self.c_hits_cut)

        # Generate statistics
        if n_hits > 0:
            total_tot = np.sum(self.c_hits_cut["tot"])
            avg_tot = np.average(self.c_hits_cut["tot"])
            min_tot = np.min(self.c_hits_cut["tot"])
            max_tot = np.max(self.c_hits_cut["tot"])
            min_toa = np.min(self.c_hits_cut["toa"])
            max_toa = np.max(self.c_hits_cut["toa"])
        else:
            total_tot = 0
            avg_tot = 0
            min_tot = 0
            max_tot = 0
            min_toa = 0
            max_toa = 0

        # Update cuts panel
        if not self.min_tot_cut_lock_checkbox.isChecked():
            self.min_tot_cut_spin_box.setMaximum(max_tot)
            self.min_tot_cut_slider.setMaximum(max_tot)

        # Update statistics panel
        self.stats__hits_label.setText(f"{n_hits}/{total_n_hits}")
        self.stats__total_tot_label.setText(f"{total_tot}ns")
        self.stats__avg_tot_label.setText(f"{float(avg_tot):.0f}ns")
        self.stats__min_tot_label.setText(f"{min_tot}ns")
        self.stats__max_tot_label.setText(f"{max_tot}ns")

        if self.mode == Mode.Events:
            self.stats__min_toa_label.setText(f"{min_toa/1000}µs")
            self.stats__max_toa_label.setText(f"{max_toa/1000}µs")

        # Update projection views
        if self.xy_projection_window and self.xy_projection_window.isVisible():
            self.xy_projection_window.update(self.c_hits_cut)

        if self.xz_projection_window and self.xz_projection_window.isVisible():
            self.xz_projection_window.update(self.c_hits_cut)

        if self.yz_projection_window and self.yz_projection_window.isVisible():
            self.yz_projection_window.update(self.c_hits_cut)

        self.view_widget.draw_hits(self.c_hits_normed_cut)


    def validate_data(self, hits: np.ndarray) -> bool:
        msg_box = QErrorMessage()

        max_col_size = np.max(hits["col"])
        max_row_size = np.max(hits["row"])

        if max_col_size >= READOUT_DIMS[0]:
            msg_box.showMessage(f"Error validating data: event had max col size of {max_col_size}")
            msg_box.exec_()
            return False
        if max_row_size >= READOUT_DIMS[1]:
            msg_box.showMessage(f"Error validating data: event had max row size of {max_row_size}")
            msg_box.exec_()
            return False

        return True


    @pyqtSlot()
    def open_file(self):
        filename = QFileDialog.getOpenFileName(self, "Open file", "", "Data files (*.bin)")[0]

        if filename:
            self.load_data(filename)


    @pyqtSlot()
    def save_as_image(self):
        if self.mode == Mode.Events:
            default_file_name = f"{self.current_event_metadata.event}.png"
        else:
            default_file_name = f"{int(self.window_start/1000)}-{int((self.window_start+self.window_size)/1000)}us.png"

        file_name, _ = QFileDialog.getSaveFileName(self, "Save as Image", default_file_name, "Images (*.png *.xpm *.jpg)")

        if file_name:
            self.view_widget.readQImage().save(file_name)


    @pyqtSlot()
    def toggle_show_sidebar(self):
        self.sidebar_widget.setVisible(self.show_sidebar_btn.isChecked())
        self.show_stats_btn.setEnabled(self.show_sidebar_btn.isChecked())

    
    @pyqtSlot()
    def toggle_show_stats(self):
        self.stats_group_box.setVisible(self.show_stats_btn.isChecked())


    @pyqtSlot()
    def show_xy_projection(self):
        self.show_xy_projection_btn.setEnabled(False)

        self.xy_projection_window = ProjectionPlotWindow(self, Projection.XY, READOUT_DIMS, self.c_hits_cut)
        self.xy_projection_window.show()


    @pyqtSlot()
    def show_xz_projection(self):
        self.show_xz_projection_btn.setEnabled(False)

        self.xz_projection_window = ProjectionPlotWindow(self, Projection.XZ, READOUT_DIMS, self.c_hits_cut)
        self.xz_projection_window.show()


    @pyqtSlot()
    def show_yz_projection(self):
        self.show_yz_projection_btn.setEnabled(False)

        self.yz_projection_window = ProjectionPlotWindow(self, Projection.YZ, READOUT_DIMS, self.c_hits_cut)
        self.yz_projection_window.show()


    @pyqtSlot()
    def show_heatmap(self):
        reply = QMessageBox.question(self,
            "Generate heatmap", "Generating a heatmap can take some time. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.show_heatmap_btn.setEnabled(False)

            heatmap = np.zeros(shape=READOUT_DIMS, dtype="int")

            if self.mode == Mode.Events:
                for i, event in self.events_metadata.iterrows():
                    print(f"\rParsing event {i}/{len(self.events_metadata)}", end="")

                    hits = np.fromfile(self.data_file, dtype=HIT_DATATYPE, count=int(event.hits), offset=int(event.offset))

                    if not self.validate_data(hits):
                        print()
                        return

                    for hit in hits:
                        heatmap[hit["row"], hit["col"]] += hit["tot"]

                print()

            else:
                for hit in self.hits_data:
                    heatmap[hit["row"], hit["col"]] += hit["tot"]

            self.heatmap_window = HeatmapPlotWindow(self, heatmap)
            self.heatmap_window.show()


    @pyqtSlot()
    def previous_event(self):
        if self.current_event > 0:
            self.current_event -= 1
            self.update()


    @pyqtSlot()
    def next_event(self):
        if self.current_event < len(self.events_metadata):
            self.current_event += 1
            self.update()


    @pyqtSlot()
    def event_changed(self):
        self.current_event = self.events_metadata.index.get_loc(self.event_spin_box.value())
        self.update()


    @pyqtSlot()
    def jump_backward(self):
        start_indices = np.where(self.hits_data["toa"] < self.window_start)[0]
        self.window_start = max(self.hits_data[start_indices[-1]]["toa"] - self.window_size, 0)

        self.update()


    @pyqtSlot()
    def jump_forward(self):
        start_indices = np.where(self.hits_data["toa"] > self.window_start + self.window_size)[0]
        self.window_start = self.hits_data[start_indices[0]]["toa"]

        self.update()


    @pyqtSlot()
    def previous_window(self):
        diff = self.step_size if (self.window_start > self.step_size) else self.window_start
        self.window_start -= diff

        self.update()


    @pyqtSlot()
    def next_window(self):
        window_end = self.window_start + self.window_size
        diff = self.step_size if (self.max_time - window_end > self.step_size) else (self.max_time - window_end)
        self.window_start += diff

        self.update()


    @pyqtSlot()
    def window_changed(self):
        self.window_start = self.current_position_spin_box.value() * 1000
        self.update()


    @pyqtSlot()
    def window_size_changed(self):
        self.window_size = self.window_size_spin_box.value() * 1000
        self.update()


    @pyqtSlot()
    def step_size_changed(self):
        self.step_size = self.step_size_spin_box.value() * 1000


    @pyqtSlot()
    def quick_jump_slider_changed(self):
        if self.mode == Mode.Rolling:
            self.window_start = self.quick_jump_slider.value() * 1000
        elif self.mode == Mode.Events:
            self.current_event = self.quick_jump_slider.value()

        self.update()


    @pyqtSlot()
    def min_tot_cut_spin_box_changed(self):
        self.min_tot_cut_slider.setValue(self.min_tot_cut_spin_box.value())
        self.update()


    @pyqtSlot()
    def min_tot_cut_slider_changed(self):
        self.min_tot_cut_spin_box.setValue(self.min_tot_cut_slider.value())
        self.update()


    @pyqtSlot()
    def toggle_min_tot_cut_lock(self):
        self.min_tot_cut_spin_box.setEnabled(not self.min_tot_cut_lock_checkbox.isChecked())
        self.min_tot_cut_slider.setEnabled(not self.min_tot_cut_lock_checkbox.isChecked())
        self.update()


    @pyqtSlot()
    def colormap_changed(self):
        self.view_widget.set_colormap(self.colormap_combo_box.currentText())


    @pyqtSlot()
    def hit_size_changed(self):
        self.view_widget.set_hit_size(self.hit_size_spin_box.value())


    @pyqtSlot()
    def toggle_background_color(self):
        light = self.light_background_check_box.isChecked()

        self.view_widget.set_background(light)

        if light:
            self.view_widget.set_display_mode("translucent")
        else:
            self.toggle_additive_view_mode()

        self.additive_view_mode_check_box.setVisible(not light)
        self.additive_view_mode_label.setVisible(not light)


    @pyqtSlot()
    def toggle_additive_view_mode(self):
        self.view_widget.set_display_mode("additive" if self.additive_view_mode_check_box.isChecked() else "translucent")


    @pyqtSlot()
    def toggle_show_planes(self):
        self.view_widget.set_show_planes(self.show_planes_check_box.isChecked())


    @pyqtSlot()
    def plane_spacing_changed(self):
        self.view_widget.set_plane_separation(self.plane_spacing_spin_box.value())


    @pyqtSlot()
    def toggle_rotate_camera(self):
        self.view_widget.auto_rotate(self.rotate_camera_btn.isChecked())


    @pyqtSlot()
    def scroll(self):
        if self.mode == Mode.Rolling:
            window_end = self.window_start + self.window_size
            diff = self.step_size/10 if (self.time_range - window_end > self.step_size/10) else (self.time_range - window_end)
            self.window_start = self.window_start + diff

            self.update()
        elif self.mode == Mode.Events:
            if self.current_event < len(self.events_metadata) - 1:
                self.current_event += 1
                self.update()
            else:
                self.auto_scroll_timer.stop()
                self.auto_scroll_btn.setChecked(False)


    @pyqtSlot()
    def toggle_auto_scroll(self):
        if self.auto_scroll_btn.isChecked():
            self.auto_scroll_timer.start(1000/self.auto_scroll_speed_spin_box.value() if self.mode == Mode.Events else 1000/FRAMES_PER_SECOND)
        else:
            self.auto_scroll_timer.stop()

        self.auto_scroll_speed_widget.setVisible(self.mode == Mode.Events and self.auto_scroll_btn.isChecked())


    @pyqtSlot()
    def change_auto_scroll_speed(self):
        self.auto_scroll_timer.setInterval(1000/self.auto_scroll_speed_spin_box.value())


    @pyqtSlot()
    def plot_time_series_histogram(self):
        datapoints = self.c_hits_cut["toa"]

        plt.title("Time series of hits")
        plt.xlabel("Time (us)")
        plt.ylabel("Hits")
        plt.hist(datapoints, bins=100)
        plt.show()


    @pyqtSlot()
    def plot_time_series_histogram_all_hits(self):
        datapoints = self.hits_data["toa"]

        plt.title("Time series of hits")
        plt.xlabel("Time (us)")
        plt.ylabel("Hits")
        plt.hist(datapoints, bins=100)
        plt.show()



class ProjectionPlotWindow(QWidget):
    # keyPressed = pyqtSignal(QEvent)

    def __init__(self, parent: MainWindow, projection: Projection, plane_size: Tuple[int, int] = (256, 256), hits: np.ndarray = None):
        super().__init__()
        uic.loadUi("projectionplotwindow.ui", self)

        self.parent = parent
        self.projection = projection
        self.plane_size = plane_size
        self.toa_mode = False

        self.cmap = plt.get_cmap(self.parent.colormap_combo_box.currentText())

        self.save_as_image_btn.setVisible(False)
        self.toggle_light_mode_btn.setVisible(False)
        self.toggle_toa_mode_btn.setVisible(projection == Projection.XY)

        # self.toggle_light_mode_btn.clicked.connect(self.toggle_light_mode)
        self.toggle_toa_mode_btn.clicked.connect(self.toggle_toa_mode)

        self.view = pg.GraphicsLayoutWidget()
        self.view.setMinimumSize(600, 600)

        # Set pxMode=False to allow spots to transform with the view
        self.plot = pg.ScatterPlotItem(pxMode=False)

        self.view.addPlot().addItem(self.plot)
        self.view_widget_container.addWidget(self.view)

        self.update(hits)


    def closeEvent(self, event):
        if self.projection == Projection.XY:
            self.parent.show_xy_projection_btn.setEnabled(True)
        elif self.projection == Projection.XZ:
            self.parent.show_xz_projection_btn.setEnabled(True)
        elif self.projection == Projection.YZ:
            self.parent.show_yz_projection_btn.setEnabled(True)
        super().closeEvent(event)


    def update(self, hits: np.ndarray):
        if hits.size == 0:
            return
        
        self.hits = hits
        points = []

        if self.toa_mode:
            cmap = plt.get_cmap("hsv")

            grid = np.zeros(READOUT_DIMS)

            hits = np.copy(hits)
            hits["toa"] -= np.min(hits["toa"])

            for hit in hits:
                grid[int(hit[0]), int(hit[1])] = hit[2]

            grid /= np.max(grid)

            for y in range(READOUT_DIMS[1]):
                for x in range(READOUT_DIMS[0]):
                    if grid[y,x]:
                        r, g, b, _ = cmap(grid[y,x])
                        points.append({"pos": (y, READOUT_DIMS[0]-x), "size": 0.9, "pen": None, "brush": pg.mkColor(r*255, g*255, b*255)})

        else:
            if self.projection == Projection.XY:
                flat_hits = self.flatten_data_xy()
            elif self.projection == Projection.XZ:
                flat_hits = self.flatten_data_xz(100)
            else:
                flat_hits = self.flatten_data_yz(100)

            flat_hits /= np.max(flat_hits)

            for y in range(flat_hits.shape[0]):
                for x in range(flat_hits.shape[1]):
                    if flat_hits[y, x]:
                        r, g, b, _ = self.cmap(flat_hits[y, x])
                        points.append({"pos": (x, flat_hits.shape[0]-y), "size": 0.9, "pen": None, "brush": pg.mkColor(r*255, g*255, b*255)})

        self.plot.clear()
        self.plot.addPoints(points)


    # @pyqtSlot()
    # def toggle_light_mode(self):
    #     if self.toggle_light_mode_btn.isChecked():
    #         pg.setConfigOption('background', 'w')
    #         pg.setConfigOption('foreground', 'k')
    #     else:
    #         pg.setConfigOption('background', 'k')
    #         pg.setConfigOption('foreground', 'w')

    #     self.update(self.hits)


    @pyqtSlot()
    def toggle_toa_mode(self):
        self.toa_mode = self.toggle_toa_mode_btn.isChecked()
        self.update(self.hits)


    def flatten_data_xy(self) -> np.ndarray:
        grid = np.zeros(self.plane_size)

        for hit in self.hits:
            grid[math.floor(hit[1]), math.floor(hit[0])] += hit[3]

        return grid


    def flatten_data_xz(self, bins: int = None) -> np.ndarray:
        z_min = np.min(self.hits["toa"])
        z_max = np.max(self.hits["toa"])
        z_range = z_max - z_min

        grid = np.zeros((math.ceil(z_range+1), self.plane_size[0]))

        for hit in self.hits:
            grid[math.floor((hit[2]-z_min)), math.floor(hit[1])] += hit[3]

        if bins is not None and bins > 0:
            binned_grid = np.zeros((bins, self.plane_size[0]))
            bin_size = math.ceil(grid.shape[0]/bins)

            for i in range(bins):
                binned_grid[i] = np.sum(grid[i*bin_size:min((i+1)*bin_size, grid.shape[0]-1)], axis=0)

            return binned_grid
        else:
            return grid


    def flatten_data_yz(self, bins: int = None) -> np.ndarray:
        z_min = np.min(self.hits["toa"])
        z_max = np.max(self.hits["toa"])
        z_range = z_max - z_min

        grid = np.zeros((math.ceil(z_range+1), self.plane_size[1]))

        for hit in self.hits:
            grid[math.floor((hit[2]-z_min)), math.floor(hit[0])] += hit[3]

        if bins is not None and bins > 0:
            binned_grid = np.zeros((bins, self.plane_size[1]))
            bin_size = math.ceil(grid.shape[0]/bins)
            for i in range(bins):
                binned_grid[i] = np.sum(grid[i*bin_size:min((i+1)*bin_size, grid.shape[0]-1)], axis=0)

            return binned_grid
        else:
            return grid


class HeatmapPlotWindow(QWidget):
    def __init__(self, parent: MainWindow, heatmap: np.ndarray = None):
        super().__init__()
        uic.loadUi("projectionplotwindow.ui", self)

        self.parent = parent

        self.cmap = plt.get_cmap(self.parent.colormap_combo_box.currentText())

        self.save_as_image_btn.setVisible(False)
        self.toggle_light_mode_btn.setVisible(False)
        self.toggle_toa_mode_btn.setVisible(False)

        self.view = pg.GraphicsLayoutWidget()
        self.view.setMinimumSize(600, 600)

        self.plot = pg.ScatterPlotItem(pxMode=False)   # Set pxMode=False to allow spots to transform with the view      

        self.view.addPlot().addItem(self.plot)
        self.view_widget_container.addWidget(self.view)

        self.update(heatmap)


    def closeEvent(self, event):
        self.parent.show_heatmap_btn.setEnabled(True)
        super().closeEvent(event)


    def update(self, heatmap: np.ndarray):
        points = []

        heatmap = heatmap.astype(float)
        heatmap /= np.max(heatmap)

        for y in range(heatmap.shape[0]):
            for x in range(heatmap.shape[1]):
                if heatmap[y, x]:
                    r, g, b, _ = self.cmap(heatmap[y, x])

                    points.append({"pos": (x, heatmap.shape[0]-y), "size": 0.9, "pen": None, "brush": pg.mkColor(r*255, g*255, b*255)})

        self.plot.clear()
        self.plot.addPoints(points)


if __name__ == "__main__":
    config = toml.load("event_viewer_config.toml")

    app = QApplication(sys.argv)
    window = MainWindow(sys.argv[1] if len(sys.argv) > 1 else None)
    sys.exit(app.exec_())
