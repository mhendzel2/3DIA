"""
Advanced Tracking Widget for Scientific Image Analyzer
Implements cell lineage tracking, gap closing, and track editing similar to Imaris Track
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
                             QTreeWidget, QTreeWidgetItem, QSplitter, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import napari
from napari.layers import Image, Labels, Tracks, Points

try:
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class TrackingThread(QThread):
    """Thread for time-consuming tracking operations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, timepoints, parameters):
        super().__init__()
        self.timepoints = timepoints
        self.parameters = parameters
        
    def run(self):
        """Execute tracking"""
        try:
            if not HAS_SCIPY:
                self.error.emit("scipy is required for tracking")
                return
            
            max_distance = self.parameters.get('max_distance', 50.0)
            max_gap = self.parameters.get('max_gap', 2)
            min_track_length = self.parameters.get('min_track_length', 3)
            
            # Extract centroids for each timepoint
            all_detections = []
            for t_idx, labels in enumerate(self.timepoints):
                self.progress.emit(int(30 * t_idx / len(self.timepoints)))
                detections = self._extract_centroids(labels, t_idx)
                all_detections.append(detections)
            
            # Perform frame-to-frame linking
            self.progress.emit(40)
            tracks = self._link_detections(all_detections, max_distance)
            
            # Gap closing
            if max_gap > 0:
                self.progress.emit(60)
                tracks = self._close_gaps(tracks, all_detections, max_distance, max_gap)
            
            # Filter short tracks
            self.progress.emit(80)
            tracks = [t for t in tracks if len(t) >= min_track_length]
            
            # Analyze tracks
            self.progress.emit(90)
            track_statistics = self._analyze_tracks(tracks)
            
            # Detect divisions/merges
            lineage_events = self._detect_lineage_events(tracks, all_detections, max_distance)
            
            self.progress.emit(100)
            self.finished.emit({
                'tracks': tracks,
                'statistics': track_statistics,
                'lineage_events': lineage_events,
                'all_detections': all_detections
            })
            
        except Exception as e:
            self.error.emit(str(e))
    
    def _extract_centroids(self, labels, timepoint):
        """Extract object centroids from label image"""
        from skimage import measure
        
        regions = measure.regionprops(labels)
        detections = []
        
        for region in regions:
            detections.append({
                'id': region.label,
                'timepoint': timepoint,
                'position': region.centroid,
                'area': region.area
            })
        
        return detections
    
    def _link_detections(self, all_detections, max_distance):
        """Link detections across frames using Hungarian algorithm"""
        tracks = []
        available_tracks = []
        track_id_counter = 0
        
        for t_idx in range(len(all_detections)):
            current_detections = all_detections[t_idx]
            
            if t_idx == 0:
                # Initialize tracks from first frame
                for det in current_detections:
                    track = {
                        'track_id': track_id_counter,
                        'detections': [det]
                    }
                    tracks.append(track)
                    available_tracks.append(track)
                    track_id_counter += 1
            else:
                # Link current detections to existing tracks
                if not available_tracks or not current_detections:
                    # Start new tracks for unmatched detections
                    for det in current_detections:
                        track = {
                            'track_id': track_id_counter,
                            'detections': [det]
                        }
                        tracks.append(track)
                        available_tracks.append(track)
                        track_id_counter += 1
                    continue
                
                # Build cost matrix
                track_positions = np.array([t['detections'][-1]['position'] for t in available_tracks])
                det_positions = np.array([d['position'] for d in current_detections])
                
                cost_matrix = cdist(track_positions, det_positions)
                
                # Apply max distance threshold
                cost_matrix[cost_matrix > max_distance] = 1e10
                
                # Solve assignment problem
                track_indices, det_indices = linear_sum_assignment(cost_matrix)
                
                # Update tracks with assignments
                new_available_tracks = []
                matched_dets = set()
                
                for track_idx, det_idx in zip(track_indices, det_indices):
                    if cost_matrix[track_idx, det_idx] < 1e10:
                        available_tracks[track_idx]['detections'].append(current_detections[det_idx])
                        new_available_tracks.append(available_tracks[track_idx])
                        matched_dets.add(det_idx)
                
                # Start new tracks for unmatched detections
                for det_idx, det in enumerate(current_detections):
                    if det_idx not in matched_dets:
                        track = {
                            'track_id': track_id_counter,
                            'detections': [det]
                        }
                        tracks.append(track)
                        new_available_tracks.append(track)
                        track_id_counter += 1
                
                available_tracks = new_available_tracks
        
        return tracks
    
    def _close_gaps(self, tracks, all_detections, max_distance, max_gap):
        """Close gaps in tracks by linking across missing frames"""
        # Build a list of track endpoints and startpoints
        endpoints = []
        startpoints = []
        
        for track in tracks:
            last_det = track['detections'][-1]
            first_det = track['detections'][0]
            
            endpoints.append({
                'track': track,
                'timepoint': last_det['timepoint'],
                'position': last_det['position']
            })
            
            startpoints.append({
                'track': track,
                'timepoint': first_det['timepoint'],
                'position': first_det['position']
            })
        
        # Try to link endpoints to startpoints within gap distance
        for endpoint in endpoints:
            for startpoint in startpoints:
                if endpoint['track'] is startpoint['track']:
                    continue
                
                gap_size = startpoint['timepoint'] - endpoint['timepoint']
                
                if 1 < gap_size <= max_gap:
                    # Check distance
                    distance = np.linalg.norm(
                        np.array(endpoint['position']) - np.array(startpoint['position'])
                    )
                    
                    if distance <= max_distance * gap_size:
                        # Merge tracks
                        endpoint['track']['detections'].extend(startpoint['track']['detections'])
                        tracks.remove(startpoint['track'])
                        break
        
        return tracks
    
    def _analyze_tracks(self, tracks):
        """Calculate track statistics"""
        statistics = {
            'total_tracks': len(tracks),
            'track_lengths': [],
            'track_displacements': [],
            'track_speeds': [],
            'track_straightness': []
        }
        
        for track in tracks:
            detections = track['detections']
            track_length = len(detections)
            statistics['track_lengths'].append(track_length)
            
            if track_length > 1:
                # Calculate displacement
                start_pos = np.array(detections[0]['position'])
                end_pos = np.array(detections[-1]['position'])
                displacement = np.linalg.norm(end_pos - start_pos)
                statistics['track_displacements'].append(displacement)
                
                # Calculate path length and speed
                path_length = 0
                for i in range(1, track_length):
                    pos1 = np.array(detections[i-1]['position'])
                    pos2 = np.array(detections[i]['position'])
                    path_length += np.linalg.norm(pos2 - pos1)
                
                speed = path_length / track_length if track_length > 1 else 0
                statistics['track_speeds'].append(speed)
                
                # Calculate straightness (displacement / path_length)
                straightness = displacement / path_length if path_length > 0 else 0
                statistics['track_straightness'].append(straightness)
        
        # Calculate means
        statistics['mean_track_length'] = np.mean(statistics['track_lengths']) if statistics['track_lengths'] else 0
        statistics['mean_displacement'] = np.mean(statistics['track_displacements']) if statistics['track_displacements'] else 0
        statistics['mean_speed'] = np.mean(statistics['track_speeds']) if statistics['track_speeds'] else 0
        statistics['mean_straightness'] = np.mean(statistics['track_straightness']) if statistics['track_straightness'] else 0
        
        return statistics
    
    def _detect_lineage_events(self, tracks, all_detections, max_distance):
        """Detect cell division and merging events"""
        events = {
            'divisions': [],
            'merges': []
        }
        
        # Look for divisions: one track splitting into two
        for t_idx in range(len(all_detections) - 1):
            # Find tracks ending at t_idx
            ending_tracks = [t for t in tracks if t['detections'][-1]['timepoint'] == t_idx]
            # Find tracks starting at t_idx + 1
            starting_tracks = [t for t in tracks if t['detections'][0]['timepoint'] == t_idx + 1]
            
            for ending_track in ending_tracks:
                end_pos = np.array(ending_track['detections'][-1]['position'])
                
                # Count nearby starting tracks
                nearby_starts = []
                for starting_track in starting_tracks:
                    start_pos = np.array(starting_track['detections'][0]['position'])
                    distance = np.linalg.norm(end_pos - start_pos)
                    
                    if distance <= max_distance:
                        nearby_starts.append(starting_track)
                
                # If exactly 2 tracks start near where 1 ended, likely a division
                if len(nearby_starts) == 2:
                    events['divisions'].append({
                        'parent_track': ending_track['track_id'],
                        'daughter_tracks': [t['track_id'] for t in nearby_starts],
                        'timepoint': t_idx
                    })
        
        return events

class AdvancedTrackingWidget(QWidget):
    """Widget for advanced cell tracking with lineage analysis"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_results = None
        self.tracking_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        
        # Layer selection
        layer_group = QGroupBox("Select Labels Layer (Time Series)")
        layer_layout = QVBoxLayout()
        
        self.layer_combo = QComboBox()
        self.update_layer_choices()
        layer_layout.addWidget(self.layer_combo)
        
        refresh_btn = QPushButton("Refresh Layers")
        refresh_btn.clicked.connect(self.update_layer_choices)
        layer_layout.addWidget(refresh_btn)
        
        layer_group.setLayout(layer_layout)
        main_layout.addWidget(layer_group)
        
        # Tracking parameters
        params_group = QGroupBox("Tracking Parameters")
        params_layout = QVBoxLayout()
        
        # Max distance
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Max distance:"))
        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(1.0, 500.0)
        self.max_distance_spin.setValue(50.0)
        self.max_distance_spin.setSuffix(" pixels")
        distance_layout.addWidget(self.max_distance_spin)
        params_layout.addLayout(distance_layout)
        
        # Max gap
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Max gap (frames):"))
        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(0, 10)
        self.max_gap_spin.setValue(2)
        gap_layout.addWidget(self.max_gap_spin)
        params_layout.addLayout(gap_layout)
        
        # Min track length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Min track length:"))
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 100)
        self.min_length_spin.setValue(3)
        self.min_length_spin.setSuffix(" frames")
        length_layout.addWidget(self.min_length_spin)
        params_layout.addLayout(length_layout)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Tracking options
        options_group = QGroupBox("Analysis Options")
        options_layout = QVBoxLayout()
        
        self.detect_divisions_check = QCheckBox("Detect cell divisions")
        self.detect_divisions_check.setChecked(True)
        options_layout.addWidget(self.detect_divisions_check)
        
        self.detect_merges_check = QCheckBox("Detect cell merges")
        self.detect_merges_check.setChecked(False)
        options_layout.addWidget(self.detect_merges_check)
        
        self.compute_velocities_check = QCheckBox("Compute velocities")
        self.compute_velocities_check.setChecked(True)
        options_layout.addWidget(self.compute_velocities_check)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Track button
        self.track_btn = QPushButton("Track Objects")
        self.track_btn.clicked.connect(self.start_tracking)
        self.track_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        main_layout.addWidget(self.track_btn)
        
        # Results splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Statistics table
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(QLabel("Track Statistics"))
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        stats_layout.addWidget(self.stats_table)
        stats_widget.setLayout(stats_layout)
        splitter.addWidget(stats_widget)
        
        # Lineage tree
        lineage_widget = QWidget()
        lineage_layout = QVBoxLayout()
        lineage_layout.addWidget(QLabel("Lineage Tree"))
        self.lineage_tree = QTreeWidget()
        self.lineage_tree.setHeaderLabels(["Track ID", "Info"])
        lineage_layout.addWidget(self.lineage_tree)
        lineage_widget.setLayout(lineage_layout)
        splitter.addWidget(lineage_widget)
        
        main_layout.addWidget(splitter)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_tracks_btn = QPushButton("Export Tracks")
        self.export_tracks_btn.clicked.connect(self.export_tracks)
        self.export_tracks_btn.setEnabled(False)
        export_layout.addWidget(self.export_tracks_btn)
        
        self.export_lineage_btn = QPushButton("Export Lineage")
        self.export_lineage_btn.clicked.connect(self.export_lineage)
        self.export_lineage_btn.setEnabled(False)
        export_layout.addWidget(self.export_lineage_btn)
        
        main_layout.addLayout(export_layout)
        
        # Status label
        self.status_label = QLabel("Select a labels layer to begin")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
        
        # Connect viewer events
        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)
    
    def update_layer_choices(self):
        """Update the layer selection combo box"""
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Labels) and layer.data.ndim >= 3:
                self.layer_combo.addItem(layer.name)
    
    def start_tracking(self):
        """Start tracking process"""
        layer_name = self.layer_combo.currentText()
        if not layer_name:
            self.status_label.setText("No layer selected")
            return
        
        # Get layer data
        layer = None
        for l in self.viewer.layers:
            if l.name == layer_name:
                layer = l
                break
        
        if layer is None or not isinstance(layer, Labels):
            self.status_label.setText("Invalid layer selection")
            return
        
        labels_data = layer.data
        
        # Extract timepoints (assuming first dimension is time)
        if labels_data.ndim < 3:
            self.status_label.setText("Layer must be 3D or higher (time dimension required)")
            return
        
        timepoints = [labels_data[t] for t in range(labels_data.shape[0])]
        
        # Prepare parameters
        parameters = {
            'max_distance': self.max_distance_spin.value(),
            'max_gap': self.max_gap_spin.value(),
            'min_track_length': self.min_length_spin.value(),
            'detect_divisions': self.detect_divisions_check.isChecked(),
            'detect_merges': self.detect_merges_check.isChecked()
        }
        
        # Start tracking thread
        self.tracking_thread = TrackingThread(timepoints, parameters)
        self.tracking_thread.progress.connect(self.update_progress)
        self.tracking_thread.finished.connect(self.on_tracking_finished)
        self.tracking_thread.error.connect(self.on_tracking_error)
        
        self.track_btn.setEnabled(False)
        self.status_label.setText("Tracking objects...")
        self.tracking_thread.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def on_tracking_finished(self, results):
        """Handle successful tracking completion"""
        self.current_results = results
        self.track_btn.setEnabled(True)
        self.export_tracks_btn.setEnabled(True)
        self.export_lineage_btn.setEnabled(True)
        
        # Display statistics
        stats = results['statistics']
        self.stats_table.setRowCount(0)
        
        stat_items = [
            ("Total Tracks", str(stats['total_tracks'])),
            ("Mean Track Length", f"{stats['mean_track_length']:.1f} frames"),
            ("Mean Displacement", f"{stats['mean_displacement']:.1f} pixels"),
            ("Mean Speed", f"{stats['mean_speed']:.2f} pixels/frame"),
            ("Mean Straightness", f"{stats['mean_straightness']:.3f}")
        ]
        
        for row, (prop, value) in enumerate(stat_items):
            self.stats_table.insertRow(row)
            self.stats_table.setItem(row, 0, QTableWidgetItem(prop))
            self.stats_table.setItem(row, 1, QTableWidgetItem(value))
        
        # Display lineage tree
        self.lineage_tree.clear()
        self._build_lineage_tree(results['tracks'], results['lineage_events'])
        
        # Convert tracks to napari format and add to viewer
        self._add_tracks_to_viewer(results['tracks'])
        
        self.status_label.setText(f"Tracking complete: {stats['total_tracks']} tracks found")
        self.status_label.setStyleSheet("color: green; font-style: italic;")
        self.progress_bar.setValue(100)
    
    def on_tracking_error(self, error_message):
        """Handle tracking errors"""
        self.track_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: red; font-style: italic;")
        self.progress_bar.setValue(0)
    
    def _build_lineage_tree(self, tracks, lineage_events):
        """Build lineage tree visualization"""
        # Create tree items for all tracks
        track_items = {}
        
        for track in tracks:
            track_id = track['track_id']
            info = f"{len(track['detections'])} frames"
            item = QTreeWidgetItem([f"Track {track_id}", info])
            track_items[track_id] = item
        
        # Organize by divisions
        parent_tracks = set()
        daughter_tracks = set()
        
        for division in lineage_events.get('divisions', []):
            parent_id = division['parent_track']
            daughter_ids = division['daughter_tracks']
            
            if parent_id in track_items:
                parent_item = track_items[parent_id]
                parent_tracks.add(parent_id)
                
                for daughter_id in daughter_ids:
                    if daughter_id in track_items:
                        daughter_item = track_items[daughter_id]
                        parent_item.addChild(daughter_item)
                        daughter_tracks.add(daughter_id)
        
        # Add root tracks (no parents) to tree
        for track_id, item in track_items.items():
            if track_id not in daughter_tracks:
                self.lineage_tree.addTopLevelItem(item)
        
        self.lineage_tree.expandAll()
    
    def _add_tracks_to_viewer(self, tracks):
        """Convert tracks to napari format and add to viewer"""
        # Convert to napari tracks format: [track_id, timepoint, y, x]
        track_data = []
        
        for track in tracks:
            track_id = track['track_id']
            for det in track['detections']:
                timepoint = det['timepoint']
                pos = det['position']
                if len(pos) == 2:
                    track_data.append([track_id, timepoint, pos[0], pos[1]])
                else:
                    # 3D data - use middle slice
                    track_data.append([track_id, timepoint, pos[1], pos[2]])
        
        if track_data:
            track_data = np.array(track_data)
            
            track_layer_name = f"Tracks_{self.layer_combo.currentText()}"
            if track_layer_name in [l.name for l in self.viewer.layers]:
                for layer in self.viewer.layers:
                    if layer.name == track_layer_name:
                        layer.data = track_data
                        break
            else:
                self.viewer.add_tracks(track_data, name=track_layer_name)
    
    def export_tracks(self):
        """Export tracks to CSV"""
        if self.current_results is None:
            return
        
        from PyQt6.QtWidgets import QFileDialog
        import csv
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Tracks",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Track_ID", "Timepoint", "Y", "X", "Area"])
                    
                    for track in self.current_results['tracks']:
                        track_id = track['track_id']
                        for det in track['detections']:
                            pos = det['position']
                            writer.writerow([
                                track_id,
                                det['timepoint'],
                                pos[0] if len(pos) > 0 else 0,
                                pos[1] if len(pos) > 1 else 0,
                                det.get('area', 0)
                            ])
                
                self.status_label.setText(f"Tracks exported to {filename}")
                self.status_label.setStyleSheet("color: green; font-style: italic;")
            except Exception as e:
                self.status_label.setText(f"Export failed: {str(e)}")
                self.status_label.setStyleSheet("color: red; font-style: italic;")
    
    def export_lineage(self):
        """Export lineage information"""
        if self.current_results is None:
            return
        
        from PyQt6.QtWidgets import QFileDialog
        import json
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Lineage",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                lineage_data = {
                    'divisions': self.current_results['lineage_events']['divisions'],
                    'tracks': [
                        {
                            'track_id': t['track_id'],
                            'length': len(t['detections']),
                            'start_time': t['detections'][0]['timepoint'],
                            'end_time': t['detections'][-1]['timepoint']
                        }
                        for t in self.current_results['tracks']
                    ]
                }
                
                with open(filename, 'w') as f:
                    json.dump(lineage_data, f, indent=2)
                
                self.status_label.setText(f"Lineage exported to {filename}")
                self.status_label.setStyleSheet("color: green; font-style: italic;")
            except Exception as e:
                self.status_label.setText(f"Export failed: {str(e)}")
                self.status_label.setStyleSheet("color: red; font-style: italic;")
