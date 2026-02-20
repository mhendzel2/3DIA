"""
Simple Scientific Image Analyzer - Fallback Implementation
Basic image analysis capabilities without external dependencies
"""

import http.server
import socketserver
import json
import urllib.parse
import base64
import io
import struct
from pathlib import Path

# Global session storage
sessions = {}

class SimpleImageAnalyzer:
    """Simple image analyzer with basic functionality"""
    
    @staticmethod
    def create_test_image(width=512, height=512):
        """Create a simple test image"""
        # Create a simple gradient image
        image_data = []
        for y in range(height):
            row = []
            for x in range(width):
                # Create a circular gradient pattern
                center_x, center_y = width // 2, height // 2
                dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                max_dist = min(width, height) // 2
                intensity = max(0, min(255, int(255 * (1 - dist / max_dist))))
                row.append(intensity)
            image_data.append(row)
        return image_data
    
    @staticmethod
    def simple_threshold(image_data, threshold=128):
        """Simple thresholding segmentation"""
        height = len(image_data)
        width = len(image_data[0]) if height > 0 else 0
        
        labels = []
        for y in range(height):
            row = []
            for x in range(width):
                if image_data[y][x] > threshold:
                    row.append(1)
                else:
                    row.append(0)
            labels.append(row)
        return labels
    
    @staticmethod
    def measure_objects(labels, image_data=None):
        """Basic object measurements"""
        measurements = []
        
        # Count objects and calculate basic properties
        unique_labels = set()
        for row in labels:
            for val in row:
                if val > 0:
                    unique_labels.add(val)
        
        for label_id in unique_labels:
            area = sum(sum(1 for val in row if val == label_id) for row in labels)
            if image_data is not None:
                intensity_values = []
                for y, row in enumerate(labels):
                    for x, val in enumerate(row):
                        if val == label_id:
                            intensity_values.append(float(image_data[y][x]))
                mean_intensity = float(sum(intensity_values) / len(intensity_values)) if intensity_values else 0.0
            else:
                mean_intensity = 0.0
            measurements.append({
                'label': label_id,
                'area': area,
                'mean_intensity': mean_intensity
            })
        
        return measurements

class SimpleRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for simple analyzer"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_main_page()
        elif self.path == '/api/session/create':
            self.create_session()
        else:
            self.send_error(404, "Not found")
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/analyze':
            self.handle_analyze()
        else:
            self.send_error(404, "Not found")
    
    def serve_main_page(self):
        """Serve simple HTML interface"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simple Scientific Image Analyzer</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 800px; margin: 0 auto; }
                button { padding: 10px 20px; margin: 5px; }
                #output { margin-top: 20px; padding: 10px; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Simple Scientific Image Analyzer</h1>
                <p>Basic fallback implementation - no external dependencies required</p>
                
                <button onclick="createTestImage()">Create Test Image</button>
                <button onclick="analyzeImage()">Analyze Image</button>
                
                <div id="output"></div>
            </div>
            
            <script>
                let sessionId = null;
                
                // Create session on page load
                fetch('/api/session/create')
                    .then(response => response.json())
                    .then(data => {
                        sessionId = data.session_id;
                        document.getElementById('output').innerHTML = 'Session created: ' + sessionId;
                    });
                
                function createTestImage() {
                    document.getElementById('output').innerHTML = 'Test image created successfully';
                }
                
                function analyzeImage() {
                    fetch('/api/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({session_id: sessionId})
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('output').innerHTML = 
                            '<h3>Analysis Results:</h3>' +
                            '<p>Objects found: ' + data.num_objects + '</p>' +
                            '<p>Total area: ' + data.total_area + ' pixels</p>';
                    });
                }
            </script>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def create_session(self):
        """Create new analysis session"""
        import uuid
        session_id = str(uuid.uuid4())[:8]
        sessions[session_id] = {
            'image_data': SimpleImageAnalyzer.create_test_image(),
            'created': True
        }
        
        response = {'session_id': session_id, 'status': 'created'}
        self.send_json_response(response)
    
    def handle_analyze(self):
        """Handle analysis request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            session_id = data.get('session_id')
            if session_id not in sessions:
                self.send_error(400, "Invalid session")
                return
            
            # Perform simple analysis
            image_data = sessions[session_id]['image_data']
            labels = SimpleImageAnalyzer.simple_threshold(image_data)
            measurements = SimpleImageAnalyzer.measure_objects(labels, image_data=image_data)
            
            response = {
                'status': 'success',
                'num_objects': len(measurements),
                'total_area': sum(m['area'] for m in measurements),
                'measurements': measurements
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            self.send_error(500, f"Analysis failed: {e}")
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

def run_server(port=5000):
    """Run the simple analyzer server"""
    print("Simple Scientific Image Analyzer - Fallback Mode")
    print("=" * 50)
    print("Basic functionality only - no external dependencies")
    print(f"Server running on http://localhost:{port}")
    print("=" * 50)
    
    with socketserver.TCPServer(("", port), SimpleRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")

if __name__ == "__main__":
    run_server()
