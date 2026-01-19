#!/usr/bin/env python3.14
"""Standalone QGIS MCP Server - Uses QGIS Python bindings directly"""
import sys
import json
import socket
import signal
import threading

# Add QGIS Python path
sys.path.insert(0, '/usr/lib/python3.14/site-packages')

from qgis.core import (
    QgsApplication, QgsProject, QgsVectorLayer,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsPointXY, QgsDistanceArea, Qgis
)

class QGISMCPServer:
    def __init__(self, port=9877):
        self.port = port
        self.running = False
        self.qgs_app = None

    def initialize_qgis(self):
        """Initialize QGIS application"""
        QgsApplication.setPrefixPath('/usr', True)
        self.qgs_app = QgsApplication([], False)
        self.qgs_app.initQgis()
        print(f"QGIS initialized - Version: {Qgis.version()}")

    def cleanup_qgis(self):
        """Cleanup QGIS application"""
        if self.qgs_app:
            self.qgs_app.exitQgis()

    def process_request(self, request):
        """Process MCP request"""
        request_type = request.get('type', '')
        params = request.get('params', {})

        if request_type == 'health_check':
            return {
                "success": True,
                "status": "healthy",
                "qgis_version": Qgis.version(),
                "layers": len(QgsProject.instance().mapLayers())
            }
        elif request_type == 'list_layers':
            layers = QgsProject.instance().mapLayers()
            return {
                "success": True,
                "layers": [
                    {"name": layer.name(), "id": layer.id()}
                    for layer in layers.values()
                ]
            }
        elif request_type == 'calculate_distance':
            # Calculate distance between two points
            try:
                p1 = params.get('point1', [])
                p2 = params.get('point2', [])
                crs_code = params.get('crs', 'EPSG:4326')

                point1 = QgsPointXY(p1[0], p1[1])
                point2 = QgsPointXY(p2[0], p2[1])

                distance_area = QgsDistanceArea()
                crs = QgsCoordinateReferenceSystem(crs_code)
                distance_area.setSourceCrs(crs, QgsProject.instance().transformContext())
                distance_area.setEllipsoid(crs.ellipsoidAcronym())

                distance = distance_area.measureLine(point1, point2)

                return {
                    "success": True,
                    "distance_meters": distance,
                    "distance_km": distance / 1000.0
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif request_type == 'transform_coordinates':
            # Transform coordinates between CRS
            try:
                coords = params.get('coordinates', [])
                source_crs = params.get('source_crs', 'EPSG:4326')
                target_crs = params.get('target_crs', 'EPSG:3857')

                source = QgsCoordinateReferenceSystem(source_crs)
                target = QgsCoordinateReferenceSystem(target_crs)
                transform = QgsCoordinateTransform(source, target, QgsProject.instance())

                point = QgsPointXY(coords[0], coords[1])
                transformed = transform.transform(point)

                return {
                    "success": True,
                    "coordinates": [transformed.x(), transformed.y()],
                    "source_crs": source_crs,
                    "target_crs": target_crs
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {
                "success": False,
                "error": f"Unknown request type: {request_type}"
            }

    def handle_client(self, client_socket, address):
        """Handle a client connection"""
        try:
            print(f"Client connected from {address}")

            # Read request
            data = b''
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b'\n' in data:
                    break

            if not data:
                return

            request = json.loads(data.decode('utf-8'))
            print(f"Request: {request.get('type', 'unknown')}")

            # Process and send response
            response = self.process_request(request)
            response_json = json.dumps(response) + '\n'
            client_socket.sendall(response_json.encode('utf-8'))

        except Exception as e:
            print(f"Error handling client: {e}")
            error_response = {"success": False, "error": str(e)}
            try:
                client_socket.sendall(json.dumps(error_response).encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()

    def start(self):
        """Start the MCP server"""
        self.initialize_qgis()
        self.running = True

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(5)
        server_socket.settimeout(1.0)

        print(f"QGIS MCP Server listening on port {self.port}")
        print("Waiting for connections...")

        try:
            while self.running:
                try:
                    client_socket, address = server_socket.accept()
                    # Handle each client in a thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            server_socket.close()
            self.cleanup_qgis()

def signal_handler(sig, frame):
    """Handle SIGTERM/SIGINT"""
    print("\nReceived signal, shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    server = QGISMCPServer(port=9877)
    server.start()
