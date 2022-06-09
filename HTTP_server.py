from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import os
from mimetypes import types_map
import simplejson

from compute_channel import run_channel


class webserverHandler(BaseHTTPRequestHandler):
    """HTTP server implementation with GET and POST methods"""

    def do_GET(self):
        # GET method for the server
        try:
            curdir = os.getcwd()
            if self.path == "/":
                self.path = "index.html"
            fname,ext = os.path.splitext(self.path)
            if fname.startswith("/"):
                self.path = self.path[1::]
            if ext in (".html", ".css", ".js", ".png"):
                with open(os.path.join(curdir, self.path)) as f:
                    self.send_response(200)
                    self.send_header('Content-type', types_map[ext])
                    self.end_headers()
                    if ext != ".png":
                        self.wfile.write(bytes(f.read(), 'UTF-8'))
                    else:
                        ff = open(os.path.join(curdir, self.path), 'rb')
                        self.wfile.write(ff.read())
                        ff.close()

            return

        except IOError:
            self.send_error(404, "File not found %s" % self.path)

    def do_POST(self):
        # POST method for the server
        try:

            data_string = self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)

            # Load JSON to dict
            data = simplejson.loads(data_string)
            output = run_channel(data['car_freq'], data['rx_pos'], data['tx_pos'], 'LOS',
                                 data['num_clusters'], data['num_rays'])

            # Send the message back
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            output_b = simplejson.dumps({'figure': output})
            self.wfile.write(output_b.encode())

        except:
            # Send error message if did not succeed
            self.send_error(404, "{}".format(sys.exc_info()[0]))
            print(sys.exc_info())


def main():
    try:
        port = 8000
        server = HTTPServer(('', port), webserverHandler)
        print("Web server running on port %s" % port)
        server.serve_forever()

    except KeyboardInterrupt:
        print(" ^C entered stopping web server...")
        server.socket.close()


if __name__ == '__main__':
    main()
