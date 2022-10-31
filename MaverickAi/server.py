from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import json
import cgi

import model_inference


class Server(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    # GET sends back a Hello world message
    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps({'msg': 'Please use this API with POST Method and appropriate parameters'}).encode('utf-8'))

    # POST echoes the message adding a JSON field
    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))

        # refuse to receive non-json content
        if ctype != 'application/json':
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'msg': 'Please set content-type to application/json'}).encode('utf-8'))
            return

        # read the message and convert it into a python dictionary
        length = int(self.headers.get('content-length'))
        try:
            message = json.loads(self.rfile.read(length))
        except ValueError:
            message = None

        if not message or 'prompt' not in message:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'msg': 'Please set the the correct parameters', 'example': {
                "prompt": "your prompt"
            }}).encode('utf-8'))
            return

        # do some stuff here ML
        result = model_inference.inference(message['prompt'])

        # send the message back
        self._set_headers()
        self.wfile.write(json.dumps(result).encode('utf-8'))


def run(server_class=HTTPServer, handler_class=Server, port=8008):
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler_class)

    print('Starting httpd on port %d...' % port)
    httpd.serve_forever()


if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
