from flask import Flask, jsonify

app = Flask(__name__)

# Loads config
app.config.from_pyfile('config.cfg')


@app.route('/', methods=['GET'])
def index():
    return jsonify({})


if __name__ == '__main__':
    app.run(use_debugger=app.config['SERVER_DEBUG'],
            use_reloader=app.config['SERVER_RELOAD'],
            host=app.config['SERVER_HOST'],
            port=int(app.config['SERVER_PORT']),
            threaded=True)
