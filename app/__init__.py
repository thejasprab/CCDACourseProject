# app/__init__.py
from flask import Flask

def create_app():
    app = Flask(__name__)
    # If you want flash messages or secure cookies later, set a secret key:
    # app.config["SECRET_KEY"] = "change-me"
    from .server import register_routes
    register_routes(app)
    return app
