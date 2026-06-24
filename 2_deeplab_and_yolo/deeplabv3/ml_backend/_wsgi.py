import argparse
import json
import logging
import logging.config
import os

log_level = os.getenv("LOG_LEVEL", "INFO")

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "stream": "ext://sys.stdout",
                "formatter": "standard",
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
            "propagate": True,
        },
    }
)

from label_studio_ml.api import init_app

from model import DeepLabPersonBackend

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepLab Label Studio ML backend")
    parser.add_argument("-p", "--port", dest="port", type=int, default=9090)
    parser.add_argument("--host", dest="host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--kwargs",
        "--with",
        dest="kwargs",
        metavar="KEY=VAL",
        nargs="+",
        type=lambda kv: kv.split("=", 1),
    )
    parser.add_argument("-d", "--debug", dest="debug", action="store_true")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=log_level,
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=os.path.dirname(__file__),
    )
    parser.add_argument("--check", dest="check", action="store_true")
    parser.add_argument(
        "--basic-auth-user",
        default=os.environ.get("ML_SERVER_BASIC_AUTH_USER"),
    )
    parser.add_argument(
        "--basic-auth-pass",
        default=os.environ.get("ML_SERVER_BASIC_AUTH_PASS"),
    )
    args = parser.parse_args()

    if args.log_level:
        logging.root.setLevel(args.log_level)

    def parse_kwargs():
        params = {}
        if not args.kwargs:
            return params
        for k, v in args.kwargs:
            if v.isdigit():
                params[k] = int(v)
            elif v.lower() == "true":
                params[k] = True
            elif v.lower() == "false":
                params[k] = False
            else:
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v
        return params

    kwargs = get_kwargs_from_config()
    kwargs.update(parse_kwargs())

    if args.check:
        print(f'Check "{DeepLabPersonBackend.__name__}" instance creation...')
        DeepLabPersonBackend.preload()
        DeepLabPersonBackend(**kwargs)

    print("Preloading model before server start (may take 10-30 sec on CPU)...")
    DeepLabPersonBackend.preload()
    print(f"Starting ML backend on http://{args.host}:{args.port}")

    app = init_app(
        model_class=DeepLabPersonBackend,
        basic_auth_user=args.basic_auth_user,
        basic_auth_pass=args.basic_auth_pass,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
else:
    app = init_app(model_class=DeepLabPersonBackend)
