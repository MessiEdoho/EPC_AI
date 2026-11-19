"""Start the EPC-AI web server: python EPC_backend.py

Settings come from environment variables (see epc/config.py and README.md).
"""
import os

from epc.server import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("EPC_PORT", "8000")), debug=False, threaded=True)
