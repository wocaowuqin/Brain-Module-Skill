#!/opt/ryu-venv/bin/python
"""Launch Ryu 4.34 with compatibility for modern Eventlet releases."""

import re
import sys

import eventlet.wsgi


if not hasattr(eventlet.wsgi, "ALREADY_HANDLED"):
    eventlet.wsgi.ALREADY_HANDLED = object()

from ryu.cmd.manager import main


if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(main())
