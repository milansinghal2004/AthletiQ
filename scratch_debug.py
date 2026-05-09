import sys
import os
import app
import app.core.pipeline
import app.ui.dashboard_ui

print(f"App location: {app.__file__}")
print(f"Pipeline location: {app.core.pipeline.__file__}")
print(f"Dashboard UI location: {app.ui.dashboard_ui.__file__}")
print(f"sys.path: {sys.path}")
