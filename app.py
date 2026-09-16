"""
AI Model Risk & GenAI Governance Platform
Hugging Face Spaces Entry Point
Runs the Streamlit Enterprise Audit Dashboard with 10 Governance Tabs.
"""

import os
import sys
import runpy

# Set project root in sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Execute the primary dashboard application
dashboard_script = os.path.join(root_dir, "dashboard", "app.py")
runpy.run_path(dashboard_script, run_name="__main__")
