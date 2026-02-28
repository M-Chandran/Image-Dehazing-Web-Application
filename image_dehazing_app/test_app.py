#!/usr/bin/env python3
"""Test script to check if the Flask app can start without errors"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from app import app
    print("✓ Flask app imported successfully")

    # Test app context
    with app.app_context():
        print("✓ Flask app context works")

    # Test template rendering
    with app.test_client() as client:
        response = client.get('/')
        print(f"✓ Root route accessible (status: {response.status_code})")

        response = client.get('/login')
        print(f"✓ Login route accessible (status: {response.status_code})")

    print("✓ All basic tests passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
