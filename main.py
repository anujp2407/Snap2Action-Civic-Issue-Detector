"""
main.py
-------
Entry point for Snap2Action_CV. Launches the CustomTkinter desktop app.

Run with:
    python main.py
"""

import database as db
from gui import Snap2ActionApp


def main():
    print("=" * 60)
    print(" Snap2Action - Civic Infrastructure Damage Detector")
    print(" Dual Engine: YOLOv8 Deep Learning + OpenCV Verification")
    print("=" * 60)
    db.init_db()
    app = Snap2ActionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
