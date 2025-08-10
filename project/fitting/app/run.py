"""Entry point to launch the desktop calibration app."""

from project.fitting.app.calibration_ui import CalibrationApp


def main() -> None:
    app = CalibrationApp()
    app.run()


if __name__ == "__main__":
    main()


