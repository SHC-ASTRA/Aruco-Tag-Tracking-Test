# Aruco-Tag-Tracking-Test

## Description

This program uses opencv and aruco to detect `4X4_50` markers. Once it detects a marker, it draws a green rectange around the detected marker and a small red box in the top-left corner of the tag if it were oriented upright.

## Requirements

- python3.9
- pipenv
  - can be installed with `pip install pipenv`

## Getting Started

1. Clone the project.
2. `cd Aruco-Tag-Tracking-Test`
3. `pipenv install` - installs dependencies
4. `pipenv shell` - activates the environment
5. `python detect_markers` - run the script
