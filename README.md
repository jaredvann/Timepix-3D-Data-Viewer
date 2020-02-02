# Timepix Event Viewer

An interactive data viewer for Timepix3 data in 3 dimensions for Linux & macOS.

![Screenshot of the application](screenshot.png?raw=true)

Requires data in a format as output by [https://github.com/jaredvann/timepix-spidr-data-parser](https://github.com/jaredvann/timepix-spidr-data-parser).


## Python Requirements

- **matplotlib**
- **numpy**
- **python** >= 3.7
- **pyqtgraph**
- **pyqt5**
- **ffmpeg-python**
- **toml** (https://github.com/uiri/toml)


## Usage

```
python event_viewer.py <optional path to .bin data file>
```

## Authors

* **Jared Vann** - [jaredvann](https://github.com/jaredvann)

## Acknowledgements

The ARIADNE program is proudly supported by the European Research Council Grant No. 677927 and the University of Liverpool.