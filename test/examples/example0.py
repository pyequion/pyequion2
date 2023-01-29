try:
    from pyequion2 import rungui
except ImportError:
    print('Please install gui extra')
    import sys
    sys.exit(1)

rungui()

#Windows Subsystem for Linux (WSL)
