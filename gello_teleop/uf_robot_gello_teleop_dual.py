#!/usr/bin/env python3
"""Run two GELLO leaders with a shared reset/alignment/follow lifecycle."""
import sys
from uf_robot_gello_teleop import main

if __name__ == '__main__':
    sys.exit(main(dual=True))
