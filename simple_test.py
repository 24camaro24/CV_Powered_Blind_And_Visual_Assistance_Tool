#!/usr/bin/env python3
"""Simple test for pipeline"""
import sys
sys.path.insert(0, 'backend')

print('[1] Loading pipeline...')
from pipeline import NavigationPipeline

print('[2] Creating instance...')
p = NavigationPipeline()

print('[3] SUCCESS - Pipeline ready!')
