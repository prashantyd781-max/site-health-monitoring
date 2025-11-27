#!/bin/bash
# Simple script to watch training in real-time

cd /Users/prashant/Documents/coding/demo

echo "=================================="
echo "  LIVE YOLOv8 TRAINING MONITOR"
echo "=================================="
echo ""
echo "Press Ctrl+C to stop watching"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Follow the last 50 lines and keep updating
tail -f -n 50 training_output.log

