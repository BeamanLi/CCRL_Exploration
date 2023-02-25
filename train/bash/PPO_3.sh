#!/bin/bash
cd ..
python3 PPO_main.py --seed 90 --device cuda:0
python3 CPPO_main.py --seed 90 --device cuda:0









