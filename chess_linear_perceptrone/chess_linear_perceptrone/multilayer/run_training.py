#!/usr/bin/env python3
"""
다층 퍼셉트론 은닉층 4개 vs 8개 동시 학습 실행 스크립트

사용법:
1. 동시 학습: python run_training.py
2. 개별 학습: 
   - 은닉층 4개만: python run_training.py --hidden4
   - 은닉층 8개만: python run_training.py --hidden8
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='다층 퍼셉트론 학습 실행')
    parser.add_argument('--hidden4', action='store_true', help='은닉층 4개만 학습')
    parser.add_argument('--hidden8', action='store_true', help='은닉층 8개만 학습')
    
    args = parser.parse_args()
    
    if args.hidden4:
        print("=== 은닉층 4개 다층 퍼셉트론만 학습 시작 ===")
        os.system('python multilayer/hidden4/main_hidden4.py')
    elif args.hidden8:
        print("=== 은닉층 8개 다층 퍼셉트론만 학습 시작 ===")
        os.system('python multilayer/hidden8/main_hidden8.py')
    else:
        print("=== 은닉층 4개 vs 8개 다층 퍼셉트론 동시 학습 시작 ===")
        os.system('python multilayer/train_both_versions.py')

if __name__ == '__main__':
    main() 