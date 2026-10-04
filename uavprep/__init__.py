"""uavprep — 연구실 공통 UAV 결함 데이터 전처리 모듈 (datadef/PRINCIPLES.md 구현).

데이터셋마다 reader가 '비행 레코드'(공통 형식)를 내고, core가 원칙대로 시간축·구간·창·정규화를 처리해
train_eval.py가 읽는 npz + 정규화 파일 + 설명 파일(meta.json)을 만든다.
사용: python -m uavprep.build configs/alfa.json
"""
