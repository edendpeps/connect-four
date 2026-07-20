커넥트포 공정 비교 실험 코드

비교 대상
- MCTS: 정책망 없는 일반 UCT MCTS
- MLP_MCTS: 동일한 UCT MCTS + 루트에서 MLP 정책망 1회
- CNN_MCTS: 동일한 UCT MCTS + 루트에서 CNN 정책망 1회

핵심 공정성 처리
1. 세 알고리즘이 완전히 같은 MCTS 구현을 사용합니다.
2. 고정 시뮬레이션 실험에서는 한 수당 탐색 횟수가 정확히 같습니다.
3. 고정 시간 실험에서는 정책망 추론 시간도 제한시간에 포함됩니다.
4. 2~6수 랜덤 오프닝을 생성하고 같은 오프닝에서 선후공을 교환합니다.
5. 42, 123, 777 세 시드를 사용합니다.
6. 매 수마다 독립적인 RNG를 새로 생성합니다.
   한 알고리즘이 난수를 많이 소비해도 상대나 다음 수에 영향을 주지 않습니다.

사용법
1. 이 폴더의 파일들을 같은 Visual Studio 프로젝트에 추가합니다.
2. 기존 connectfour.cpp는 빼고 이 폴더의 connectfour.cpp를 사용합니다.
3. 기존 MCTS.cpp, PolicyMCTS.cpp, Monte_Carlo.cpp는 이 실험에서 사용하지 않습니다.
   프로젝트에 남아 있어도 되지만, 기존 connectfour.cpp의 main과 중복되면 안 됩니다.
4. mlp_policy.bin과 cnn_policy.bin을 실행 파일의 작업 디렉터리에 둡니다.
5. x64 Release로 실행하는 것을 권장합니다.

상단 설정
- RUN_FIXED_SIMULATIONS: 동일 탐색량 실험 실행 여부
- RUN_FIXED_TIME: 동일 제한시간 실험 실행 여부
- FIXED_SIMULATIONS = 0:
  일반 MCTS가 100ms 동안 수행한 시뮬레이션 수의 중앙값을 자동 사용합니다.
- FIXED_TIME_MS = 100
- OPENINGS_PER_SEED = 50
  시드별 50개 오프닝 × 선후공 2판 = 대진당 시드별 100판입니다.

출력 파일
- fair_mcts_games.csv: 게임 하나마다 상세 결과
- fair_mcts_summary.csv: 시드별 결과와 세 시드 통합 결과

재현성
- 고정 시뮬레이션 실험은 같은 바이너리와 환경에서 같은 결과가 재현됩니다.
- 고정 시간 실험은 CPU 스케줄링에 따라 실제 시뮬레이션 수가 달라질 수 있어
  같은 시드를 사용해도 승패가 완전히 같지 않을 수 있습니다.

보고서 해석
- fixed_simulations: 정책 정보 자체가 같은 탐색량에서 도움이 되는지 비교
- fixed_time: 정책망 추론 비용까지 포함한 노트북 CPU 실전 성능 비교
