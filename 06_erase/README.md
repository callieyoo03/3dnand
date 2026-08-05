# 06_erase

MoS₂ GAA Charge-Trap Flash Memory의 erase 동작을 구현하는 폴더입니다.

## 구현 목표

- 음의 게이트 전압 sweep
- Tunnel oxide 전계 방향 확인
- Reverse Fowler–Nordheim tunneling 계산
- Trapped electron density 감소
- Charge-feedback 기반 erase time 계산
- Erase 이후 Id–Vg 및 threshold voltage 분석

## 기본 구조

- MoS₂ channel thickness: 2 nm
- Tunnel oxide: Al₂O₃, 3 nm
- Charge-trap layer: HfO₂, 5 nm
- Blocking oxide: Al₂O₃, 16 nm
- Channel length: 100 nm

## 초기 erase 모델

초기 구현에서는 음의 게이트 전압에 의해 HfO₂에 저장된 전자가
Al₂O₃ tunnel oxide를 통과하여 MoS₂ 채널 방향으로 방출되는
reverse Fowler–Nordheim tunneling을 가정합니다.

현재 모델은 이상화된 erase 모델이며 다음 현상은 포함하지 않습니다.

- GIDL-induced hole generation
- Hole injection
- Poole–Frenkel emission
- Trap-assisted tunneling
- Trap-to-trap hopping
- Erase saturation